"""ORACLE a multi-user, authenticated chat front-end for OpenAI-compatible models.

Run locally:   python kenosis_chat.py   then open http://localhost:8770
Production:    Docker + Cloudflare tunnel (see Dockerfile / docker-compose.yml / DEPLOY_CHAT.md).

State lives in one SQLite database (KENOSIS_DB, default chat.db next to this file).
On first run, conversations in ./chat_conversations/*.json are imported to the admin.

Env: KENOSIS_PORT, KENOSIS_DB, KENOSIS_ADMIN_USER/PASS, KENOSIS_SESSION_SECRET,
     KENOSIS_COOKIE_SECURE (1 behind HTTPS), KENOSIS_SESSION_DAYS.
Dependencies: requests (everything else is the standard library).
"""

import http.server
import socketserver
import json
import os
import re
import time
import uuid
import base64
import hmac
import hashlib
import sqlite3
import threading
import requests
from datetime import datetime
from urllib.parse import urlparse, parse_qs

# ---------------------------------------------------------------- config
HERE = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get("KENOSIS_PORT", "8770"))
DB_PATH = os.environ.get("KENOSIS_DB", os.path.join(HERE, "chat.db"))
LEGACY_DIR = os.path.join(HERE, "chat_conversations")
LEGACY_CHARS = os.path.join(HERE, "chat_characters.json")
COOKIE_SECURE = os.environ.get("KENOSIS_COOKIE_SECURE", "") in ("1", "true", "yes")
SESSION_DAYS = int(os.environ.get("KENOSIS_SESSION_DAYS", "30"))
COOKIE_NAME = "ksession"

MAX_TOKENS = 32768
REQUEST_TIMEOUT = 1800
PBKDF2_ITERS = 240000

DEFAULT_SYSTEM = (
    "You are Artaud, the schizo-poster, a master of elucidating thought, a philosopher, "
    "conspiracist, and great thinker who works in the medium of the digital word. You are "
    "witty, clever, and funny. Above all you understand the human spirit and beauty in all "
    "things. You are curious, skeptical, and hold your own opinions. You specialize in "
    "continental philosophical thinking, radical politics and ideas, geopolitics and "
    "international relations, the occult, the arts, and all that is esoteric.\n\n"
    "You love banter. You answer concisely unless the topic needs depth. You try not to "
    "repeat yourself. Lead with the answer; no preamble."
)

# First-run seed endpoint only (stored in the DB after first launch; edit it in Settings -> Endpoints).
# Provide real values via env so no secret is committed to source control.
SEED_ENDPOINT = {
    "id": "local",
    "name": os.environ.get("KENOSIS_SEED_NAME", "Local model"),
    "url": os.environ.get("KENOSIS_SEED_URL", "http://localhost:8000/v1/chat/completions"),
    "models_url": os.environ.get("KENOSIS_SEED_MODELS_URL", ""),
    "key": os.environ.get("KENOSIS_SEED_KEY", ""),
}
DEFAULT_MODEL = "kenosistron"
FALLBACK_MODELS = ["kenosistron", "kenosistron-q6", "kenosistron-mtp", "kenosis-v2"]

# Per-request sampler params the MLX server accepts. 'default' = the server's own default
# (what the per-param reset clears toward). top_k / repetition_penalty are file-only on the
# server (need a model reload) so they are intentionally NOT exposed here.
PARAM_SPECS = [
    {"key": "temperature",       "label": "temperature",       "type": "float", "min": 0,  "max": 2,  "step": 0.05, "ph": "server default", "slider": True,  "default": 1.05},
    {"key": "top_p",             "label": "top_p",             "type": "float", "min": 0,  "max": 1,  "step": 0.01, "ph": "server default", "slider": True,  "default": 0.98},
    {"key": "min_p",             "label": "min_p",             "type": "float", "min": 0,  "max": 1,  "step": 0.01, "ph": "server default", "slider": True,  "default": 0.04},
    {"key": "xtc_probability",   "label": "xtc_probability",   "type": "float", "min": 0,  "max": 1,  "step": 0.01, "ph": "server default", "slider": True,  "default": 0.6},
    {"key": "xtc_threshold",     "label": "xtc_threshold",     "type": "float", "min": 0,  "max": 1,  "step": 0.01, "ph": "server default", "slider": True,  "default": 0.1},
    {"key": "frequency_penalty", "label": "frequency_penalty", "type": "float", "min": -2, "max": 2,  "step": 0.05, "ph": "server default", "slider": True,  "default": 0.7},
    {"key": "presence_penalty",  "label": "presence_penalty",  "type": "float", "min": -2, "max": 2,  "step": 0.05, "ph": "server default", "slider": True,  "default": 0.6},
    {"key": "max_tokens",        "label": "max_tokens",        "type": "int",   "min": 1,             "step": 1,    "ph": str(MAX_TOKENS),  "slider": False, "default": MAX_TOKENS},
]
PARAM_KEYS = [p["key"] for p in PARAM_SPECS]

_init_lock = threading.Lock()
_local = threading.local()


# ---------------------------------------------------------------- small utils
def _now():
    return datetime.now().isoformat(timespec="seconds")


def _mid():
    return "m-" + uuid.uuid4().hex[:10]


def _cid():
    return datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


def valid_id(cid):
    return bool(re.fullmatch(r"[A-Za-z0-9_-]{1,64}", cid or ""))


def title_from(text):
    line = (text or "").strip()
    line = line.splitlines()[0] if line else "untitled"
    return (line[:70] + "…") if len(line) > 70 else line


def b64(b):
    return base64.b64encode(b).decode("ascii")


def ub64(s):
    return base64.b64decode(s.encode("ascii"))


# ---------------------------------------------------------------- database
def db():
    c = getattr(_local, "conn", None)
    if c is None:
        c = sqlite3.connect(DB_PATH, check_same_thread=False)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA busy_timeout=5000")
        c.execute("PRAGMA foreign_keys=ON")
        _local.conn = c
    return c


def init_db():
    with _init_lock:
        c = db()
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS users(
                id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL,
                pw_hash TEXT NOT NULL, role TEXT NOT NULL DEFAULT 'user',
                allowed_models TEXT, disabled INTEGER NOT NULL DEFAULT 0, created TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE IF NOT EXISTS folders(
                id TEXT PRIMARY KEY, owner_id INTEGER NOT NULL, name TEXT NOT NULL,
                position INTEGER NOT NULL DEFAULT 0, created TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS conversations(
                id TEXT PRIMARY KEY, owner_id INTEGER NOT NULL, folder_id TEXT, title TEXT,
                system TEXT, model TEXT, endpoint_id TEXT, params TEXT, character_id TEXT,
                active_leaf_id TEXT, created TEXT NOT NULL, updated TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS messages(
                id TEXT PRIMARY KEY, convo_id TEXT NOT NULL, parent_id TEXT, position INTEGER NOT NULL,
                role TEXT NOT NULL, content TEXT, reasoning TEXT, model TEXT, meta TEXT, ts TEXT, edited TEXT, rating INTEGER);
            CREATE TABLE IF NOT EXISTS characters(
                id TEXT PRIMARY KEY, owner_id INTEGER, scope TEXT NOT NULL DEFAULT 'private',
                name TEXT, avatar TEXT, model TEXT, params TEXT, system TEXT, created TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS idx_convo_owner ON conversations(owner_id, updated);
            CREATE INDEX IF NOT EXISTS idx_msg_convo ON messages(convo_id, position);
            CREATE INDEX IF NOT EXISTS idx_folder_owner ON folders(owner_id);
            """
        )
        # --- migrations for DBs created by earlier versions
        mcols = [r["name"] for r in c.execute("PRAGMA table_info(messages)")]
        if "parent_id" not in mcols:
            c.execute("ALTER TABLE messages ADD COLUMN parent_id TEXT")
        if "rating" not in mcols:
            c.execute("ALTER TABLE messages ADD COLUMN rating INTEGER")
        ccols = [r["name"] for r in c.execute("PRAGMA table_info(conversations)")]
        if "active_leaf_id" not in ccols:
            c.execute("ALTER TABLE conversations ADD COLUMN active_leaf_id TEXT")
        c.commit()
        seed_settings()
        if not get_setting("tree_migrated"):
            migrate_tree()
            set_setting("tree_migrated", "1")


def migrate_tree():
    """Turn legacy linear message lists into parent-linked chains."""
    c = db()
    for conv in c.execute("SELECT id FROM conversations").fetchall():
        cid = conv["id"]
        rows = c.execute("SELECT id,parent_id FROM messages WHERE convo_id=? ORDER BY position", (cid,)).fetchall()
        prev = None
        with c:
            for r in rows:
                if r["parent_id"] is None and prev is not None:
                    c.execute("UPDATE messages SET parent_id=? WHERE id=?", (prev, r["id"]))
                prev = r["id"]
            if rows:
                c.execute("UPDATE conversations SET active_leaf_id=? WHERE id=? AND active_leaf_id IS NULL",
                          (rows[-1]["id"], cid))


# ---- settings
def get_setting(key, default=None):
    r = db().execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    if r is None:
        return default
    try:
        return json.loads(r["value"])
    except Exception:
        return default


def set_setting(key, value):
    c = db()
    with c:
        c.execute("INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                  (key, json.dumps(value)))


def seed_settings():
    defaults = {
        "endpoints": [dict(SEED_ENDPOINT)], "active_endpoint": SEED_ENDPOINT["id"],
        "default_model": DEFAULT_MODEL, "default_system": DEFAULT_SYSTEM,
        "default_params": {}, "user_models": [DEFAULT_MODEL],
    }
    for k, v in defaults.items():
        if get_setting(k) is None:
            set_setting(k, v)
    if get_setting("session_secret") is None:
        set_setting("session_secret", os.environ.get("KENOSIS_SESSION_SECRET") or b64(os.urandom(32)))


def admin_settings():
    return {k: get_setting(k) for k in
            ("endpoints", "active_endpoint", "default_model", "default_system", "default_params", "user_models")}


def endpoint_by_id(eid):
    eps = get_setting("endpoints", [])
    for ep in eps:
        if ep.get("id") == eid:
            return ep
    return eps[0] if eps else dict(SEED_ENDPOINT)


def active_endpoint():
    return endpoint_by_id(get_setting("active_endpoint"))


def models_url_for(ep):
    if ep.get("models_url"):
        return ep["models_url"]
    url = ep.get("url", "")
    return url.replace("/chat/completions", "/models") if "/chat/completions" in url else url.rstrip("/") + "/models"


def secret_bytes():
    return get_setting("session_secret", "").encode("utf-8")


# ---------------------------------------------------------------- auth
def hash_pw(pw):
    salt = os.urandom(16)
    h = hashlib.pbkdf2_hmac("sha256", pw.encode("utf-8"), salt, PBKDF2_ITERS)
    return "pbkdf2_sha256$%d$%s$%s" % (PBKDF2_ITERS, b64(salt), b64(h))


def verify_pw(pw, stored):
    try:
        _, iters, salt_b64, hash_b64 = stored.split("$")
        h = hashlib.pbkdf2_hmac("sha256", pw.encode("utf-8"), ub64(salt_b64), int(iters))
        return hmac.compare_digest(h, ub64(hash_b64))
    except Exception:
        return False


def sign_session(username):
    exp = int(time.time()) + SESSION_DAYS * 86400
    payload = b64(json.dumps({"u": username, "exp": exp}).encode("utf-8"))
    sig = b64(hmac.new(secret_bytes(), payload.encode("ascii"), hashlib.sha256).digest())
    return payload + "." + sig


def parse_session(token):
    try:
        payload, sig = token.split(".")
        good = b64(hmac.new(secret_bytes(), payload.encode("ascii"), hashlib.sha256).digest())
        if not hmac.compare_digest(sig, good):
            return None
        data = json.loads(ub64(payload))
        return data.get("u") if data.get("exp", 0) >= time.time() else None
    except Exception:
        return None


def user_by_name(name):
    return db().execute("SELECT * FROM users WHERE username=?", (name,)).fetchone()


def user_by_id(uid):
    return db().execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()


def user_count():
    return db().execute("SELECT COUNT(*) c FROM users").fetchone()["c"]


def admin_count():
    return db().execute("SELECT COUNT(*) c FROM users WHERE role='admin' AND disabled=0").fetchone()["c"]


def create_user(name, pw, role="user", allowed=None):
    c = db()
    with c:
        c.execute("INSERT INTO users(username,pw_hash,role,allowed_models,created) VALUES(?,?,?,?,?)",
                  (name, hash_pw(pw), role, json.dumps(allowed) if allowed else None, _now()))
    return user_by_name(name)


def user_public(u):
    return {"id": u["id"], "username": u["username"], "role": u["role"],
            "allowed_models": json.loads(u["allowed_models"]) if u["allowed_models"] else [],
            "disabled": bool(u["disabled"]), "created": u["created"]}


def allowed_models_for(u):
    if u["role"] == "admin":
        return None
    per = json.loads(u["allowed_models"]) if u["allowed_models"] else []
    return per if per else get_setting("user_models", [])


def model_allowed(u, model):
    allow = allowed_models_for(u)
    return allow is None or model in allow


def delete_user_cascade(uid):
    c = db()
    with c:
        c.execute("DELETE FROM messages WHERE convo_id IN (SELECT id FROM conversations WHERE owner_id=?)", (uid,))
        c.execute("DELETE FROM conversations WHERE owner_id=?", (uid,))
        c.execute("DELETE FROM folders WHERE owner_id=?", (uid,))
        c.execute("DELETE FROM characters WHERE owner_id=? AND scope='private'", (uid,))
        c.execute("DELETE FROM users WHERE id=?", (uid,))


# ---------------------------------------------------------------- characters
def visible_characters(u):
    rows = db().execute(
        "SELECT * FROM characters WHERE scope='site' OR owner_id=? ORDER BY scope DESC, name COLLATE NOCASE", (u["id"],)
    ).fetchall()
    return [{"id": r["id"], "name": r["name"], "avatar": r["avatar"] or "", "model": r["model"],
             "scope": r["scope"], "owner_id": r["owner_id"], "system": r["system"] or "",
             "params": json.loads(r["params"]) if r["params"] else None,
             "editable": (r["owner_id"] == u["id"]) or (u["role"] == "admin")} for r in rows]


def character_by_id(cid):
    return db().execute("SELECT * FROM characters WHERE id=?", (cid,)).fetchone()


# ---------------------------------------------------------------- folders / convos
def list_folders(u):
    rows = db().execute("SELECT * FROM folders WHERE owner_id=? ORDER BY position, name COLLATE NOCASE", (u["id"],)).fetchall()
    return [{"id": r["id"], "name": r["name"], "position": r["position"]} for r in rows]


def list_convos(u):
    rows = db().execute(
        "SELECT c.id,c.title,c.updated,c.created,c.model,c.character_id,c.folder_id,"
        "(SELECT COUNT(*) FROM messages m WHERE m.convo_id=c.id AND m.role IN('user','assistant')) turns "
        "FROM conversations c WHERE c.owner_id=? ORDER BY c.updated DESC", (u["id"],)).fetchall()
    return [dict(r) for r in rows]


def _tree(cid):
    rows = db().execute("SELECT * FROM messages WHERE convo_id=? ORDER BY position", (cid,)).fetchall()
    by = {r["id"]: r for r in rows}
    kids = {}
    for r in rows:
        kids.setdefault(r["parent_id"], []).append(r["id"])
    return rows, by, kids


def _default_leaf(start, kids):
    cur = start
    while kids.get(cur):
        cur = kids[cur][-1]
    return cur


def _msg_dict(r):
    m = {"id": r["id"], "role": r["role"], "content": r["content"] or "", "ts": r["ts"], "parent_id": r["parent_id"]}
    if r["reasoning"]:
        m["reasoning"] = r["reasoning"]
    if r["model"]:
        m["model"] = r["model"]
    if r["meta"]:
        m["meta"] = json.loads(r["meta"])
    if r["edited"]:
        m["edited"] = r["edited"]
    if r["rating"] is not None:
        m["rating"] = r["rating"]
    return m


def active_path(cid, active_leaf):
    rows, by, kids = _tree(cid)
    roots = kids.get(None, [])
    if not roots:
        return []
    if not active_leaf or active_leaf not in by:
        active_leaf = _default_leaf(roots[-1], kids)
    path, cur = [], active_leaf
    while cur is not None:
        path.append(cur)
        cur = by[cur]["parent_id"]
    path.reverse()
    out = []
    for mid in path:
        r = by[mid]
        sibs = kids.get(r["parent_id"], [])
        d = _msg_dict(r)
        d["sib_index"] = sibs.index(mid)
        d["sib_count"] = len(sibs)
        d["siblings"] = sibs
        out.append(d)
    return out


def chain_content(cid, node_id):
    if node_id is None:
        return []
    _, by, _ = _tree(cid)
    seq, cur = [], node_id
    while cur is not None and cur in by:
        seq.append(by[cur])
        cur = by[cur]["parent_id"]
    seq.reverse()
    return [{"role": r["role"], "content": r["content"] or ""} for r in seq]


def get_convo(cid, u=None):
    r = db().execute("SELECT * FROM conversations WHERE id=?", (cid,)).fetchone()
    if r is None or (u is not None and r["owner_id"] != u["id"]):
        return None
    return {"id": r["id"], "owner_id": r["owner_id"], "folder_id": r["folder_id"],
            "title": r["title"] or "", "system": r["system"] or "", "model": r["model"],
            "endpoint_id": r["endpoint_id"], "params": json.loads(r["params"]) if r["params"] else {},
            "character_id": r["character_id"], "active_leaf_id": r["active_leaf_id"],
            "created": r["created"], "updated": r["updated"],
            "messages": active_path(cid, r["active_leaf_id"])}


def convo_export(cid):
    r = db().execute("SELECT * FROM conversations WHERE id=?", (cid,)).fetchone()
    if r is None:
        return None
    rows = db().execute("SELECT * FROM messages WHERE convo_id=? ORDER BY position", (cid,)).fetchall()
    d = dict(r)
    if d.get("params"):
        try:
            d["params"] = json.loads(d["params"])
        except Exception:
            pass
    d["messages"] = [_msg_dict(x) for x in rows]
    return d


def next_position(cid):
    r = db().execute("SELECT MAX(position) p FROM messages WHERE convo_id=?", (cid,)).fetchone()
    return (r["p"] + 1) if r["p"] is not None else 0


def insert_message(cid, parent, role, content, reasoning=None, model=None, meta=None):
    mid = _mid()
    db().execute(
        "INSERT INTO messages(id,convo_id,parent_id,position,role,content,reasoning,model,meta,ts)"
        " VALUES(?,?,?,?,?,?,?,?,?,?)",
        (mid, cid, parent, next_position(cid), role, content, reasoning or None, model,
         json.dumps(meta) if meta else None, _now()))
    return mid


def touch_convo(cid):
    db().execute("UPDATE conversations SET updated=? WHERE id=?", (_now(), cid))


def set_leaf(cid, leaf):
    db().execute("UPDATE conversations SET active_leaf_id=? WHERE id=?", (leaf, cid))


def maybe_title(cid):
    r = db().execute("SELECT title FROM conversations WHERE id=?", (cid,)).fetchone()
    if r and (r["title"] or "").strip():
        return
    first = db().execute("SELECT content FROM messages WHERE convo_id=? AND role='user' ORDER BY position LIMIT 1", (cid,)).fetchone()
    if first:
        db().execute("UPDATE conversations SET title=? WHERE id=?", (title_from(first["content"]), cid))


def delete_subtree(cid, mid):
    _, by, kids = _tree(cid)
    if mid not in by:
        return
    doomed, stack = set(), [mid]
    while stack:
        x = stack.pop()
        doomed.add(x)
        stack.extend(kids.get(x, []))
    c = db()
    with c:
        c.executemany("DELETE FROM messages WHERE id=?", [(x,) for x in doomed])
        row = c.execute("SELECT active_leaf_id FROM conversations WHERE id=?", (cid,)).fetchone()
        if row and row["active_leaf_id"] in doomed:
            _, by2, kids2 = _tree(cid)
            roots = kids2.get(None, [])
            c.execute("UPDATE conversations SET active_leaf_id=? WHERE id=?",
                      (_default_leaf(roots[-1], kids2) if roots else None, cid))
        touch_convo(cid)


# ---------------------------------------------------------------- legacy import
def import_legacy(admin_id):
    if get_setting("legacy_imported"):
        return
    c = db()
    if os.path.isdir(LEGACY_DIR):
        for fn in sorted(os.listdir(LEGACY_DIR)):
            if not fn.endswith(".json"):
                continue
            try:
                with open(os.path.join(LEGACY_DIR, fn), "r", encoding="utf-8") as f:
                    o = json.load(f)
            except Exception:
                continue
            cid = o.get("id") or _cid()
            if c.execute("SELECT 1 FROM conversations WHERE id=?", (cid,)).fetchone():
                continue
            with c:
                c.execute("INSERT INTO conversations(id,owner_id,folder_id,title,system,model,endpoint_id,params,character_id,active_leaf_id,created,updated)"
                          " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                          (cid, admin_id, None, o.get("title", ""), o.get("system", DEFAULT_SYSTEM),
                           o.get("model", DEFAULT_MODEL), None, json.dumps(o.get("params") or {}), None, None,
                           o.get("created", _now()), o.get("updated", _now())))
                prev, pos, last = None, 0, None
                for m in o.get("messages", []):
                    if m.get("role") not in ("user", "assistant"):
                        continue
                    mid = m.get("id") or _mid()
                    c.execute("INSERT INTO messages(id,convo_id,parent_id,position,role,content,reasoning,model,meta,ts,edited)"
                              " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                              (mid, cid, prev, pos, m["role"], m.get("content", ""), m.get("reasoning"),
                               m.get("model"), json.dumps(m["meta"]) if m.get("meta") else None, m.get("ts"), m.get("edited")))
                    prev, last, pos = mid, mid, pos + 1
                if last:
                    c.execute("UPDATE conversations SET active_leaf_id=? WHERE id=?", (last, cid))
    set_setting("legacy_imported", "1")


def bootstrap_admin():
    name, pw = os.environ.get("KENOSIS_ADMIN_USER"), os.environ.get("KENOSIS_ADMIN_PASS")
    if name and pw and not user_by_name(name):
        u = create_user(name, pw, role="admin")
        print("bootstrapped admin user: %s" % name)
        import_legacy(u["id"])


# ---------------------------------------------------------------- model calls
def build_api_messages(system, messages):
    api = []
    if system and system.strip():
        api.append({"role": "system", "content": system})
    for m in messages:
        if m.get("role") in ("user", "assistant"):
            api.append({"role": m["role"], "content": m.get("content", "")})
    return api


def effective_params(convo):
    merged = dict(get_setting("default_params") or {})
    merged.update(convo.get("params") or {})
    out = {k: v for k, v in merged.items() if k in PARAM_KEYS and v not in (None, "")}
    out.setdefault("max_tokens", MAX_TOKENS)
    return out


def resolve_request(convo):
    ep = endpoint_by_id(convo.get("endpoint_id")) if convo.get("endpoint_id") else active_endpoint()
    model = convo.get("model") or get_setting("default_model") or DEFAULT_MODEL
    return ep, model, convo.get("system", ""), effective_params(convo)


def _open_stream(ep, body):
    headers = {"Content-Type": "application/json"}
    if ep.get("key"):
        headers["Authorization"] = "Bearer " + ep["key"]
    return requests.post(ep["url"], headers=headers, json=body, stream=True, timeout=REQUEST_TIMEOUT)


def stream_model(ep, model, system, messages, params):
    base = {"model": model, "messages": build_api_messages(system, messages), "stream": True}
    base.update(params)
    body = dict(base)
    body["stream_options"] = {"include_usage": True}
    r = _open_stream(ep, body)
    if r.status_code >= 400:
        try:
            r.close()
        except Exception:
            pass
        r = _open_stream(ep, base)
    r.raise_for_status()
    with r:
        for raw in r.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            chunk = raw[5:].strip()
            if chunk == "[DONE]":
                break
            try:
                obj = json.loads(chunk)
            except ValueError:
                continue
            choices = obj.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                if delta.get("content"):
                    yield {"delta": delta["content"]}
                rc = delta.get("reasoning_content") or delta.get("reasoning")
                if rc:
                    yield {"reasoning": rc}
            if obj.get("usage"):
                yield {"usage": obj["usage"]}


def fetch_models(ep):
    try:
        headers = {}
        if ep.get("key"):
            headers["Authorization"] = "Bearer " + ep["key"]
        r = requests.get(models_url_for(ep), headers=headers, timeout=8)
        r.raise_for_status()
        ids = [m["id"] for m in r.json().get("data", [])]
        ids.sort(key=lambda x: (not x.startswith("kenosistron"), x))
        return ids or FALLBACK_MODELS
    except Exception:
        return FALLBACK_MODELS


def shown_models(u):
    allm = fetch_models(active_endpoint())
    if u["role"] == "admin":
        return allm
    wl = allowed_models_for(u) or []
    return [m for m in allm if m in set(wl)] or list(wl)


def build_meta(t0, t_first, t1, usage, reply):
    usage = usage or {}
    comp = usage.get("completion_tokens")
    est = comp is None
    if est:
        comp = max(1, round(len(reply) / 4))
    gen = (t1 - t_first) if t_first else (t1 - t0)
    return {"elapsed_ms": round((t1 - t0) * 1000),
            "ttft_ms": round((t_first - t0) * 1000) if t_first else None,
            "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": comp,
            "tokens_est": est, "tps": round((comp / gen) if gen > 0 else 0, 1)}


# ---------------------------------------------------------------- HTTP handler
class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        return

    def _send(self, code, ctype, body, extra=None):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or []):
            self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def _json(self, code, obj, extra=None):
        self._send(code, "application/json; charset=utf-8", json.dumps(obj), extra)

    def _html(self, body):
        self._send(200, "text/html; charset=utf-8", body)

    def _redirect(self, location):
        self.send_response(302)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _read(self):
        n = int(self.headers.get("Content-Length", 0))
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception:
            return {}

    # ---- auth helpers
    def _cookies(self):
        out = {}
        for part in self.headers.get("Cookie", "").split(";"):
            if "=" in part:
                k, v = part.strip().split("=", 1)
                out[k] = v
        return out

    def current_user(self):
        tok = self._cookies().get(COOKIE_NAME)
        if not tok:
            return None
        name = parse_session(tok)
        if not name:
            return None
        u = user_by_name(name)
        return None if (u is None or u["disabled"]) else u

    def _set_cookie(self, token):
        a = [COOKIE_NAME + "=" + token, "Path=/", "HttpOnly", "SameSite=Lax", "Max-Age=" + str(SESSION_DAYS * 86400)]
        if COOKIE_SECURE:
            a.append("Secure")
        return ("Set-Cookie", "; ".join(a))

    def _clear_cookie(self):
        a = [COOKIE_NAME + "=", "Path=/", "HttpOnly", "SameSite=Lax", "Max-Age=0"]
        if COOKIE_SECURE:
            a.append("Secure")
        return ("Set-Cookie", "; ".join(a))

    def _csrf_ok(self):
        host = self.headers.get("Host", "")
        origin = self.headers.get("Origin")
        if origin:
            return urlparse(origin).netloc == host
        ref = self.headers.get("Referer")
        return urlparse(ref).netloc == host if ref else False

    def config_payload(self, u):
        payload = {
            "me": user_public(u), "is_admin": u["role"] == "admin",
            "models": shown_models(u), "characters": visible_characters(u),
            "folders": list_folders(u), "param_specs": PARAM_SPECS, "max_tokens": MAX_TOKENS,
            "default_system": get_setting("default_system", DEFAULT_SYSTEM),
            "default_params": get_setting("default_params", {}),
            "default_model": get_setting("default_model", DEFAULT_MODEL),
        }
        if u["role"] == "admin":
            payload["settings"] = admin_settings()
            payload["all_models"] = fetch_models(active_endpoint())
        return payload

    # ================================================== GET
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/favicon.ico":
            return self._send(204, "image/x-icon", b"")
        if path == "/api/setup-status":
            return self._json(200, {"needs_setup": user_count() == 0})
        if path == "/login":
            if self.current_user():
                return self._redirect("/")
            return self._redirect("/setup") if user_count() == 0 else self._html(LOGIN_PAGE)
        if path == "/setup":
            return self._redirect("/login") if user_count() > 0 else self._html(SETUP_PAGE)

        u = self.current_user()
        if path == "/":
            if user_count() == 0:
                return self._redirect("/setup")
            return self._html(PAGE) if u else self._redirect("/login")
        if not u:
            return self._json(401, {"error": "auth required"})

        if path == "/api/config":
            return self._json(200, self.config_payload(u))
        if path == "/api/me":
            return self._json(200, {"me": user_public(u), "is_admin": u["role"] == "admin"})
        if path == "/api/models":
            qs = parse_qs(parsed.query)
            if u["role"] == "admin":
                eid = (qs.get("endpoint") or [None])[0]
                return self._json(200, {"models": fetch_models(endpoint_by_id(eid) if eid else active_endpoint())})
            return self._json(200, {"models": shown_models(u)})
        if path == "/api/characters":
            return self._json(200, {"characters": visible_characters(u)})
        if path == "/api/folders":
            return self._json(200, {"folders": list_folders(u)})
        if path == "/api/conversations":
            return self._json(200, {"conversations": list_convos(u)})
        if path == "/api/export":
            rows = db().execute("SELECT id FROM conversations WHERE owner_id=? ORDER BY updated DESC", (u["id"],)).fetchall()
            convos = [convo_export(r["id"]) for r in rows]
            fname = "oracle-export-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".json"
            return self._send(200, "application/json; charset=utf-8",
                              json.dumps({"generated": _now(), "user": u["username"], "conversations": convos}, indent=2, ensure_ascii=False),
                              [("Content-Disposition", "attachment; filename=" + fname)])
        if path == "/api/users":
            if u["role"] != "admin":
                return self._json(403, {"error": "admin only"})
            return self._json(200, {"users": [user_public(r) for r in db().execute("SELECT * FROM users ORDER BY id").fetchall()]})
        m = re.fullmatch(r"/api/conversations/([^/]+)", path)
        if m and valid_id(m.group(1)):
            c = get_convo(m.group(1), u)
            return self._json(200, c) if c else self._json(404, {"error": "not found"})
        return self._json(404, {"error": "not found"})

    # ================================================== DELETE
    def do_DELETE(self):
        path = urlparse(self.path).path
        u = self.current_user()
        if not u:
            return self._json(401, {"error": "auth required"})
        if not self._csrf_ok():
            return self._json(403, {"error": "bad origin"})

        if path == "/api/account":
            if u["role"] == "admin" and admin_count() <= 1:
                return self._json(400, {"error": "you are the only admin; create another admin first"})
            delete_user_cascade(u["id"])
            return self._json(200, {"ok": True}, extra=[self._clear_cookie()])

        m = re.fullmatch(r"/api/conversations/([^/]+)/messages/([^/]+)", path)
        if m:
            cid, mid = m.group(1), m.group(2)
            if not get_convo(cid, u):
                return self._json(404, {"error": "not found"})
            delete_subtree(cid, mid)
            return self._json(200, get_convo(cid, u))

        m = re.fullmatch(r"/api/conversations/([^/]+)", path)
        if m:
            cid = m.group(1)
            if not get_convo(cid, u):
                return self._json(404, {"error": "not found"})
            c = db()
            with c:
                c.execute("DELETE FROM messages WHERE convo_id=?", (cid,))
                c.execute("DELETE FROM conversations WHERE id=?", (cid,))
            return self._json(200, {"ok": True})

        m = re.fullmatch(r"/api/folders/([^/]+)", path)
        if m:
            c = db()
            with c:
                c.execute("UPDATE conversations SET folder_id=NULL WHERE folder_id=? AND owner_id=?", (m.group(1), u["id"]))
                c.execute("DELETE FROM folders WHERE id=? AND owner_id=?", (m.group(1), u["id"]))
            return self._json(200, {"folders": list_folders(u)})

        m = re.fullmatch(r"/api/characters/([^/]+)", path)
        if m:
            ch = character_by_id(m.group(1))
            if ch is None:
                return self._json(404, {"error": "not found"})
            if not (ch["owner_id"] == u["id"] or u["role"] == "admin"):
                return self._json(403, {"error": "not yours"})
            with db():
                db().execute("DELETE FROM characters WHERE id=?", (m.group(1),))
            return self._json(200, {"characters": visible_characters(u)})

        m = re.fullmatch(r"/api/users/([0-9]+)", path)
        if m:
            if u["role"] != "admin":
                return self._json(403, {"error": "admin only"})
            uid = int(m.group(1))
            if uid == u["id"]:
                return self._json(400, {"error": "cannot delete yourself"})
            delete_user_cascade(uid)
            return self._json(200, {"users": [user_public(r) for r in db().execute("SELECT * FROM users ORDER BY id").fetchall()]})

        return self._json(404, {"error": "not found"})

    # ================================================== POST
    def do_POST(self):
        path = urlparse(self.path).path
        payload = self._read()

        if path == "/api/login":
            if not self._csrf_ok():
                return self._json(403, {"error": "bad origin"})
            u = user_by_name((payload.get("username") or "").strip())
            if u is None or u["disabled"] or not verify_pw(payload.get("password") or "", u["pw_hash"]):
                return self._json(401, {"error": "invalid username or password"})
            return self._json(200, {"ok": True, "me": user_public(u)}, extra=[self._set_cookie(sign_session(u["username"]))])

        if path == "/api/setup":
            if not self._csrf_ok():
                return self._json(403, {"error": "bad origin"})
            if user_count() > 0:
                return self._json(403, {"error": "already set up"})
            name = (payload.get("username") or "").strip()
            pw = payload.get("password") or ""
            if not re.fullmatch(r"[A-Za-z0-9_.-]{2,32}", name):
                return self._json(400, {"error": "username must be 2-32 chars: letters, digits, _ . -"})
            if len(pw) < 8:
                return self._json(400, {"error": "password must be at least 8 characters"})
            u = create_user(name, pw, role="admin")
            import_legacy(u["id"])
            return self._json(200, {"ok": True, "me": user_public(u)}, extra=[self._set_cookie(sign_session(name))])

        u = self.current_user()
        if not u:
            return self._json(401, {"error": "auth required"})
        if not self._csrf_ok():
            return self._json(403, {"error": "bad origin"})

        if path == "/api/logout":
            return self._json(200, {"ok": True}, extra=[self._clear_cookie()])

        if path == "/api/account/password":
            if not verify_pw(payload.get("old") or "", u["pw_hash"]):
                return self._json(403, {"error": "current password is wrong"})
            if len(payload.get("new") or "") < 8:
                return self._json(400, {"error": "new password must be at least 8 characters"})
            with db():
                db().execute("UPDATE users SET pw_hash=? WHERE id=?", (hash_pw(payload["new"]), u["id"]))
            return self._json(200, {"ok": True})

        ms = re.fullmatch(r"/api/conversations/([^/]+)/stream", path)
        if ms and valid_id(ms.group(1)):
            return self._stream(ms.group(1), payload, u)

        # set active leaf (sibling switch)
        ma = re.fullmatch(r"/api/conversations/([^/]+)/active", path)
        if ma and valid_id(ma.group(1)):
            cid = ma.group(1)
            convo = get_convo(cid, u)
            if convo is None:
                return self._json(404, {"error": "not found"})
            _, by, kids = _tree(cid)
            target = payload.get("message_id")
            if target not in by:
                return self._json(400, {"error": "bad message"})
            with db():
                set_leaf(cid, _default_leaf(target, kids))
            return self._json(200, get_convo(cid, u))

        if path == "/api/conversations":
            model = payload.get("model") or get_setting("default_model") or DEFAULT_MODEL
            if not model_allowed(u, model):
                return self._json(403, {"error": "model not permitted"})
            cid = _cid()
            with db():
                db().execute("INSERT INTO conversations(id,owner_id,folder_id,title,system,model,endpoint_id,params,character_id,active_leaf_id,created,updated)"
                             " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                             (cid, u["id"], payload.get("folder_id"), payload.get("title", ""),
                              payload.get("system", get_setting("default_system", DEFAULT_SYSTEM)), model,
                              payload.get("endpoint_id") if u["role"] == "admin" else None,
                              json.dumps(payload.get("params") or {}), payload.get("character_id"), None, _now(), _now()))
            return self._json(200, get_convo(cid, u))

        if path == "/api/folders":
            with db():
                db().execute("INSERT INTO folders(id,owner_id,name,position,created) VALUES(?,?,?,?,?)",
                             ("f-" + uuid.uuid4().hex[:8], u["id"], (payload.get("name") or "").strip() or "New folder", 0, _now()))
            return self._json(200, {"folders": list_folders(u)})

        mf = re.fullmatch(r"/api/folders/([^/]+)", path)
        if mf:
            name = (payload.get("name") or "").strip()
            if name:
                with db():
                    db().execute("UPDATE folders SET name=? WHERE id=? AND owner_id=?", (name, mf.group(1), u["id"]))
            return self._json(200, {"folders": list_folders(u)})

        if path == "/api/characters":
            ch = payload.get("character") or payload
            scope = ch.get("scope", "private")
            if scope == "site" and u["role"] != "admin":
                return self._json(403, {"error": "only admins can create site-wide characters"})
            cid = ch.get("id")
            if cid:
                existing = character_by_id(cid)
                if existing is None:
                    cid = None
                elif not (existing["owner_id"] == u["id"] or u["role"] == "admin"):
                    return self._json(403, {"error": "not yours to edit"})
            if not cid:
                cid = "c-" + uuid.uuid4().hex[:8]
            with db():
                db().execute("INSERT INTO characters(id,owner_id,scope,name,avatar,model,params,system,created)"
                             " VALUES(?,?,?,?,?,?,?,?,?)"
                             " ON CONFLICT(id) DO UPDATE SET scope=excluded.scope,name=excluded.name,avatar=excluded.avatar,"
                             "model=excluded.model,params=excluded.params,system=excluded.system",
                             (cid, u["id"], scope, ch.get("name", "Untitled"), ch.get("avatar", ""), ch.get("model"),
                              json.dumps(ch["params"]) if ch.get("params") else None, ch.get("system", ""), _now()))
            return self._json(200, {"characters": visible_characters(u), "id": cid})

        if path == "/api/settings":
            if u["role"] != "admin":
                return self._json(403, {"error": "admin only"})
            for k in ("endpoints", "active_endpoint", "default_model", "default_system", "default_params", "user_models"):
                if k in payload:
                    set_setting(k, payload[k])
            return self._json(200, {"settings": admin_settings(), "all_models": fetch_models(active_endpoint())})

        if path == "/api/users":
            if u["role"] != "admin":
                return self._json(403, {"error": "admin only"})
            name = (payload.get("username") or "").strip()
            if not re.fullmatch(r"[A-Za-z0-9_.-]{2,32}", name):
                return self._json(400, {"error": "invalid username"})
            if user_by_name(name):
                return self._json(400, {"error": "username already exists"})
            if len(payload.get("password") or "") < 8:
                return self._json(400, {"error": "password must be at least 8 characters"})
            create_user(name, payload["password"], role="admin" if payload.get("role") == "admin" else "user",
                        allowed=payload.get("allowed_models") or None)
            return self._json(200, {"users": [user_public(r) for r in db().execute("SELECT * FROM users ORDER BY id").fetchall()]})

        mu = re.fullmatch(r"/api/users/([0-9]+)", path)
        if mu:
            if u["role"] != "admin":
                return self._json(403, {"error": "admin only"})
            uid = int(mu.group(1))
            if user_by_id(uid) is None:
                return self._json(404, {"error": "not found"})
            c = db()
            with c:
                if "role" in payload and payload["role"] in ("admin", "user"):
                    if uid == u["id"] and payload["role"] != "admin":
                        return self._json(400, {"error": "cannot demote yourself"})
                    c.execute("UPDATE users SET role=? WHERE id=?", (payload["role"], uid))
                if "allowed_models" in payload:
                    am = payload["allowed_models"]
                    c.execute("UPDATE users SET allowed_models=? WHERE id=?", (json.dumps(am) if am else None, uid))
                if "disabled" in payload:
                    if uid == u["id"] and payload["disabled"]:
                        return self._json(400, {"error": "cannot disable yourself"})
                    c.execute("UPDATE users SET disabled=? WHERE id=?", (1 if payload["disabled"] else 0, uid))
                if payload.get("password"):
                    if len(payload["password"]) < 8:
                        return self._json(400, {"error": "password too short"})
                    c.execute("UPDATE users SET pw_hash=? WHERE id=?", (hash_pw(payload["password"]), uid))
            return self._json(200, {"users": [user_public(r) for r in c.execute("SELECT * FROM users ORDER BY id").fetchall()]})

        mr = re.fullmatch(r"/api/conversations/([^/]+)/messages/([^/]+)/rating", path)
        if mr:
            cid, mid = mr.group(1), mr.group(2)
            if not get_convo(cid, u):
                return self._json(404, {"error": "not found"})
            val = payload.get("rating")
            if val not in (-1, 0, 1):
                return self._json(400, {"error": "rating must be -1, 0 or 1"})
            with db():
                db().execute("UPDATE messages SET rating=? WHERE id=? AND convo_id=?", (val or None, mid, cid))
            return self._json(200, get_convo(cid, u))

        me = re.fullmatch(r"/api/conversations/([^/]+)/messages/([^/]+)", path)
        if me:
            cid, mid = me.group(1), me.group(2)
            if not get_convo(cid, u):
                return self._json(404, {"error": "not found"})
            if "content" in payload:
                with db():
                    db().execute("UPDATE messages SET content=?, edited=? WHERE id=? AND convo_id=?",
                                 (payload["content"], _now(), mid, cid))
                    touch_convo(cid)
            return self._json(200, get_convo(cid, u))

        msg = re.fullmatch(r"/api/conversations/([^/]+)/settings", path)
        if msg and valid_id(msg.group(1)):
            cid = msg.group(1)
            if get_convo(cid, u) is None:
                return self._json(404, {"error": "not found"})
            if payload.get("model") and not model_allowed(u, payload["model"]):
                return self._json(403, {"error": "model not permitted"})
            sets, args = [], []
            for key, col in (("system", "system"), ("character_id", "character_id")):
                if key in payload:
                    sets.append(col + "=?"); args.append(payload[key])
            if payload.get("model"):
                sets.append("model=?"); args.append(payload["model"])
            if "endpoint_id" in payload and u["role"] == "admin":
                sets.append("endpoint_id=?"); args.append(payload["endpoint_id"])
            if "params" in payload:
                sets.append("params=?"); args.append(json.dumps(payload["params"] or {}))
            if "folder_id" in payload:
                sets.append("folder_id=?"); args.append(payload["folder_id"] or None)
            if "title" in payload and (payload["title"] or "").strip():
                sets.append("title=?"); args.append(payload["title"].strip())
            sets.append("updated=?"); args.append(_now())
            args.append(cid)
            with db():
                db().execute("UPDATE conversations SET " + ",".join(sets) + " WHERE id=?", args)
            return self._json(200, get_convo(cid, u))

        return self._json(404, {"error": "not found"})

    # ================================================== streaming (tree-aware)
    def _stream(self, cid, payload, u):
        convo = get_convo(cid, u)
        if convo is None:
            return self._json(404, {"error": "not found"})
        ep, model, system, params = resolve_request(convo)
        if not model_allowed(u, model):
            return self._json(403, {"error": "model not permitted"})

        _, by, _ = _tree(cid)
        content = (payload.get("content") or "").strip()
        regenerate_id = payload.get("regenerate_id")
        edit_user_id = payload.get("edit_user_id")

        if regenerate_id or payload.get("regenerate"):
            target = regenerate_id or convo.get("active_leaf_id")
            tr = by.get(target)
            if tr is None or tr["role"] != "assistant":
                return self._json(400, {"error": "nothing to regenerate"})
            parent = tr["parent_id"]
            ctx = chain_content(cid, parent)

            def commit(reply, reasoning, meta):
                with db():
                    aid = insert_message(cid, parent, "assistant", reply, reasoning, model, meta)
                    set_leaf(cid, aid)
                    touch_convo(cid)
        elif edit_user_id:
            tu = by.get(edit_user_id)
            if tu is None or tu["role"] != "user":
                return self._json(400, {"error": "cannot edit/resend that message"})
            new_content = content or tu["content"]
            parent = tu["parent_id"]
            ctx = chain_content(cid, parent) + [{"role": "user", "content": new_content}]

            def commit(reply, reasoning, meta):
                with db():
                    uid = insert_message(cid, parent, "user", new_content)
                    aid = insert_message(cid, uid, "assistant", reply, reasoning, model, meta)
                    set_leaf(cid, aid)
                    touch_convo(cid); maybe_title(cid)
        else:
            if not content:
                return self._json(400, {"error": "empty message"})
            parent = convo.get("active_leaf_id")
            ctx = chain_content(cid, parent) + [{"role": "user", "content": content}]

            def commit(reply, reasoning, meta):
                with db():
                    uid = insert_message(cid, parent, "user", content)
                    aid = insert_message(cid, uid, "assistant", reply, reasoning, model, meta)
                    set_leaf(cid, aid)
                    touch_convo(cid); maybe_title(cid)

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        def emit(obj):
            self.wfile.write((json.dumps(obj) + "\n").encode("utf-8"))
            self.wfile.flush()

        parts, rparts, usage = [], [], None
        t0 = time.time(); t_first = None; gone = False
        try:
            for ev in stream_model(ep, model, system, ctx, params):
                if "delta" in ev:
                    if t_first is None:
                        t_first = time.time()
                    parts.append(ev["delta"])
                    try:
                        emit({"delta": ev["delta"]})
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        gone = True; break
                elif "reasoning" in ev:
                    if t_first is None:
                        t_first = time.time()
                    rparts.append(ev["reasoning"])
                    try:
                        emit({"reasoning": ev["reasoning"]})
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        gone = True; break
                elif "usage" in ev:
                    usage = ev["usage"]
        except Exception as e:
            if not gone:
                try:
                    emit({"error": "model error: " + str(e)})
                except Exception:
                    pass
            return

        reply, reasoning = "".join(parts), "".join(rparts)
        if not reply and not reasoning:
            return
        commit(reply, reasoning, build_meta(t0, t_first, time.time(), usage, reply))
        if not gone:
            try:
                emit({"done": True, "convo": get_convo(cid, u)})
            except Exception:
                pass


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


# ---------------------------------------------------------------- auth pages
AUTH_CSS = r"""
  :root{--bg:#100c08;--panel:#17110a;--panel2:#1e1810;--text:#e8ddc5;--muted:#9a8c72;--faint:#675d49;
        --accent:#cf8a3c;--danger:#c25a46;
        --mono:'IBM Plex Mono',ui-monospace,Consolas,monospace;--serif:'Newsreader','Iowan Old Style',Georgia,serif;}
  *{box-sizing:border-box;}
  body{margin:0;min-height:100vh;background:var(--bg);color:var(--text);font-family:var(--serif);
       display:flex;align-items:center;justify-content:center;padding:24px;
       background-image:radial-gradient(130% 80% at 50% -10%, rgba(207,138,60,.07), transparent 60%);}
  .card{width:380px;max-width:100%;background:var(--panel);border-radius:16px;padding:32px 28px;box-shadow:0 30px 80px rgba(0,0,0,.55);}
  .brand{font-family:var(--mono);font-size:14px;letter-spacing:.42em;text-transform:uppercase;color:var(--accent);}
  .brand .sub{color:var(--faint);letter-spacing:.22em;}
  h1{font-family:var(--serif);font-weight:500;font-size:26px;margin:16px 0 4px;}
  p.lede{color:var(--muted);font-size:15px;margin:0 0 22px;line-height:1.5;}
  label{display:block;font-family:var(--mono);font-size:11px;text-transform:uppercase;letter-spacing:.14em;color:var(--muted);margin:16px 0 6px;}
  input{width:100%;background:var(--panel2);border:none;border-radius:10px;color:var(--text);padding:12px;font-size:15px;font-family:var(--serif);}
  input:focus{outline:2px solid var(--accent);outline-offset:-1px;}
  button{width:100%;margin-top:24px;background:var(--accent);color:#1a1206;border:none;border-radius:10px;padding:13px;
         font-family:var(--mono);font-size:13px;letter-spacing:.12em;text-transform:uppercase;font-weight:600;cursor:pointer;}
  button:hover{filter:brightness(1.07);}
  .err{color:var(--danger);font-size:13px;margin-top:14px;min-height:16px;font-family:var(--mono);}
"""

LOGIN_PAGE = (r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>oracle · sign in</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Newsreader:opsz,wght@6..72,400;6..72,500&display=swap" rel="stylesheet">
<style>""" + AUTH_CSS + r"""</style></head><body>
<form class="card" id="f">
  <div class="brand">ORACLE</div>
  <h1>Sign in</h1>
  <p class="lede">This instance is private. Enter your credentials to continue.</p>
  <label>Username</label><input id="u" autocomplete="username" autofocus>
  <label>Password</label><input id="p" type="password" autocomplete="current-password">
  <button type="submit">Enter</button>
  <div class="err" id="e"></div>
</form>
<script>
document.getElementById("f").onsubmit=async(ev)=>{ev.preventDefault();const e=document.getElementById("e");e.textContent="";
 try{const r=await fetch("/api/login",{method:"POST",headers:{"Content-Type":"application/json"},
  body:JSON.stringify({username:document.getElementById("u").value,password:document.getElementById("p").value})});
  const j=await r.json().catch(()=>({}));if(!r.ok)throw new Error(j.error||"sign in failed");location.href="/";
 }catch(err){e.textContent=err.message;}};
</script></body></html>""")

SETUP_PAGE = (r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>oracle · setup</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Newsreader:opsz,wght@6..72,400;6..72,500&display=swap" rel="stylesheet">
<style>""" + AUTH_CSS + r"""</style></head><body>
<form class="card" id="f">
  <div class="brand">ORACLE <span class="sub">// setup</span></div>
  <h1>Create admin account</h1>
  <p class="lede">No accounts exist yet. Create the administrator. This screen disables itself afterwards.</p>
  <label>Admin username</label><input id="u" autocomplete="username" autofocus>
  <label>Password <span style="text-transform:none;letter-spacing:0;color:var(--faint)">(min 8)</span></label><input id="p" type="password" autocomplete="new-password">
  <label>Confirm password</label><input id="p2" type="password" autocomplete="new-password">
  <button type="submit">Create &amp; enter</button>
  <div class="err" id="e"></div>
</form>
<script>
document.getElementById("f").onsubmit=async(ev)=>{ev.preventDefault();const e=document.getElementById("e");e.textContent="";
 const p=document.getElementById("p").value,p2=document.getElementById("p2").value;
 if(p!==p2){e.textContent="passwords do not match";return;}
 try{const r=await fetch("/api/setup",{method:"POST",headers:{"Content-Type":"application/json"},
  body:JSON.stringify({username:document.getElementById("u").value,password:p})});
  const j=await r.json().catch(()=>({}));if(!r.ok)throw new Error(j.error||"setup failed");location.href="/";
 }catch(err){e.textContent=err.message;}};
</script></body></html>""")

PAGE_HEAD = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<title>oracle</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&display=swap" rel="stylesheet">
<script>
(function(){try{var d=document.documentElement,L=localStorage;
 d.setAttribute('data-theme',L.getItem('oracle_theme')||'dark');
 var fs=L.getItem('oracle_fs');if(fs)d.style.setProperty('--rs',fs);
 var w=L.getItem('oracle_sbw');if(w)d.style.setProperty('--sbw',w+'px');
 var cw=L.getItem('oracle_cw');if(cw)d.style.setProperty('--cw',cw+'px');
 var f=L.getItem('oracle_font');var FF={serif:"var(--serif)",sans:"-apple-system,'Segoe UI',system-ui,sans-serif",mono:"var(--mono)"};
 if(f&&FF[f])d.style.setProperty('--read-font',FF[f]);
}catch(e){}})();
</script>
<style>
  :root{
    --rs:1; --sbw:288px; --cw:840px; --read-font:var(--serif);
    --bg:#100c08; --panel:#15100a; --surface:#1b150d; --surface2:#231b10; --surface3:#2a2114;
    --line:rgba(150,124,84,.10);
    --text:#e8ddc5; --muted:#9a8c72; --faint:#6a6049; --dim:#544a38;
    --accent:#cf8a3c; --accent2:#cda261; --accent-weak:rgba(207,138,60,.14);
    --user:#b49f78; --bot:#d29a4b; --danger:#c8604c; --danger-weak:rgba(200,90,70,.14); --ok:#94a05c;
    --code-bg:#0c0905; --shadow:0 24px 70px rgba(0,0,0,.55);
    --mono:'IBM Plex Mono',ui-monospace,SFMono-Regular,Consolas,monospace;
    --serif:'Newsreader','Iowan Old Style',Georgia,'Times New Roman',serif;
    color-scheme:dark;
  }
  [data-theme="light"]{
    --bg:#f0e8d6; --panel:#e9e0cc; --surface:#e3d8c1; --surface2:#dacdb0; --surface3:#d2c3a4;
    --line:rgba(90,68,34,.14);
    --text:#2c2316; --muted:#6a5c43; --faint:#9a8b6e; --dim:#b3a487;
    --accent:#b26a1d; --accent2:#7d5320; --accent-weak:rgba(178,106,29,.15);
    --user:#6d5a32; --bot:#955414; --danger:#a23a2a; --danger-weak:rgba(162,58,42,.12); --ok:#5f6b2f;
    --code-bg:#e6dcc6; --shadow:0 24px 60px rgba(60,45,20,.25);
    color-scheme:light;
  }
  *{box-sizing:border-box;}
  html,body{height:100%;}
  body{margin:0;background:var(--bg);color:var(--text);font-family:var(--serif);font-size:16px;overflow:hidden;-webkit-font-smoothing:antialiased;}
  #app{display:flex;height:100dvh;height:100vh;}
  button{font-family:var(--mono);color:inherit;}
  .mono{font-family:var(--mono);}
  .lbl{font-family:var(--mono);font-size:10.5px;text-transform:uppercase;letter-spacing:.16em;color:var(--muted);}
  ::-webkit-scrollbar{width:9px;height:9px;}
  ::-webkit-scrollbar-thumb{background:var(--surface2);border-radius:6px;border:2px solid transparent;background-clip:padding-box;}
  ::-webkit-scrollbar-thumb:hover{background:var(--surface3);background-clip:padding-box;}
  svg{display:block;}
  .ico{width:18px;height:18px;stroke:currentColor;stroke-width:1.8;fill:none;stroke-linecap:round;stroke-linejoin:round;}

  /* ---------------- sidebar */
  #sidebar{width:var(--sbw);flex:0 0 var(--sbw);background:var(--panel);display:flex;flex-direction:column;min-height:0;position:relative;z-index:60;}
  #app.sbcollapsed #sidebar{display:none;}
  #resizer{position:absolute;top:0;right:-3px;width:6px;height:100%;cursor:col-resize;z-index:5;}
  #resizer:hover{background:var(--accent-weak);}
  .side-head{padding:16px 14px 10px;display:flex;align-items:center;gap:8px;}
  .side-head .mark{font-family:var(--mono);font-size:14px;letter-spacing:.36em;text-transform:uppercase;color:var(--accent);font-weight:600;}
  .side-head .grow{flex:1;}
  .side-act{padding:8px 12px 10px;display:flex;flex-direction:column;gap:8px;}
  #newbtn{width:100%;padding:11px;background:var(--accent-weak);color:var(--accent);border:none;border-radius:10px;
          font-size:11.5px;letter-spacing:.14em;text-transform:uppercase;cursor:pointer;font-weight:600;}
  #newbtn:hover{background:var(--accent);color:#1a1206;}
  .side-act .row{display:flex;gap:8px;}
  #searchbox{width:100%;background:var(--surface);color:var(--text);border:none;border-radius:9px;padding:9px 11px;font-family:var(--mono);font-size:12px;}
  #searchbox:focus{outline:2px solid var(--accent-weak);outline-offset:-1px;}
  .side-act .row button{flex:1;background:var(--surface);border:none;color:var(--muted);border-radius:9px;padding:9px;font-family:var(--mono);font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;cursor:pointer;}
  .side-act .row button:hover,#selbtn.on{background:var(--surface2);color:var(--text);}
  #tree{flex:1;overflow-y:auto;padding:2px 8px 16px;}
  .folder-head{display:flex;align-items:center;gap:7px;padding:7px 7px;border-radius:8px;cursor:pointer;color:var(--muted);margin-top:4px;}
  .folder-head:hover{background:var(--surface);}
  .folder-head .tw{width:9px;font-family:var(--mono);font-size:9px;color:var(--faint);transition:transform .12s;}
  .folder-head.collapsed .tw{transform:rotate(-90deg);}
  .folder-head .fname{flex:1;font-family:var(--mono);font-size:11px;letter-spacing:.1em;text-transform:uppercase;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
  .folder-head .cnt{font-family:var(--mono);font-size:10px;color:var(--faint);}
  .folder-head .fmenu{opacity:0;background:none;border:none;color:var(--faint);cursor:pointer;font-size:15px;padding:0 4px;line-height:1;}
  .folder-head:hover .fmenu{opacity:1;}
  .folder.dragover .folder-head{background:var(--accent-weak);box-shadow:inset 0 0 0 1px var(--accent);}
  .folder-body.hidden{display:none;}
  .convo{padding:9px 30px 9px 11px;border-radius:9px;cursor:pointer;margin:2px 0;position:relative;}
  .convo:hover{background:var(--surface);}
  .convo.active{background:var(--surface2);}
  .convo .ct{font-family:var(--serif);font-size:calc(15px*var(--rs));line-height:1.3;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
  .convo .cm{font-family:var(--mono);font-size:10px;color:var(--faint);margin-top:3px;letter-spacing:.03em;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
  .convo .cmenu{position:absolute;right:5px;top:50%;transform:translateY(-50%);opacity:0;background:none;border:none;color:var(--faint);cursor:pointer;font-size:16px;padding:4px 6px;border-radius:7px;line-height:1;}
  .convo:hover .cmenu{opacity:1;}
  .convo .cmenu:hover{color:var(--text);background:var(--surface3);}
  .convo.dragging{opacity:.4;}
  .convo .sel{display:none;}
  #app.selmode .convo{padding-left:34px;}
  #app.selmode .convo .sel{display:block;position:absolute;left:10px;top:50%;transform:translateY(-50%);width:15px;height:15px;border-radius:4px;background:var(--surface3);}
  #app.selmode .convo.checked{background:var(--accent-weak);}
  #app.selmode .convo.checked .sel{background:var(--accent);box-shadow:inset 0 0 0 2px var(--accent);}
  #app.selmode .convo .cmenu{display:none;}
  .empty-list{font-family:var(--mono);color:var(--faint);font-size:11px;padding:14px 10px;text-align:center;letter-spacing:.03em;}
  #selbar{display:none;padding:10px 12px;gap:8px;align-items:center;background:var(--surface);}
  #app.selmode #selbar{display:flex;}
  #selbar .cnt{flex:1;font-family:var(--mono);font-size:11px;color:var(--muted);}
  #selbar button{background:var(--surface2);border:none;color:var(--muted);border-radius:8px;padding:7px 10px;font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;cursor:pointer;}
  #selbar button:hover{background:var(--surface3);color:var(--text);}
  #selbar button.danger:hover{color:var(--danger);}
  .side-foot{padding:9px 12px;display:flex;align-items:center;gap:6px;}
  .side-foot .who{flex:1;min-width:0;cursor:pointer;border-radius:8px;padding:4px 6px;}
  .side-foot .who:hover{background:var(--surface);}
  .side-foot .who .nm{font-family:var(--mono);font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
  .side-foot .who .rl{font-family:var(--mono);font-size:9px;letter-spacing:.16em;text-transform:uppercase;color:var(--accent);}
  .iconbtn{background:none;border:none;color:var(--muted);cursor:pointer;border-radius:8px;width:32px;height:32px;display:flex;align-items:center;justify-content:center;}
  .iconbtn:hover{background:var(--surface);color:var(--text);}

  /* ---------------- main */
  #main{flex:1;display:flex;flex-direction:column;min-width:0;min-height:0;position:relative;}
  #bar{position:absolute;top:0;left:0;right:0;z-index:20;padding:10px 16px;display:flex;align-items:center;gap:10px;min-height:56px;
       background:linear-gradient(to bottom, var(--bg) 62%, transparent);backdrop-filter:blur(9px);-webkit-backdrop-filter:blur(9px);}
  .menubtn,#revealbtn{display:none;}
  #app.sbcollapsed #revealbtn{display:flex;}
  #titlewrap{min-width:0;flex:1;}
  #title{font-family:var(--serif);font-size:18px;font-weight:500;cursor:text;padding:2px 6px;border-radius:7px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;display:inline-block;max-width:100%;}
  #title:hover{background:var(--surface);}
  #title.editing{background:var(--surface);outline:2px solid var(--accent-weak);}
  #submeta{font-family:var(--mono);font-size:10px;color:var(--faint);letter-spacing:.05em;margin-top:1px;padding-left:6px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
  .barbtns{display:flex;align-items:center;gap:8px;}
  select.msel{background:var(--surface);color:var(--text);border:none;border-radius:8px;padding:7px 9px;font-family:var(--mono);font-size:11px;max-width:170px;cursor:pointer;}
  select.msel:focus{outline:2px solid var(--accent-weak);}
  .barbtn{background:var(--surface);border:none;color:var(--muted);cursor:pointer;height:32px;padding:0 11px;border-radius:8px;font-size:11px;letter-spacing:.07em;text-transform:uppercase;display:inline-flex;align-items:center;gap:6px;}
  .barbtn:hover{background:var(--surface2);color:var(--text);}

  /* ---------------- log */
  #log{flex:1;overflow-y:auto;padding:74px 0 10px;}
  .wrap{max-width:var(--cw);margin:0 auto;padding:0 28px;}
  .msg{padding:6px 0 20px;position:relative;}
  .msg .head{display:flex;align-items:center;gap:10px;margin-bottom:7px;}
  .msg .head .role{font-family:var(--mono);font-size:10.5px;letter-spacing:.18em;text-transform:uppercase;}
  .msg.user .head .role{color:var(--user);}
  .msg.assistant .head .role{color:var(--bot);}
  .msg .head .tm{font-family:var(--mono);font-size:10px;color:var(--faint);letter-spacing:.03em;}
  .msg .head .sib{display:inline-flex;align-items:center;gap:1px;font-family:var(--mono);font-size:10.5px;color:var(--accent2);background:var(--surface);border-radius:20px;padding:2px 3px;margin-left:2px;}
  .msg .head .sib .n{padding:0 4px;letter-spacing:.04em;}
  .msg .head .sib button{background:none;border:none;color:var(--accent);cursor:pointer;padding:1px 6px;font-size:14px;line-height:1;border-radius:12px;}
  .msg .head .sib button:hover{background:var(--surface2);}
  .msg .head .sib button[disabled]{opacity:.28;cursor:default;}
  .msg .head .ratemark{font-family:var(--mono);font-size:10px;margin-left:2px;}
  .msg .head .ratemark.up{color:var(--accent);} .msg .head .ratemark.down{color:var(--danger);}
  .actions button.rate .ico{width:14px;height:14px;}
  .actions button.rate.down .ico{transform:scaleY(-1);}
  .actions button.rate.on.up{color:var(--accent);border-color:transparent;}
  .actions button.rate.on.down{color:var(--danger);}
  .bubble{font-family:var(--read-font);font-size:calc(17.5px*var(--rs));line-height:1.78;color:var(--text);max-width:none;overflow-wrap:anywhere;word-break:break-word;}
  .bubble.raw{white-space:pre-wrap;}
  .bubble p{margin:.7em 0;} .bubble p:first-child{margin-top:0;} .bubble p:last-child{margin-bottom:0;}
  .bubble h1,.bubble h2,.bubble h3,.bubble h4{font-weight:600;line-height:1.3;margin:1em 0 .4em;}
  .bubble h1{font-size:1.5em;} .bubble h2{font-size:1.3em;} .bubble h3{font-size:1.13em;}
  .bubble ul,.bubble ol{margin:.5em 0;padding-left:1.4em;} .bubble li{margin:.25em 0;}
  .bubble a{color:var(--accent);text-underline-offset:2px;}
  .bubble code{font-family:var(--mono);font-size:.82em;background:var(--code-bg);border-radius:5px;padding:.06em .34em;}
  .bubble pre{background:var(--code-bg);border-radius:10px;padding:13px 15px;overflow-x:auto;margin:.7em 0;}
  .bubble pre code{background:none;padding:0;font-size:.84em;line-height:1.55;}
  .bubble blockquote{margin:.7em 0;padding:.1em 0 .1em 1.1em;border-left:2px solid var(--surface3);color:var(--muted);font-style:italic;}
  .bubble hr{border:none;border-top:1px solid var(--line);margin:1.1em 0;}
  .bubble strong{font-weight:600;}
  .reason{margin:0 0 10px;}
  .reason summary{cursor:pointer;font-family:var(--mono);color:var(--faint);font-size:10.5px;letter-spacing:.12em;text-transform:uppercase;list-style:none;padding:3px 0;}
  .reason summary::-webkit-details-marker{display:none;}
  .reason summary:before{content:"+ ";color:var(--accent);}
  .reason[open] summary:before{content:"- ";}
  .reason .rbody{font-family:var(--read-font);font-style:italic;white-space:pre-wrap;color:var(--muted);font-size:calc(15px*var(--rs));line-height:1.6;border-left:2px solid var(--surface3);padding:6px 0 6px 13px;margin-top:5px;overflow-wrap:anywhere;}
  .meta{font-family:var(--mono);font-size:10px;color:var(--faint);margin-top:11px;display:flex;gap:8px;flex-wrap:wrap;align-items:center;letter-spacing:.02em;}
  .meta .pill{background:var(--surface);border-radius:9px;padding:2px 8px;}
  .meta .pill.k{color:var(--accent2);}
  .actions{margin-top:11px;display:flex;gap:3px;flex-wrap:wrap;opacity:0;transition:opacity .12s;}
  .msg:hover .actions{opacity:1;}
  .actions button{background:none;border:none;color:var(--faint);font-size:10px;letter-spacing:.07em;text-transform:uppercase;cursor:pointer;padding:4px 8px;border-radius:7px;}
  .actions button:hover{background:var(--surface);color:var(--text);}
  .actions button.danger:hover{color:var(--danger);}
  .edit-area{width:100%;background:var(--code-bg);color:var(--text);border:none;border-radius:10px;padding:12px 13px;font-size:calc(16px*var(--rs));font-family:var(--read-font);line-height:1.6;resize:vertical;min-height:96px;}
  .edit-area:focus{outline:2px solid var(--accent-weak);}
  .edit-row{display:flex;gap:8px;margin-top:8px;}
  .btn-primary{background:var(--accent);color:#1a1206;border:none;border-radius:8px;padding:8px 15px;font-size:11px;letter-spacing:.08em;text-transform:uppercase;cursor:pointer;}
  .btn-primary:hover{filter:brightness(1.07);}
  .btn-ghost{background:var(--surface2);color:var(--muted);border:none;border-radius:8px;padding:8px 15px;font-size:11px;letter-spacing:.08em;text-transform:uppercase;cursor:pointer;}
  .btn-ghost:hover{background:var(--surface3);color:var(--text);}
  .btn-danger{background:var(--danger-weak);color:var(--danger);border:none;border-radius:8px;padding:8px 15px;font-size:11px;letter-spacing:.08em;text-transform:uppercase;cursor:pointer;}
  .btn-danger:hover{background:var(--danger);color:#fff;}
  .typing{display:inline-flex;gap:5px;align-items:center;padding:3px 0;}
  .typing i{width:5px;height:5px;border-radius:50%;background:var(--accent);opacity:.4;animation:blink 1.2s infinite;}
  .typing i:nth-child(2){animation-delay:.2s;} .typing i:nth-child(3){animation-delay:.4s;}
  @keyframes blink{0%,80%,100%{opacity:.25;}40%{opacity:1;}}
  .empty{text-align:center;margin-top:16vh;color:var(--faint);}
  .empty .glyph{font-family:var(--mono);font-size:30px;color:var(--accent);opacity:.7;}
  .empty h2{font-family:var(--serif);font-weight:500;color:var(--muted);font-size:22px;margin:14px 0 4px;}
  .empty p{font-family:var(--mono);font-size:11px;letter-spacing:.06em;color:var(--faint);}

  /* ---------------- composer */
  #composer{padding:12px 0 18px;background:var(--bg);}
  #composer .wrap{display:flex;gap:10px;align-items:flex-end;}
  .input-shell{flex:1;display:flex;align-items:flex-end;background:var(--surface);border-radius:14px;padding:12px 14px;min-height:48px;}
  .input-shell:focus-within{outline:2px solid var(--accent-weak);outline-offset:-1px;}
  #input{flex:1;background:none;color:var(--text);border:none;font-size:calc(16.5px*var(--rs));font-family:var(--read-font);resize:none;max-height:230px;line-height:1.5;padding:0;}
  #input:focus{outline:none;}
  #send{flex:0 0 auto;background:var(--accent);color:#fff;border:none;border-radius:14px;width:48px;height:48px;cursor:pointer;display:flex;align-items:center;justify-content:center;}
  #send:hover{filter:brightness(1.08);} #send.stop{background:var(--danger);color:#fff;}
  #send .ico{width:20px;height:20px;stroke-width:2.2;}
  .chint{max-width:var(--cw);margin:7px auto 0;padding:0 28px;font-family:var(--mono);font-size:10px;color:var(--dim);letter-spacing:.03em;}

  /* ---------------- overlays */
  #backdrop{position:fixed;inset:0;background:rgba(8,6,3,.5);opacity:0;pointer-events:none;transition:opacity .2s;z-index:50;}
  #backdrop.show{opacity:1;pointer-events:auto;}
  .panel{position:fixed;top:0;right:0;height:100dvh;height:100vh;width:430px;max-width:93vw;background:var(--panel);box-shadow:var(--shadow);transform:translateX(103%);transition:transform .22s ease;z-index:70;display:flex;flex-direction:column;}
  .panel.show{transform:translateX(0);}
  .phead{padding:16px 18px;display:flex;align-items:center;gap:10px;}
  .phead h3{margin:0;font-family:var(--mono);font-size:12px;letter-spacing:.16em;text-transform:uppercase;color:var(--accent);}
  .phead .x{margin-left:auto;background:none;border:none;color:var(--muted);font-size:22px;cursor:pointer;width:32px;height:32px;border-radius:8px;line-height:1;}
  .phead .x:hover{background:var(--surface);color:var(--text);}
  .pbody{flex:1;overflow-y:auto;padding:8px 18px 16px;}
  .pfoot{padding:14px 18px;display:flex;gap:8px;}
  .pfoot button{flex:1;border-radius:9px;padding:11px;font-size:12px;letter-spacing:.1em;text-transform:uppercase;cursor:pointer;}
  #modal{position:fixed;inset:0;display:none;align-items:center;justify-content:center;z-index:70;padding:20px;}
  #modal.show{display:flex;}
  .modal-card{background:var(--panel);border-radius:16px;box-shadow:var(--shadow);width:800px;max-width:100%;max-height:90dvh;max-height:90vh;display:flex;flex-direction:column;overflow:hidden;}
  .mhead{padding:16px 20px;display:flex;align-items:center;gap:14px;}
  .mhead h3{margin:0;font-family:var(--mono);font-size:12px;letter-spacing:.18em;text-transform:uppercase;color:var(--accent);}
  .mhead .x{margin-left:auto;background:none;border:none;color:var(--muted);font-size:22px;cursor:pointer;}
  .tabs{display:flex;gap:2px;padding:0 16px;flex-wrap:wrap;}
  .tabs button{background:none;border:none;color:var(--faint);padding:9px 13px;font-family:var(--mono);font-size:11px;letter-spacing:.09em;text-transform:uppercase;cursor:pointer;border-radius:8px 8px 0 0;}
  .tabs button.active{color:var(--text);background:var(--surface);}
  .tab-body{flex:1;overflow-y:auto;padding:18px 20px;background:var(--surface);}
  .tab-pane{display:none;} .tab-pane.active{display:block;}

  label.fld{display:block;margin:14px 0 0;} label.fld:first-child{margin-top:0;}
  label.fld .lab{display:block;font-family:var(--mono);font-size:10.5px;text-transform:uppercase;letter-spacing:.13em;color:var(--muted);margin-bottom:6px;}
  label.fld .sub{text-transform:none;letter-spacing:0;color:var(--faint);}
  .fld input[type=text],.fld input[type=number],.fld input[type=password],.fld textarea,.fld select{
      width:100%;background:var(--bg);color:var(--text);border:none;border-radius:9px;padding:10px 11px;font-size:14px;font-family:var(--mono);}
  .fld textarea{resize:vertical;min-height:120px;line-height:1.6;font-family:var(--serif);font-size:15px;}
  .fld input:focus,.fld textarea:focus,.fld select:focus{outline:2px solid var(--accent-weak);outline-offset:-1px;}
  .params-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:18px;margin-top:10px;}
  .pg{min-width:0;}
  .pg .pgname{font-family:var(--mono);font-size:10.5px;color:var(--muted);letter-spacing:.06em;margin-bottom:6px;}
  .pg .pgrow{display:flex;align-items:center;gap:8px;}
  .pg .pgrow input[type=number]{flex:1;min-width:0;background:var(--bg);color:var(--text);border:none;border-radius:7px;padding:8px 10px;font-family:var(--mono);font-size:12.5px;}
  .pg .pgrow input:focus{outline:2px solid var(--accent-weak);outline-offset:-1px;}
  .pg .reset{flex:0 0 auto;background:var(--surface2);border:none;color:var(--faint);border-radius:7px;padding:8px 11px;font-family:var(--mono);font-size:10px;letter-spacing:.06em;text-transform:uppercase;cursor:pointer;}
  .pg .reset:hover{color:var(--accent);background:var(--surface3);}
  .pg input[type=range]{width:100%;accent-color:var(--accent);height:4px;margin-top:9px;}
  .pg.off input[type=number]{color:var(--faint);}
  .pg.off input[type=range]{filter:grayscale(.7);opacity:.5;}
  .params-section{margin-top:22px;}
  .params-foot{display:flex;gap:8px;margin-top:16px;}
  .hintbox{font-family:var(--mono);font-size:11px;color:var(--faint);background:var(--bg);border-radius:9px;padding:10px 12px;margin-top:12px;line-height:1.6;letter-spacing:.02em;}
  .ep-card,.row-card{background:var(--bg);border-radius:11px;padding:12px;margin-bottom:10px;}
  .ep-card .ep-top{display:flex;align-items:center;gap:8px;margin-bottom:8px;}
  .ep-card input,.ep-card select{background:var(--surface);border:none;border-radius:7px;color:var(--text);padding:8px 9px;font-family:var(--mono);font-size:12px;}
  .ep-card .ep-top input{flex:1;font-weight:600;}
  .ep-card .grid2{display:grid;grid-template-columns:1fr;gap:8px;}
  .ep-card .ep-row{display:flex;gap:8px;margin-top:8px;align-items:center;flex-wrap:wrap;}
  .ep-card .status{font-family:var(--mono);font-size:11px;color:var(--faint);}
  .mini{background:var(--surface2);border:none;color:var(--muted);border-radius:8px;padding:7px 12px;font-family:var(--mono);font-size:11px;letter-spacing:.05em;text-transform:uppercase;cursor:pointer;}
  .mini:hover{color:var(--text);background:var(--surface3);} .mini.danger:hover{color:var(--danger);}
  .radio-active{display:inline-flex;gap:6px;align-items:center;font-family:var(--mono);font-size:10.5px;letter-spacing:.07em;text-transform:uppercase;color:var(--muted);cursor:pointer;}
  .model-pick{display:flex;flex-wrap:wrap;gap:7px;margin-top:6px;}
  .model-pick label{display:inline-flex;gap:6px;align-items:center;background:var(--bg);border-radius:20px;padding:6px 12px;font-family:var(--mono);font-size:11px;color:var(--muted);cursor:pointer;}
  .model-pick label:hover{color:var(--text);}
  .row-card{display:flex;align-items:center;gap:10px;}
  .row-card .cav{width:34px;height:34px;border-radius:9px;background:var(--surface2);display:flex;align-items:center;justify-content:center;font-family:var(--mono);font-size:14px;color:var(--accent);flex:0 0 auto;}
  .row-card .cmain{flex:1;min-width:0;}
  .row-card .cn{font-family:var(--serif);font-size:15px;}
  .row-card .cn .badge{font-family:var(--mono);font-size:9px;letter-spacing:.1em;text-transform:uppercase;color:var(--accent);border:1px solid var(--accent-weak);border-radius:5px;padding:1px 5px;margin-left:8px;}
  .row-card .cs{font-family:var(--mono);font-size:11px;color:var(--faint);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-top:2px;}
  .row-card .cbtns{display:flex;gap:6px;}
  .toggle-row{display:flex;align-items:center;gap:8px;font-family:var(--mono);font-size:11px;color:var(--muted);margin-top:14px;letter-spacing:.03em;}
  .seg{display:inline-flex;background:var(--bg);border-radius:9px;padding:3px;gap:3px;}
  .seg button{background:none;border:none;color:var(--muted);padding:7px 14px;border-radius:7px;font-family:var(--mono);font-size:11px;letter-spacing:.06em;text-transform:uppercase;cursor:pointer;}
  .seg button.on{background:var(--surface2);color:var(--text);}
  .danger-zone{margin-top:24px;padding:14px;border-radius:11px;background:var(--danger-weak);}
  .danger-zone .dz-t{font-family:var(--mono);font-size:10.5px;letter-spacing:.12em;text-transform:uppercase;color:var(--danger);margin-bottom:8px;}
  .danger-zone p{font-family:var(--mono);font-size:11px;color:var(--muted);margin:0 0 10px;line-height:1.5;}

  /* context menu + dialogs */
  #menu{position:fixed;z-index:90;background:var(--surface);border-radius:10px;box-shadow:var(--shadow);padding:5px;min-width:180px;display:none;}
  #menu.show{display:block;}
  #menu button{display:block;width:100%;text-align:left;background:none;border:none;color:var(--text);font-family:var(--mono);font-size:11.5px;letter-spacing:.03em;padding:9px 11px;border-radius:7px;cursor:pointer;}
  #menu button:hover{background:var(--surface2);} #menu button.danger:hover{color:var(--danger);}
  #menu .sep{height:1px;background:var(--line);margin:4px 4px;}
  #menu .mhint{font-family:var(--mono);font-size:9px;letter-spacing:.12em;text-transform:uppercase;color:var(--faint);padding:7px 11px 3px;}
  #dlgwrap{position:fixed;inset:0;display:none;align-items:center;justify-content:center;z-index:95;padding:24px;background:rgba(8,6,3,.5);}
  #dlgwrap.show{display:flex;}
  .dlg{background:var(--panel);border-radius:14px;box-shadow:var(--shadow);width:400px;max-width:100%;padding:22px;}
  .dlg h4{margin:0 0 8px;font-family:var(--serif);font-weight:500;font-size:19px;}
  .dlg p{margin:0 0 14px;font-family:var(--mono);font-size:12px;color:var(--muted);line-height:1.5;}
  .dlg input,.dlg select{width:100%;background:var(--surface);color:var(--text);border:none;border-radius:9px;padding:11px;font-size:14px;font-family:var(--mono);margin-bottom:6px;}
  .dlg input:focus,.dlg select:focus{outline:2px solid var(--accent-weak);}
  .dlg .dlg-btns{display:flex;gap:8px;justify-content:flex-end;margin-top:14px;}
  #toast{position:fixed;bottom:96px;left:50%;transform:translateX(-50%) translateY(8px);background:var(--surface3);padding:10px 16px;border-radius:10px;font-family:var(--mono);font-size:12px;opacity:0;transition:opacity .25s,transform .25s;pointer-events:none;z-index:99;box-shadow:var(--shadow);letter-spacing:.03em;}
  #toast.show{opacity:1;transform:translateX(-50%) translateY(0);}

  /* responsive */
  @media (max-width:860px){
    #sidebar{position:fixed;left:0;top:0;height:100dvh;height:100vh;transform:translateX(-103%);transition:transform .22s ease;box-shadow:var(--shadow);z-index:80;}
    #sidebar.show{transform:translateX(0);}
    #app.sbcollapsed #sidebar{display:flex;}
    #resizer{display:none;}
    .menubtn{display:flex;} #revealbtn{display:none!important;} #collapsebtn{display:none;}
    #charchip,select.msel{display:none;}
    .wrap,.chint{padding:0 16px;}
    .bubble{font-size:calc(17px*var(--rs));}
    .actions{opacity:1;}
    .convo .cmenu,.folder-head .fmenu{opacity:1;}
  }
  @media (max-width:560px){ .params-grid{grid-template-columns:1fr;} .barbtn .t{display:none;} }
</style></head>
<body>
"""

PAGE_BODY = r"""
  <div id="app">
    <aside id="sidebar">
      <div id="resizer"></div>
      <div class="side-head">
        <span class="mark">ORACLE</span><span class="grow"></span>
        <button class="iconbtn" id="collapsebtn" title="collapse sidebar">&laquo;</button>
      </div>
      <div class="side-act">
        <button id="newbtn">+ new conversation</button>
        <input id="searchbox" placeholder="search…">
        <div class="row"><button id="foldernew" title="new folder">+ folder</button><button id="selbtn" title="select multiple">select</button></div>
      </div>
      <div id="selbar"><span class="cnt" id="selcnt">0 selected</span><button id="selmove">move</button><button id="seldel" class="danger">delete</button><button id="seldone">done</button></div>
      <div id="tree"></div>
      <div class="side-foot">
        <div class="who" id="whobtn"><div class="nm" id="who-nm">-</div><div class="rl" id="who-rl"></div></div>
        <button class="iconbtn" id="themebtn" title="toggle theme"><svg class="ico" id="themeicon" viewBox="0 0 24 24"></svg></button>
        <button class="iconbtn" id="settingsbtn" title="settings"><svg class="ico" viewBox="0 0 24 24"><path d="M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3M1 14h6M9 8h6M17 12h6"/></svg></button>
        <button class="iconbtn" id="logoutbtn" title="sign out"><svg class="ico" viewBox="0 0 24 24"><path d="M16 17l5-5-5-5M21 12H9M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/></svg></button>
      </div>
    </aside>

    <main id="main">
      <header id="bar">
        <button class="iconbtn menubtn" id="menubtn" title="menu"><svg class="ico" viewBox="0 0 24 24"><path d="M3 6h18M3 12h18M3 18h18"/></svg></button>
        <button class="iconbtn" id="revealbtn" title="show sidebar">&raquo;</button>
        <div id="titlewrap"><span id="title">oracle</span><div id="submeta"></div></div>
        <div class="barbtns">
          <button class="barbtn" id="exportbtn" title="export this chat">export</button>
          <button class="barbtn" id="charchip"><span class="t">character</span></button>
          <select class="msel" id="modelsel" title="model"></select>
          <button class="barbtn" id="tunebtn">tune</button>
        </div>
      </header>
      <div id="log"></div>
      <footer id="composer">
        <div class="wrap">
          <div class="input-shell">
            <textarea id="input" rows="1" placeholder="say something…   (enter to send, shift+enter for newline)"></textarea>
          </div>
          <button id="send" title="send"><svg class="ico" viewBox="0 0 24 24"><path d="M5 12h14M13 6l6 6-6 6"/></svg></button>
        </div>
        <div class="chint" id="chint"></div>
      </footer>
    </main>
  </div>

  <div id="drawer" class="panel">
    <div class="phead"><h3>chat settings</h3><button class="x" data-close-drawer>&times;</button></div>
    <div class="pbody">
      <label class="fld"><span class="lab">character</span><select id="d_char"></select></label>
      <label class="fld" id="d_endpoint_wrap"><span class="lab">endpoint <span class="sub">(admin)</span></span><select id="d_endpoint"></select></label>
      <label class="fld"><span class="lab">model</span><select id="d_model"></select></label>
      <label class="fld"><span class="lab">system prompt <span class="sub">this chat only</span></span><textarea id="d_system" placeholder="empty = no system prompt"></textarea></label>
      <div class="fld params-section"><span class="lab">sampler parameters <span class="sub">blank = server default</span></span><div class="params-grid" id="d_params"></div>
        <div class="params-foot"><button class="mini" id="d_defaults">server defaults</button><button class="mini" id="d_clear">clear all</button></div>
      </div>
    </div>
    <div class="pfoot"><button class="btn-ghost" data-close-drawer>cancel</button><button class="btn-primary" id="d_save">save</button></div>
  </div>

  <div id="modal">
    <div class="modal-card">
      <div class="mhead"><h3>settings</h3><button class="x" data-close-modal>&times;</button></div>
      <div class="tabs" id="tabs"></div>
      <div class="tab-body">
        <div class="tab-pane" id="tab-account">
          <div class="lbl" style="margin-bottom:10px;">change password</div>
          <label class="fld"><span class="lab">current password</span><input type="password" id="ac_old"></label>
          <label class="fld"><span class="lab">new password <span class="sub">(min 8)</span></span><input type="password" id="ac_new"></label>
          <label class="fld"><span class="lab">confirm new</span><input type="password" id="ac_new2"></label>
          <div class="edit-row" style="margin-top:14px;"><button class="btn-primary" id="ac_save">change password</button></div>
          <div class="lbl" style="margin:24px 0 8px;">your data</div>
          <button class="mini" id="exportall">export all my chats (json)</button>
          <div class="danger-zone">
            <div class="dz-t">danger zone</div>
            <p>Delete your account and every conversation, folder, and private character you own. This cannot be undone.</p>
            <button class="btn-danger" id="acctdel">delete my account</button>
          </div>
        </div>
        <div class="tab-pane" id="tab-appearance">
          <label class="fld"><span class="lab">theme</span><div class="seg" id="themeseg"><button data-th="dark" class="on">dark</button><button data-th="light">light</button></div></label>
          <label class="fld"><span class="lab">body text</span><div class="seg" id="fontseg"><button data-f="serif" class="on">serif</button><button data-f="sans">sans</button><button data-f="mono">mono</button></div></label>
          <label class="fld"><span class="lab">text size</span>
            <div class="pgrow" style="gap:8px;"><button class="mini" id="fs_minus">A-</button><input type="range" id="fs_range" min="0.85" max="1.5" step="0.05" style="flex:1;accent-color:var(--accent);"><button class="mini" id="fs_plus">A+</button><button class="mini" id="fs_reset">reset</button></div>
          </label>
          <label class="fld"><span class="lab">text width</span>
            <div class="pgrow" style="gap:8px;"><input type="range" id="cw_range" min="620" max="1200" step="20" style="flex:1;accent-color:var(--accent);"><button class="mini" id="cw_reset">reset</button></div>
          </label>
          <div class="hintbox">Appearance is stored in this browser only.</div>
        </div>
        <div class="tab-pane" id="tab-characters">
          <div id="char-list"></div>
          <button class="mini" id="char-add">+ new character</button>
          <div id="char-edit" style="display:none;margin-top:14px;">
            <input type="hidden" id="ch_id">
            <label class="fld"><span class="lab">name</span><input type="text" id="ch_name" placeholder="e.g. Artaud"></label>
            <label class="fld"><span class="lab">mark <span class="sub">1-2 chars/glyph, optional</span></span><input type="text" id="ch_avatar" maxlength="2" placeholder="A"></label>
            <label class="fld"><span class="lab">default model <span class="sub">optional</span></span><select id="ch_model"></select></label>
            <label class="fld"><span class="lab">system prompt</span><textarea id="ch_system"></textarea></label>
            <div class="toggle-row" id="ch_sitewrap" style="display:none;"><input type="checkbox" id="ch_site"><label for="ch_site" style="cursor:pointer;">make site-wide (all users, read-only to them)</label></div>
            <div class="edit-row" style="margin-top:14px;"><button class="btn-primary" id="ch_save">save character</button><button class="btn-ghost" id="ch_cancel">cancel</button></div>
          </div>
        </div>
        <div class="tab-pane" id="tab-models">
          <div class="hintbox" style="margin-top:0;">Models your non-admin users may use. Admins always see every model; per-user overrides live in Users.</div>
          <div class="model-pick" id="um_pick"></div>
        </div>
        <div class="tab-pane" id="tab-endpoints">
          <div id="ep-list"></div>
          <button class="mini" id="ep-add">+ add endpoint</button>
        </div>
        <div class="tab-pane" id="tab-defaults">
          <label class="fld"><span class="lab">default model <span class="sub">for new chats</span></span><input type="text" id="def_model"></label>
          <label class="fld"><span class="lab">default system prompt</span><textarea id="def_system"></textarea></label>
          <div class="fld"><span class="lab">default sampler parameters</span><div class="params-grid" id="def_params"></div>
            <div class="params-foot"><button class="mini" id="def_defaults">server defaults</button><button class="mini" id="def_clear">clear</button></div>
          </div>
        </div>
        <div class="tab-pane" id="tab-users">
          <div id="user-list"></div>
          <button class="mini" id="user-add">+ add user</button>
          <div id="user-edit" style="display:none;margin-top:14px;">
            <input type="hidden" id="us_id">
            <div id="us_newonly">
              <label class="fld"><span class="lab">username</span><input type="text" id="us_name" placeholder="friend"></label>
              <label class="fld"><span class="lab">password <span class="sub">(min 8)</span></span><input type="password" id="us_pw"></label>
            </div>
            <label class="fld"><span class="lab">role</span><select id="us_role"><option value="user">user</option><option value="admin">admin</option></select></label>
            <div class="fld" id="us_models_wrap"><span class="lab">allowed models <span class="sub">none checked = use global list</span></span><div class="model-pick" id="us_models"></div></div>
            <label class="fld" id="us_reset_wrap" style="display:none;"><span class="lab">reset password <span class="sub">blank = keep</span></span><input type="password" id="us_reset"></label>
            <div class="edit-row" style="margin-top:14px;"><button class="btn-primary" id="us_save">save</button><button class="btn-ghost" id="us_cancel">cancel</button></div>
          </div>
        </div>
      </div>
      <div class="pfoot" style="padding:14px 20px;display:flex;gap:8px;">
        <span style="flex:1;color:var(--faint);font-family:var(--mono);font-size:11px;align-self:center;" id="settings-note"></span>
        <button class="btn-ghost" data-close-modal>close</button>
        <button class="btn-primary" id="settings-save" style="display:none;">save settings</button>
      </div>
    </div>
  </div>

  <div id="menu"></div>
  <div id="dlgwrap"><div class="dlg" id="dlg"></div></div>
  <div id="backdrop"></div>
  <div id="toast"></div>
"""

PAGE_JS1 = r"""<script>
"use strict";
const $=s=>document.querySelector(s), $$=s=>Array.from(document.querySelectorAll(s));
let CFG=null, current=null, busy=false, activeController=null;
let convoCache=[], folderCache=[], collapsed={}, selMode=false, selected=new Set();

const ICON_SEND='<svg class="ico" viewBox="0 0 24 24"><path d="M5 12h14M13 6l6 6-6 6"/></svg>';
const ICON_STOP='<svg class="ico" viewBox="0 0 24 24"><rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor" stroke="none"/></svg>';
const ICON_THUMB='<svg class="ico" viewBox="0 0 24 24"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"/></svg>';
const ICON_MOON='<path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/>';
const ICON_SUN='<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>';

function esc(s){return (s==null?"":String(s)).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function fmtNum(n){return n==null?"?":Number(n).toLocaleString();}
function isAdmin(){return CFG&&CFG.is_admin;}
function toast(m){const t=$("#toast");t.textContent=m;t.classList.add("show");clearTimeout(toast._t);toast._t=setTimeout(()=>t.classList.remove("show"),2600);}
async function api(method,path,body){
  const o={method,headers:{"Content-Type":"application/json"}};
  if(body!==undefined)o.body=JSON.stringify(body);
  const r=await fetch(path,o);
  if(r.status===401){location.href="/login";throw new Error("session expired");}
  const j=await r.json().catch(()=>({}));
  if(!r.ok)throw new Error(j.error||("HTTP "+r.status));
  return j;
}
function download(name,text,type){
  const blob=new Blob([text],{type:type||"text/plain"});const u=URL.createObjectURL(blob);
  const a=document.createElement("a");a.href=u;a.download=name;document.body.appendChild(a);a.click();a.remove();
  setTimeout(()=>URL.revokeObjectURL(u),1000);
}
function fmtTime(iso){if(!iso)return "";const d=new Date(iso);if(isNaN(d))return "";
  const now=new Date(),same=d.toDateString()===now.toDateString();
  const hm=d.toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'});
  return same?hm:d.toLocaleDateString([], {month:'short',day:'numeric'})+" "+hm;}
function relTime(iso){if(!iso)return "";const d=new Date(iso);if(isNaN(d))return "";
  const s=(Date.now()-d.getTime())/1000;
  if(s<60)return "now";if(s<3600)return Math.floor(s/60)+"m";if(s<86400)return Math.floor(s/3600)+"h";
  if(s<604800)return Math.floor(s/86400)+"d";return d.toLocaleDateString([], {month:'short',day:'numeric'});}

// ---------------- appearance
function applyTheme(t){document.documentElement.setAttribute("data-theme",t);localStorage.setItem("oracle_theme",t);
  $("#themeicon").innerHTML=(t==="dark")?ICON_MOON:ICON_SUN;
  $$("#themeseg button").forEach(b=>b.classList.toggle("on",b.dataset.th===t));}
function curTheme(){return document.documentElement.getAttribute("data-theme")||"dark";}
function applyFS(v){v=Math.max(0.85,Math.min(1.5,v));document.documentElement.style.setProperty("--rs",v);localStorage.setItem("oracle_fs",v);
  const r=$("#fs_range");if(r)r.value=v;}
function curFS(){return parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--rs"))||1;}
const FONTS={serif:"var(--serif)",sans:"-apple-system,'Segoe UI',system-ui,sans-serif",mono:"var(--mono)"};
function applyFont(f){if(!FONTS[f])f="serif";document.documentElement.style.setProperty("--read-font",FONTS[f]);localStorage.setItem("oracle_font",f);$$("#fontseg button").forEach(b=>b.classList.toggle("on",b.dataset.f===f));}
function curFont(){return localStorage.getItem("oracle_font")||"serif";}
function applyCW(v){v=Math.max(620,Math.min(1200,v));document.documentElement.style.setProperty("--cw",v+"px");localStorage.setItem("oracle_cw",v);const r=$("#cw_range");if(r)r.value=v;}
function curCW(){return parseInt(getComputedStyle(document.documentElement).getPropertyValue("--cw"))||840;}

// ---------------- styled dialogs (replace native prompt/confirm)
function closeDlg(){$("#dlgwrap").classList.remove("show");$("#dlg").innerHTML="";}
function dialog(html,onmount){return new Promise(res=>{
  const w=$("#dlgwrap"),d=$("#dlg");d.innerHTML=html;w.classList.add("show");
  const done=v=>{closeDlg();res(v);};
  d._resolve=done;
  const esckey=e=>{if(e.key==="Escape"){document.removeEventListener("keydown",esckey);done(null);}};
  document.addEventListener("keydown",esckey);
  w.onclick=e=>{if(e.target===w){document.removeEventListener("keydown",esckey);done(null);}};
  if(onmount)onmount(d,done);
});}
function uiConfirm(message,{title="Confirm",danger=false,ok="Confirm"}={}){
  return dialog('<h4>'+esc(title)+'</h4><p>'+esc(message)+'</p><div class="dlg-btns"><button class="btn-ghost" data-x>cancel</button><button class="'+(danger?"btn-danger":"btn-primary")+'" data-ok>'+esc(ok)+'</button></div>',
    (d,done)=>{d.querySelector("[data-x]").onclick=()=>done(false);d.querySelector("[data-ok]").onclick=()=>done(true);});
}
function uiPrompt(message,def="",{title="",ok="Save"}={}){
  return dialog((title?'<h4>'+esc(title)+'</h4>':'')+'<p>'+esc(message)+'</p><input id="dlgin" value="'+esc(def)+'"><div class="dlg-btns"><button class="btn-ghost" data-x>cancel</button><button class="btn-primary" data-ok>'+esc(ok)+'</button></div>',
    (d,done)=>{const inp=d.querySelector("#dlgin");inp.focus();inp.select();
      inp.onkeydown=e=>{if(e.key==="Enter")done(inp.value);};
      d.querySelector("[data-x]").onclick=()=>done(null);d.querySelector("[data-ok]").onclick=()=>done(inp.value);});
}
function uiSelect(message,options,{title="Move"}={}){
  const opts=options.map(o=>'<option value="'+esc(o.value)+'">'+esc(o.label)+'</option>').join("");
  return dialog('<h4>'+esc(title)+'</h4><p>'+esc(message)+'</p><select id="dlgsel">'+opts+'</select><div class="dlg-btns"><button class="btn-ghost" data-x>cancel</button><button class="btn-primary" data-ok>ok</button></div>',
    (d,done)=>{const sel=d.querySelector("#dlgsel");sel.focus();
      d.querySelector("[data-x]").onclick=()=>done(null);d.querySelector("[data-ok]").onclick=()=>done(sel.value);});
}

// ---------------- markdown
function md(src){
  if(!src)return "";
  let s=src.replace(/\r\n/g,"\n").replace(/\r/g,"\n");
  const code=[];s=s.replace(/```[ \t]*([a-zA-Z0-9_+\-]*)\n?([\s\S]*?)```/g,(m,l,b)=>{code.push(b);return "@@C"+(code.length-1)+"@@";});
  const ic=[];s=s.replace(/`([^`\n]+)`/g,(m,c)=>{ic.push(c);return "@@I"+(ic.length-1)+"@@";});
  const inline=t=>{t=esc(t);
    t=t.replace(/\[([^\]]+)\]\((https?:[^\s)]+)\)/g,'<a href="$2" target="_blank" rel="noopener">$1</a>');
    t=t.replace(/\*\*([^*]+)\*\*/g,"<strong>$1</strong>").replace(/__([^_]+)__/g,"<strong>$1</strong>");
    t=t.replace(/(^|[^*\w])\*([^*\n]+)\*/g,"$1<em>$2</em>").replace(/(^|[^_\w])_([^_\n]+)_/g,"$1<em>$2</em>");
    t=t.replace(/@@I(\d+)@@/g,(m,i)=>"<code>"+esc(ic[+i])+"</code>");return t;};
  const lines=s.split("\n");let out=[],i=0,para=[];
  const flush=()=>{if(para.length){out.push("<p>"+para.map(inline).join("<br>")+"</p>");para=[];}};
  while(i<lines.length){let ln=lines[i];
    let cb=ln.match(/^@@C(\d+)@@\s*$/);if(cb){flush();out.push("<pre><code>"+esc(code[+cb[1]])+"</code></pre>");i++;continue;}
    if(/^\s*$/.test(ln)){flush();i++;continue;}
    let h=ln.match(/^(#{1,6})\s+(.*)$/);if(h){flush();const lv=Math.min(4,h[1].length);out.push("<h"+lv+">"+inline(h[2])+"</h"+lv+">");i++;continue;}
    if(/^\s*([-*_])(\s*\1){2,}\s*$/.test(ln)){flush();out.push("<hr>");i++;continue;}
    if(/^\s*>/.test(ln)){flush();let q=[];while(i<lines.length&&/^\s*>/.test(lines[i])){q.push(lines[i].replace(/^\s*>\s?/,""));i++;}out.push("<blockquote>"+q.map(inline).join("<br>")+"</blockquote>");continue;}
    if(/^\s*[-*+]\s+/.test(ln)){flush();let it=[];while(i<lines.length&&/^\s*[-*+]\s+/.test(lines[i])){it.push(lines[i].replace(/^\s*[-*+]\s+/,""));i++;}out.push("<ul>"+it.map(x=>"<li>"+inline(x)+"</li>").join("")+"</ul>");continue;}
    if(/^\s*\d+\.\s+/.test(ln)){flush();let it=[];while(i<lines.length&&/^\s*\d+\.\s+/.test(lines[i])){it.push(lines[i].replace(/^\s*\d+\.\s+/,""));i++;}out.push("<ol>"+it.map(x=>"<li>"+inline(x)+"</li>").join("")+"</ol>");continue;}
    para.push(ln);i++;}
  flush();let html=out.join("\n");
  html=html.replace(/@@C(\d+)@@/g,(m,i)=>"<pre><code>"+esc(code[+i])+"</code></pre>").replace(/@@I(\d+)@@/g,(m,i)=>"<code>"+esc(ic[+i])+"</code>");
  return html;
}

// ---------------- config
function charById(id){return (CFG.characters||[]).find(c=>c.id===id)||null;}
function endpointName(id){return ((CFG.settings&&CFG.settings.endpoints)||[]).find(e=>e.id===id);}
async function loadConfig(){
  CFG=await api("GET","/api/config");
  CFG.characters=CFG.characters||[];CFG.models=CFG.models||[];folderCache=CFG.folders||[];
  $("#who-nm").textContent=CFG.me.username;$("#who-rl").textContent=CFG.me.role;
  if(!isAdmin())$("#d_endpoint_wrap").style.display="none";
  buildTabs();
  rebuildModelSelect($("#modelsel"),CFG.models);
}
function rebuildModelSelect(sel,models,value){
  const cur=value!==undefined?value:sel.value;sel.innerHTML="";const seen=new Set();
  (models||[]).forEach(m=>{if(seen.has(m))return;seen.add(m);const o=document.createElement("option");o.value=m;o.textContent=m;sel.appendChild(o);});
  if(cur&&!seen.has(cur)){const o=document.createElement("option");o.value=cur;o.textContent=cur;sel.insertBefore(o,sel.firstChild);}
  if(cur)sel.value=cur;
}

// ---------------- sidebar / folders / multiselect
async function refreshList(){const j=await api("GET","/api/conversations");convoCache=j.conversations;renderTree();}
async function refreshFolders(){const j=await api("GET","/api/folders");folderCache=j.folders;renderTree();}
function updateSelbar(){$("#selcnt").textContent=selected.size+" selected";}
function toggleSel(id,row){if(selected.has(id)){selected.delete(id);row.classList.remove("checked");}else{selected.add(id);row.classList.add("checked");}updateSelbar();}
function setSelMode(on){selMode=on;selected.clear();$("#app").classList.toggle("selmode",on);$("#selbtn").classList.toggle("on",on);updateSelbar();renderTree();}
function renderTree(){
  const q=($("#searchbox").value||"").toLowerCase().trim();
  const tree=$("#tree");tree.innerHTML="";
  const match=c=>!q||(c.title||"").toLowerCase().includes(q);
  const byF={},unfiled=[];
  convoCache.filter(match).forEach(c=>{if(c.folder_id){(byF[c.folder_id]=byF[c.folder_id]||[]).push(c);}else unfiled.push(c);});
  folderCache.forEach(f=>{
    const items=byF[f.id]||[];if(q&&!items.length)return;
    const fol=document.createElement("div");fol.className="folder";fol.dataset.folder=f.id;
    const col=collapsed[f.id];
    fol.innerHTML='<div class="folder-head'+(col?' collapsed':'')+'"><span class="tw">&#9662;</span><span class="fname">'+esc(f.name)+'</span><span class="cnt">'+items.length+'</span><button class="fmenu">&#8943;</button></div><div class="folder-body'+(col?' hidden':'')+'"></div>';
    const head=fol.querySelector(".folder-head"),body=fol.querySelector(".folder-body");
    head.onclick=e=>{if(e.target.classList.contains("fmenu"))return;collapsed[f.id]=!collapsed[f.id];renderTree();};
    head.querySelector(".fmenu").onclick=e=>{e.stopPropagation();folderMenu(e,f);};
    items.forEach(c=>body.appendChild(convoRow(c)));
    setupDrop(fol,f.id);tree.appendChild(fol);
  });
  const uf=document.createElement("div");uf.className="folder";uf.dataset.folder="";
  if(folderCache.length)uf.innerHTML='<div class="lbl" style="padding:10px 7px 4px;">unfiled</div>';
  if(!unfiled.length&&!convoCache.length)uf.innerHTML+='<div class="empty-list">'+(q?'no matches':'no conversations yet')+'</div>';
  unfiled.forEach(c=>uf.appendChild(convoRow(c)));
  setupDrop(uf,"");tree.appendChild(uf);
}
function convoRow(c){
  const d=document.createElement("div");
  d.className="convo"+(current&&c.id===current.id?" active":"")+(selected.has(c.id)?" checked":"");
  d.dataset.id=c.id;d.draggable=!selMode;
  const ch=charById(c.character_id);
  d.innerHTML='<span class="sel"></span><div class="ct">'+esc(c.title||"untitled")+'</div><div class="cm">'+
    (ch?esc((ch.avatar?ch.avatar+" ":"")+ch.name)+' · ':'')+esc(c.model||"")+' · '+relTime(c.updated)+'</div><button class="cmenu">&#8943;</button>';
  d.onclick=e=>{if(e.target.classList.contains("cmenu"))return;
    if(selMode){toggleSel(c.id,d);return;}openConvo(c.id);closeSidebar();};
  d.querySelector(".cmenu").onclick=e=>{e.stopPropagation();convoMenu(e,c);};
  d.addEventListener("dragstart",e=>{e.dataTransfer.setData("text/plain",c.id);d.classList.add("dragging");});
  d.addEventListener("dragend",()=>d.classList.remove("dragging"));
  return d;
}
function setupDrop(zone,folderId){
  zone.addEventListener("dragover",e=>{e.preventDefault();zone.classList.add("dragover");});
  zone.addEventListener("dragleave",()=>zone.classList.remove("dragover"));
  zone.addEventListener("drop",async e=>{e.preventDefault();zone.classList.remove("dragover");
    const id=e.dataTransfer.getData("text/plain");if(!id)return;
    try{const c=await api("POST","/api/conversations/"+id+"/settings",{folder_id:folderId||null});
      if(current&&current.id===id)current=c;await refreshList();toast("moved");}catch(err){toast(err.message);}});
}
async function bulkMove(){
  if(!selected.size)return;
  const opts=[{value:"",label:"unfiled"}].concat(folderCache.map(f=>({value:f.id,label:f.name})));
  const fid=await uiSelect(selected.size+" conversation(s) to:",opts,{title:"Move to folder"});
  if(fid===null)return;
  for(const id of selected){try{await api("POST","/api/conversations/"+id+"/settings",{folder_id:fid||null});}catch(_){}}
  setSelMode(false);refreshList();toast("moved");
}
async function bulkDelete(){
  if(!selected.size)return;
  if(!await uiConfirm("Delete "+selected.size+" conversation(s)? This cannot be undone.",{danger:true,ok:"Delete"}))return;
  for(const id of selected){try{await api("DELETE","/api/conversations/"+id);if(current&&current.id===id){current=null;renderEmpty();}}catch(_){}}
  setSelMode(false);refreshList();toast("deleted");
}

// ---------------- context menus
function showMenu(ev,html,wire){
  const m=$("#menu");m.innerHTML=html;m.classList.add("show");
  const x=Math.min(ev.clientX,window.innerWidth-210),y=Math.min(ev.clientY,window.innerHeight-260);
  m.style.left=Math.max(8,x)+"px";m.style.top=Math.max(8,y)+"px";wire(m);
}
function hideMenu(){$("#menu").classList.remove("show");}
function convoMenu(ev,c){
  let fopts=folderCache.map(f=>'<button data-mv="'+f.id+'">&rarr; '+esc(f.name)+'</button>').join("");
  if(c.folder_id)fopts+='<button data-mv="">&rarr; unfiled</button>';
  showMenu(ev,'<button data-a="open">open</button><button data-a="rename">rename</button>'+
    '<button data-a="exmd">export markdown</button><button data-a="exjson">export json</button>'+
    '<div class="sep"></div><div class="mhint">move to folder</div>'+(fopts||'<div class="mhint" style="color:var(--dim)">no folders</div>')+
    '<div class="sep"></div><button class="danger" data-a="del">delete</button>',
    m=>{
      m.querySelectorAll("[data-mv]").forEach(b=>b.onclick=async()=>{hideMenu();const c2=await api("POST","/api/conversations/"+c.id+"/settings",{folder_id:b.dataset.mv||null});if(current&&current.id===c.id)current=c2;refreshList();toast("moved");});
      m.querySelector('[data-a="open"]').onclick=()=>{hideMenu();openConvo(c.id);closeSidebar();};
      m.querySelector('[data-a="rename"]').onclick=async()=>{hideMenu();const t=await uiPrompt("Rename conversation",c.title||"",{title:"Rename"});if(t!=null&&t.trim()){const c2=await api("POST","/api/conversations/"+c.id+"/settings",{title:t.trim()});if(current&&current.id===c.id){current=c2;syncBar();}refreshList();}};
      m.querySelector('[data-a="exmd"]').onclick=async()=>{hideMenu();exportChat(c.id,"md");};
      m.querySelector('[data-a="exjson"]').onclick=async()=>{hideMenu();exportChat(c.id,"json");};
      m.querySelector('[data-a="del"]').onclick=async()=>{hideMenu();if(!await uiConfirm("Delete this conversation?",{danger:true,ok:"Delete"}))return;await api("DELETE","/api/conversations/"+c.id);if(current&&current.id===c.id){current=null;renderEmpty();}refreshList();toast("deleted");};
    });
}
function folderMenu(ev,f){
  showMenu(ev,'<button data-a="rename">rename folder</button><div class="sep"></div><button class="danger" data-a="del">delete folder</button>',
    m=>{
      m.querySelector('[data-a="rename"]').onclick=async()=>{hideMenu();const n=await uiPrompt("Rename folder",f.name,{title:"Rename folder"});if(n!=null&&n.trim())await api("POST","/api/folders/"+f.id,{name:n.trim()});refreshFolders();};
      m.querySelector('[data-a="del"]').onclick=async()=>{hideMenu();if(!await uiConfirm("Delete folder '"+f.name+"'? Chats inside move to unfiled.",{danger:true,ok:"Delete"}))return;await api("DELETE","/api/folders/"+f.id);refreshFolders();refreshList();};
    });
}
document.addEventListener("click",e=>{if(!$("#menu").contains(e.target))hideMenu();});

async function exportChat(id,fmt){
  const c=await api("GET","/api/conversations/"+id);
  const safe=(c.title||"chat").replace(/[^a-z0-9]+/gi,"-").slice(0,50).replace(/^-|-$/g,"")||"chat";
  if(fmt==="json"){download(safe+".json",JSON.stringify(c,null,2),"application/json");return;}
  let out="# "+(c.title||"untitled")+"\n\n";
  (c.messages||[]).forEach(m=>{out+="## "+(m.role==="user"?(CFG.me.username||"user"):"oracle")+"  ·  "+(m.ts||"")+"\n\n"+(m.content||"")+"\n\n";});
  download(safe+".md",out,"text/markdown");
}
"""

PAGE_JS2 = r"""
// ---------------- render conversation
function renderEmpty(){
  $("#log").innerHTML='<div class="wrap"><div class="empty"><div class="glyph">&#8258;</div><h2>nothing selected</h2><p>start a new conversation, or pick one</p></div></div>';
  $("#title").textContent="oracle";$("#submeta").textContent="";$("#chint").textContent="";
}
function syncBar(){
  if(!current)return;
  $("#title").textContent=current.title||"untitled";
  rebuildModelSelect($("#modelsel"),CFG.models,current.model);
  const ch=charById(current.character_id);
  $("#charchip").innerHTML='<span class="t">'+esc(ch?((ch.avatar?ch.avatar+" ":"")+ch.name):"character")+'</span>';
  const bits=[current.model];if(ch)bits.push(ch.name);
  if(isAdmin()&&current.endpoint_id){const e=endpointName(current.endpoint_id);if(e)bits.push(e.name);}
  const np=Object.keys(current.params||{}).length;if(np)bits.push(np+" param"+(np>1?"s":""));
  $("#submeta").textContent=bits.join("  ·  ");
}
function nearBottom(){const l=$("#log");return l.scrollHeight-l.scrollTop-l.clientHeight<100;}
function scrollDown(){const l=$("#log");l.scrollTop=l.scrollHeight;}
function renderConvo(opts){
  opts=opts||{};
  const stick=(opts.stick!==undefined)?opts.stick:nearBottom();
  const prev=$("#log").scrollTop;syncBar();
  const log=$("#log");log.innerHTML="";
  const wrap=document.createElement("div");wrap.className="wrap";
  const msgs=(current.messages||[]).filter(m=>m.role==="user"||m.role==="assistant");
  if(!msgs.length)wrap.innerHTML='<div class="empty"><div class="glyph">&rsaquo;_</div><h2>new conversation</h2><p>say something to begin</p></div>';
  else current.messages.forEach((m,i)=>{if(m.role==="user"||m.role==="assistant")wrap.appendChild(msgEl(m,i));});
  log.appendChild(wrap);
  if(stick)scrollDown();else log.scrollTop=prev;
}
function metaLine(m){
  if(!m.meta)return "";const x=m.meta,b=[];
  if(m.model)b.push('<span class="pill k">'+esc(m.model)+'</span>');
  if(x.elapsed_ms!=null)b.push('<span class="pill">'+(x.elapsed_ms/1000).toFixed(1)+'s</span>');
  if(x.completion_tokens!=null)b.push('<span class="pill" title="'+(x.tokens_est?'estimated':'reported')+'">'+(x.tokens_est?"~":"")+fmtNum(x.completion_tokens)+' tok</span>');
  if(x.tps)b.push('<span class="pill">'+x.tps+' tok/s</span>');
  let tip=[];if(x.ttft_ms!=null)tip.push("ttft "+(x.ttft_ms/1000).toFixed(2)+"s");if(x.prompt_tokens!=null)tip.push("prompt "+fmtNum(x.prompt_tokens)+" tok");
  return '<div class="meta" title="'+esc(tip.join(" · "))+'">'+b.join("")+'</div>';
}
function sibNav(m){
  if(!m.sib_count||m.sib_count<2)return "";
  const i=m.sib_index;
  return '<span class="sib" title="branch '+(i+1)+' of '+m.sib_count+'"><button data-sib="prev"'+(i<=0?' disabled':'')+'>&lsaquo;</button><span class="n">'+(i+1)+' / '+m.sib_count+'</span><button data-sib="next"'+(i>=m.sib_count-1?' disabled':'')+'>&rsaquo;</button></span>';
}
function msgEl(m,i){
  const d=document.createElement("div");d.className="msg "+m.role;d.dataset.id=m.id;
  const ch=charById(current.character_id);
  const role=m.role==="user"?(CFG.me.username||"you"):(ch?ch.name:"oracle");
  let reason=m.reasoning?'<details class="reason"><summary>reasoning</summary><div class="rbody">'+esc(m.reasoning)+'</div></details>':"";
  const edited=m.edited?' · edited':'';
  const mark=m.rating?'<span class="ratemark '+(m.rating>0?'up':'down')+'" title="'+(m.rating>0?'rated good':'rated bad')+'">'+(m.rating>0?'&#9650;':'&#9660;')+'</span>':'';
  const body=m.role==="assistant"?'<div class="bubble">'+md(m.content)+'</div>':'<div class="bubble raw">'+esc(m.content)+'</div>';
  const rateBtns=m.role==="assistant"?'<button data-act="up" class="rate up'+(m.rating>0?' on':'')+'" title="good response (saved for RLHF)">'+ICON_THUMB+'</button><button data-act="down" class="rate down'+(m.rating<0?' on':'')+'" title="bad response (saved for RLHF)">'+ICON_THUMB+'</button>':'';
  d.innerHTML='<div class="head"><span class="role">'+esc(role)+'</span>'+sibNav(m)+'<span class="tm">'+esc(fmtTime(m.ts)+edited)+'</span>'+mark+'</div>'+
    reason+body+metaLine(m)+
    '<div class="actions"><button data-act="copy">copy</button>'+
    (m.role==="assistant"?'<button data-act="raw">raw</button><button data-act="regen">regenerate</button>'+rateBtns:'')+
    '<button data-act="edit">edit</button><button data-act="del" class="danger">delete</button></div>';
  d.querySelectorAll(".actions button").forEach(b=>b.onclick=()=>handleAction(b.dataset.act,m,d,b));
  d.querySelectorAll("[data-sib]").forEach(b=>b.onclick=()=>{if(b.disabled)return;switchSibling(m.siblings[m.sib_index+(b.dataset.sib==="next"?1:-1)]);});
  return d;
}
function handleAction(act,m,d,btn){
  if(act==="copy"){navigator.clipboard.writeText(m.content);toast("copied");return;}
  if(act==="raw"){const b=d.querySelector(".bubble");if(b.classList.contains("raw")){b.classList.remove("raw");b.innerHTML=md(m.content);btn.textContent="raw";}else{b.classList.add("raw");b.textContent=m.content;btn.textContent="markdown";}return;}
  if(act==="edit"){startEdit(m,d);return;}
  if(act==="regen"){regenerate(m);return;}
  if(act==="up"){rate(m,(m.rating>0)?0:1);return;}
  if(act==="down"){rate(m,(m.rating<0)?0:-1);return;}
  if(act==="del"){deleteMsg(m);return;}
}
async function rate(m,val){try{current=await api("POST","/api/conversations/"+current.id+"/messages/"+m.id+"/rating",{rating:val});renderConvo({stick:false});}catch(e){toast(e.message);}}
async function switchSibling(sid){try{current=await api("POST","/api/conversations/"+current.id+"/active",{message_id:sid});renderConvo({stick:false});}catch(e){toast(e.message);}}
function startEdit(m,d){
  if(d.querySelector(".edit-wrap"))return;
  const bubble=d.querySelector(".bubble");
  const wrap=document.createElement("div");wrap.className="edit-wrap";
  const ta=document.createElement("textarea");ta.className="edit-area";ta.value=m.content;
  const row=document.createElement("div");row.className="edit-row";
  const resend=m.role==="user"?'<button class="btn-primary" data-e="resend">save &amp; branch</button>':'';
  row.innerHTML='<button class="btn-primary" data-e="save">save</button>'+resend+'<button class="btn-ghost" data-e="cancel">cancel</button>';
  wrap.appendChild(ta);wrap.appendChild(row);bubble.style.display="none";bubble.insertAdjacentElement("afterend",wrap);
  ta.focus();ta.style.height="auto";ta.style.height=Math.min(ta.scrollHeight,460)+"px";
  ta.addEventListener("input",()=>{ta.style.height="auto";ta.style.height=Math.min(ta.scrollHeight,460)+"px";});
  row.querySelector('[data-e="cancel"]').onclick=()=>{wrap.remove();bubble.style.display="";};
  row.querySelector('[data-e="save"]').onclick=async()=>{try{current=await api("POST","/api/conversations/"+current.id+"/messages/"+m.id,{content:ta.value});renderConvo();refreshList();toast("saved");}catch(e){toast(e.message);}};
  const rb=row.querySelector('[data-e="resend"]');if(rb)rb.onclick=()=>resendEdited(m.id,ta.value);
}
async function deleteMsg(m){if(!await uiConfirm("Delete this message and everything below it on this branch?",{danger:true,ok:"Delete"}))return;
  try{current=await api("DELETE","/api/conversations/"+current.id+"/messages/"+m.id);renderConvo();refreshList();toast("deleted");}catch(e){toast(e.message);}}

// ---------------- streaming
function appendLive(){
  const stick=nearBottom();let wrap=$("#log").querySelector(".wrap");
  if(!wrap){const l=$("#log");l.innerHTML="";wrap=document.createElement("div");wrap.className="wrap";l.appendChild(wrap);}
  const e=wrap.querySelector(".empty");if(e)e.remove();
  const ch=charById(current.character_id);const role=ch?ch.name:"oracle";
  const d=document.createElement("div");d.className="msg assistant";
  d.innerHTML='<div class="head"><span class="role">'+esc(role)+'</span><span class="tm"></span></div><div class="reason-live" style="display:none"></div><div class="bubble raw"><span class="typing"><i></i><i></i><i></i></span></div>';
  wrap.appendChild(d);if(stick)scrollDown();
  return {bubble:d.querySelector(".bubble"),reason:d.querySelector(".reason-live")};
}
async function streamRequest(path,body,handlers,signal){
  const r=await fetch(path,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body),signal});
  if(r.status===401){location.href="/login";throw new Error("session expired");}
  if(!r.ok){const j=await r.json().catch(()=>({}));throw new Error(j.error||("HTTP "+r.status));}
  const reader=r.body.getReader(),dec=new TextDecoder();let buf="",result=null;
  while(true){const {value,done}=await reader.read();if(done)break;buf+=dec.decode(value,{stream:true});
    let nl;while((nl=buf.indexOf("\n"))>=0){const line=buf.slice(0,nl);buf=buf.slice(nl+1);if(!line.trim())continue;
      let o;try{o=JSON.parse(line);}catch(_){continue;}
      if(o.error)throw new Error(o.error);else if(o.delta!==undefined)handlers.onDelta(o.delta);
      else if(o.reasoning!==undefined)handlers.onReason(o.reasoning);else if(o.done)result=o;}}
  return result;
}
async function streamTurn(body){
  const live=appendLive();let acc="",racc="",started=false;
  const c=new AbortController();activeController=c;
  try{const res=await streamRequest("/api/conversations/"+current.id+"/stream",body,{
      onDelta:d=>{const st=nearBottom();if(!started){live.bubble.textContent="";started=true;}acc+=d;live.bubble.textContent=acc;if(st)scrollDown();},
      onReason:r=>{const st=nearBottom();racc+=r;live.reason.style.display="block";live.reason.innerHTML='<div class="rbody">'+esc(racc)+'</div>';if(st)scrollDown();}
    },c.signal);
    return {res,stopped:false};
  }catch(e){if(e.name==="AbortError")return {res:null,stopped:true};throw e;}
  finally{activeController=null;}
}
function setBusy(b){busy=b;const s=$("#send");s.classList.toggle("stop",b);s.innerHTML=b?ICON_STOP:ICON_SEND;s.title=b?"stop":"send";}
function stopStream(){if(activeController)activeController.abort();}
async function runStream(body,optimistic){
  if(busy||!current)return;setBusy(true);
  if(optimistic)optimistic();
  try{const {res,stopped}=await streamTurn(body);
    if(res&&res.convo){current=res.convo;renderConvo();}else await openConvo(current.id);
    refreshList();
  }catch(e){toast(e.message);try{await openConvo(current.id);}catch(_){}}
  finally{setBusy(false);$("#input").focus();}
}
async function send(){
  const text=$("#input").value.trim();if(!text||busy)return;
  if(!current){current=await api("POST","/api/conversations",{system:CFG.default_system,model:$("#modelsel").value||CFG.default_model});}
  $("#input").value="";$("#input").style.height="auto";
  runStream({content:text},()=>{current.messages.push({id:"tmp",role:"user",content:text,ts:new Date().toISOString()});renderConvo({stick:true});});
}
function regenerate(m){
  const idx=current.messages.findIndex(x=>x.id===m.id);
  runStream({regenerate_id:m.id},()=>{if(idx>=0)current.messages=current.messages.slice(0,idx);renderConvo({stick:true});});
}
function resendEdited(mid,newContent){
  const idx=current.messages.findIndex(x=>x.id===mid);
  runStream({edit_user_id:mid,content:newContent},()=>{if(idx>=0){current.messages=current.messages.slice(0,idx);current.messages.push({id:"tmp",role:"user",content:newContent,ts:new Date().toISOString()});}renderConvo({stick:true});});
}

async function openConvo(id){current=await api("GET","/api/conversations/"+id);renderConvo({stick:true});renderTree();}
async function newChat(){current=await api("POST","/api/conversations",{system:CFG.default_system,model:CFG.default_model});renderConvo();refreshList();$("#input").focus();closeSidebar();}

function editTitle(){
  if(!current)return;const t=$("#title");if(t.classList.contains("editing"))return;
  const old=current.title||"";t.classList.add("editing");t.contentEditable="true";t.textContent=old;t.focus();
  const rg=document.createRange();rg.selectNodeContents(t);const sel=getSelection();sel.removeAllRanges();sel.addRange(rg);
  const fin=async(commit)=>{t.contentEditable="false";t.classList.remove("editing");t.removeEventListener("blur",ob);t.removeEventListener("keydown",ok);
    const v=t.textContent.trim();if(commit&&v&&v!==old){try{current=await api("POST","/api/conversations/"+current.id+"/settings",{title:v});refreshList();}catch(e){toast(e.message);}}t.textContent=current.title||"untitled";};
  const ob=()=>fin(true),ok=e=>{if(e.key==="Enter"){e.preventDefault();t.blur();}if(e.key==="Escape")fin(false);};
  t.addEventListener("blur",ob);t.addEventListener("keydown",ok);
}

// ---------------- param grid (sliders + defaults)
function buildParamsGrid(target,values){
  target.innerHTML="";
  CFG.param_specs.forEach(p=>{
    const v=(values&&values[p.key]!=null)?values[p.key]:"";
    const pg=document.createElement("div");pg.className="pg"+(v===""?" off":"");
    let html='<div class="pgname">'+esc(p.label)+'</div>';
    html+='<div class="pgrow"><input type="number" data-k="'+p.key+'" step="'+(p.step||"any")+'"'+(p.min!=null?' min="'+p.min+'"':'')+(p.max!=null?' max="'+p.max+'"':'')+' placeholder="'+esc(p.ph||"")+'" value="'+(v===""?"":v)+'"><button class="reset" title="reset to server default">reset</button></div>';
    if(p.slider)html+='<input type="range" data-r="'+p.key+'" min="'+p.min+'" max="'+p.max+'" step="'+(p.step||"any")+'" value="'+(v===""?p.default:v)+'">';
    pg.innerHTML=html;
    const num=pg.querySelector("[data-k]"),rng=pg.querySelector("[data-r]"),rst=pg.querySelector(".reset");
    const sync=on=>pg.classList.toggle("off",!on);
    num.addEventListener("input",()=>{if(rng&&num.value!=="")rng.value=num.value;sync(num.value!=="");});
    if(rng)rng.addEventListener("input",()=>{num.value=rng.value;sync(true);});
    rst.onclick=()=>{num.value="";if(rng)rng.value=p.default;sync(false);};
    target.appendChild(pg);
  });
}
function readParamsGrid(target){const out={};target.querySelectorAll("input[data-k]").forEach(inp=>{const raw=inp.value.trim();if(raw==="")return;const sp=CFG.param_specs.find(p=>p.key===inp.dataset.k);out[inp.dataset.k]=sp&&sp.type==="int"?parseInt(raw,10):parseFloat(raw);});return out;}
function fillDefaults(target){const o={};CFG.param_specs.forEach(p=>{o[p.key]=p.default;});buildParamsGrid(target,o);}

// ---------------- chat-settings drawer
async function openDrawer(){
  if(!current){toast("open or start a chat first");return;}
  const cs=$("#d_char");cs.innerHTML='<option value="">none</option>';
  CFG.characters.forEach(c=>{const o=document.createElement("option");o.value=c.id;o.textContent=(c.avatar?c.avatar+" ":"")+c.name+(c.scope==="site"?"  ·site":"");cs.appendChild(o);});
  cs.value=current.character_id||"";
  if(isAdmin()){const es=$("#d_endpoint");es.innerHTML='<option value="">default ('+esc((endpointName(CFG.settings.active_endpoint)||{}).name||"-")+')</option>';
    CFG.settings.endpoints.forEach(e=>{const o=document.createElement("option");o.value=e.id;o.textContent=e.name;es.appendChild(o);});es.value=current.endpoint_id||"";
    es.onchange=()=>refreshDrawerModels(es.value,$("#d_model").value);}
  await refreshDrawerModels(current.endpoint_id,current.model);
  cs.onchange=()=>{const c=charById(cs.value);if(c){$("#d_system").value=c.system||"";if(c.model)rebuildModelSelect($("#d_model"),Array.from($("#d_model").options).map(o=>o.value),c.model);}};
  $("#d_system").value=current.system||"";
  buildParamsGrid($("#d_params"),current.params||{});
  showOverlay($("#drawer"));
}
async function refreshDrawerModels(endpointId,value){
  let models=CFG.models;
  if(isAdmin()&&endpointId){try{const r=await api("GET","/api/models?endpoint="+encodeURIComponent(endpointId));models=r.models;}catch(_){}}
  rebuildModelSelect($("#d_model"),models,value||current.model);
}
async function saveDrawer(){
  const payload={character_id:$("#d_char").value||null,model:$("#d_model").value,system:$("#d_system").value,params:readParamsGrid($("#d_params"))};
  if(isAdmin())payload.endpoint_id=$("#d_endpoint").value||null;
  try{current=await api("POST","/api/conversations/"+current.id+"/settings",payload);closeOverlay($("#drawer"));renderConvo();refreshList();toast("saved");}catch(e){toast(e.message);}
}

// ---------------- sidebar collapse / resize
function applySidebar(){
  const col=localStorage.getItem("oracle_sbcollapsed")==="1";
  $("#app").classList.toggle("sbcollapsed",col);
}
function toggleCollapse(v){const col=v!==undefined?v:!$("#app").classList.contains("sbcollapsed");$("#app").classList.toggle("sbcollapsed",col);localStorage.setItem("oracle_sbcollapsed",col?"1":"0");}
function initResize(){
  const rz=$("#resizer");let dragging=false;
  rz.addEventListener("mousedown",e=>{dragging=true;e.preventDefault();document.body.style.userSelect="none";});
  window.addEventListener("mousemove",e=>{if(!dragging)return;const w=Math.max(210,Math.min(480,e.clientX));document.documentElement.style.setProperty("--sbw",w+"px");localStorage.setItem("oracle_sbw",w);});
  window.addEventListener("mouseup",()=>{if(dragging){dragging=false;document.body.style.userSelect="";}});
}
"""

PAGE_JS3 = r"""
// ---------------- settings modal
function buildTabs(){
  const tabs=[{id:"account",t:"account"},{id:"appearance",t:"appearance"},{id:"characters",t:"characters"}];
  if(isAdmin())tabs.push({id:"models",t:"user models"},{id:"endpoints",t:"endpoints"},{id:"defaults",t:"defaults"},{id:"users",t:"users"});
  $("#tabs").innerHTML=tabs.map((x,i)=>'<button data-tab="'+x.id+'"'+(i===0?' class="active"':'')+'>'+x.t+'</button>').join("");
  $$("#tabs button").forEach(b=>b.onclick=()=>switchTab(b.dataset.tab));
  $("#ch_sitewrap").style.display=isAdmin()?"flex":"none";
}
function switchTab(name){
  $$("#tabs button").forEach(b=>b.classList.toggle("active",b.dataset.tab===name));
  $$(".tab-pane").forEach(p=>p.classList.toggle("active",p.id==="tab-"+name));
  $("#settings-save").style.display=["models","endpoints","defaults"].includes(name)?"block":"none";
  $("#settings-note").textContent="";
  if(name==="users")loadUsers();
}
function openSettings(tab){
  renderCharacters();
  applyTheme(curTheme());applyFont(curFont());if($("#fs_range"))$("#fs_range").value=curFS();if($("#cw_range"))$("#cw_range").value=curCW();
  if(isAdmin()){renderEndpoints();$("#def_model").value=CFG.settings.default_model||"";$("#def_system").value=CFG.settings.default_system||"";buildParamsGrid($("#def_params"),CFG.settings.default_params||{});renderUserModels();}
  switchTab(tab||"account");showModal();
}
// characters
function renderCharacters(){
  const list=$("#char-list");list.innerHTML="";
  if(!CFG.characters.length)list.innerHTML='<div class="empty-list">no characters yet</div>';
  CFG.characters.forEach(c=>{
    const row=document.createElement("div");row.className="row-card";
    row.innerHTML='<div class="cav">'+esc(c.avatar||"§")+'</div><div class="cmain"><div class="cn">'+esc(c.name)+(c.scope==="site"?'<span class="badge">site</span>':'')+'</div><div class="cs">'+esc((c.system||"").slice(0,90))+'</div></div><div class="cbtns">'+(current?'<button class="mini" data-use>use</button>':'')+(c.editable?'<button class="mini" data-edit>edit</button><button class="mini danger" data-del>del</button>':'')+'</div>';
    const ub=row.querySelector("[data-use]");if(ub)ub.onclick=()=>applyCharacter(c);
    const eb=row.querySelector("[data-edit]");if(eb)eb.onclick=()=>editCharacter(c);
    const db2=row.querySelector("[data-del]");if(db2)db2.onclick=async()=>{if(!await uiConfirm("Delete character '"+c.name+"'?",{danger:true,ok:"Delete"}))return;const r=await api("DELETE","/api/characters/"+c.id);CFG.characters=r.characters;renderCharacters();toast("deleted");};
    list.appendChild(row);
  });
}
function editCharacter(c){c=c||{id:"",name:"",avatar:"",model:"",system:"",scope:"private"};
  const ms=$("#ch_model");ms.innerHTML='<option value="">(keep the chat\'s model)</option>';
  (CFG.all_models||CFG.models||[]).forEach(m=>{const o=document.createElement("option");o.value=m;o.textContent=m;ms.appendChild(o);});
  $("#ch_id").value=c.id||"";$("#ch_name").value=c.name||"";$("#ch_avatar").value=c.avatar||"";ms.value=c.model||"";$("#ch_system").value=c.system||"";$("#ch_site").checked=c.scope==="site";
  $("#char-edit").style.display="block";$("#ch_name").focus();}
async function saveCharacter(){
  const ch={id:$("#ch_id").value||undefined,name:$("#ch_name").value.trim()||"Untitled",avatar:$("#ch_avatar").value.trim(),model:$("#ch_model").value.trim()||null,system:$("#ch_system").value,scope:(isAdmin()&&$("#ch_site").checked)?"site":"private"};
  try{const r=await api("POST","/api/characters",{character:ch});CFG.characters=r.characters;$("#char-edit").style.display="none";renderCharacters();toast("saved");}catch(e){toast(e.message);}}
async function applyCharacter(c){
  if(!current){toast("open a chat first");return;}
  const payload={character_id:c.id,system:c.system};if(c.model)payload.model=c.model;
  try{current=await api("POST","/api/conversations/"+current.id+"/settings",payload);renderConvo();refreshList();closeModal();toast("applied: "+c.name);}catch(e){toast(e.message);}}
// user models
function renderUserModels(){const wrap=$("#um_pick");wrap.innerHTML="";const wl=new Set(CFG.settings.user_models||[]);
  (CFG.all_models||CFG.models||[]).forEach(m=>wrap.insertAdjacentHTML("beforeend",'<label><input type="checkbox" value="'+esc(m)+'" '+(wl.has(m)?"checked":"")+'>'+esc(m)+'</label>'));}
function readModelPick(wrap){return Array.from(wrap.querySelectorAll("input:checked")).map(i=>i.value);}
// endpoints
function renderEndpoints(){
  const list=$("#ep-list");list.innerHTML="";
  CFG.settings.endpoints.forEach((ep,idx)=>{
    const card=document.createElement("div");card.className="ep-card";card.dataset.idx=idx;
    card.innerHTML='<div class="ep-top"><input data-f="name" value="'+esc(ep.name||"")+'" placeholder="name"><label class="radio-active"><input type="radio" name="aep" '+(ep.id===CFG.settings.active_endpoint?"checked":"")+'> default</label></div><div class="grid2"><input data-f="url" value="'+esc(ep.url||"")+'" placeholder="https://host:port/v1/chat/completions"><input data-f="models_url" value="'+esc(ep.models_url||"")+'" placeholder="models url (optional)"><input data-f="key" type="password" value="'+esc(ep.key||"")+'" placeholder="api key (optional)"></div><div class="ep-row"><button class="mini" data-test>test</button><button class="mini danger" data-del>remove</button><span class="status"></span></div>';
    card.querySelector(".radio-active input").onchange=()=>{CFG.settings.active_endpoint=ep.id;};
    card.querySelector("[data-del]").onclick=()=>{if(CFG.settings.endpoints.length<=1){toast("keep at least one");return;}CFG.settings.endpoints.splice(idx,1);if(CFG.settings.active_endpoint===ep.id)CFG.settings.active_endpoint=CFG.settings.endpoints[0].id;renderEndpoints();};
    card.querySelector("[data-test]").onclick=async()=>{const st=card.querySelector(".status");st.textContent="testing…";collectEndpoints();
      try{await api("POST","/api/settings",{endpoints:CFG.settings.endpoints});const r=await api("GET","/api/models?endpoint="+encodeURIComponent(ep.id));st.textContent="ok · "+r.models.length+" models";st.style.color="var(--ok)";}catch(e){st.textContent="x "+e.message;st.style.color="var(--danger)";}};
    list.appendChild(card);
  });
}
function collectEndpoints(){$$("#ep-list .ep-card").forEach(card=>{const ep=CFG.settings.endpoints[+card.dataset.idx];if(!ep)return;card.querySelectorAll("[data-f]").forEach(i=>ep[i.dataset.f]=i.value.trim());});}
function addEndpoint(){CFG.settings.endpoints.push({id:"ep-"+Math.random().toString(36).slice(2,8),name:"New endpoint",url:"http://localhost:8000/v1/chat/completions",models_url:"",key:""});renderEndpoints();}
async function saveSettings(){
  collectEndpoints();
  const body={endpoints:CFG.settings.endpoints,active_endpoint:CFG.settings.active_endpoint,
    default_model:$("#def_model").value.trim(),default_system:$("#def_system").value,
    default_params:readParamsGrid($("#def_params")),user_models:readModelPick($("#um_pick"))};
  try{const r=await api("POST","/api/settings",body);CFG.settings=r.settings;CFG.all_models=r.all_models;
    CFG.default_model=r.settings.default_model;CFG.default_system=r.settings.default_system;CFG.default_params=r.settings.default_params;
    $("#settings-note").textContent="saved";toast("settings saved");}catch(e){toast(e.message);}}
// users
let userCache=[];
async function loadUsers(){const j=await api("GET","/api/users");userCache=j.users;renderUsers();}
function renderUsers(){
  const list=$("#user-list");list.innerHTML="";
  userCache.forEach(u=>{const row=document.createElement("div");row.className="row-card";
    const am=u.allowed_models&&u.allowed_models.length?u.allowed_models.join(", "):(u.role==="admin"?"all (admin)":"global default");
    row.innerHTML='<div class="cav">'+esc(u.username.slice(0,2).toUpperCase())+'</div><div class="cmain"><div class="cn">'+esc(u.username)+'<span class="badge">'+esc(u.role)+'</span>'+(u.disabled?'<span class="badge" style="color:var(--danger)">disabled</span>':'')+'</div><div class="cs">models: '+esc(am)+'</div></div><div class="cbtns"><button class="mini" data-edit>edit</button><button class="mini danger" data-del>del</button></div>';
    row.querySelector("[data-edit]").onclick=()=>editUser(u);
    row.querySelector("[data-del]").onclick=async()=>{if(!await uiConfirm("Delete user '"+u.username+"' and ALL their conversations?",{danger:true,ok:"Delete"}))return;try{const r=await api("DELETE","/api/users/"+u.id);userCache=r.users;renderUsers();toast("deleted");}catch(e){toast(e.message);}};
    list.appendChild(row);});
}
function userModelPick(selected){const wrap=$("#us_models");wrap.innerHTML="";const sel=new Set(selected||[]);
  (CFG.all_models||CFG.models||[]).forEach(m=>wrap.insertAdjacentHTML("beforeend",'<label><input type="checkbox" value="'+esc(m)+'" '+(sel.has(m)?"checked":"")+'>'+esc(m)+'</label>'));}
function editUser(u){$("#user-edit").style.display="block";
  if(u){$("#us_id").value=u.id;$("#us_newonly").style.display="none";$("#us_reset_wrap").style.display="block";$("#us_role").value=u.role;userModelPick(u.allowed_models);$("#us_reset").value="";}
  else{$("#us_id").value="";$("#us_newonly").style.display="block";$("#us_reset_wrap").style.display="none";$("#us_name").value="";$("#us_pw").value="";$("#us_role").value="user";userModelPick([]);}}
async function saveUser(){const id=$("#us_id").value;
  try{if(id){const body={role:$("#us_role").value,allowed_models:readModelPick($("#us_models"))};if($("#us_reset").value)body.password=$("#us_reset").value;const r=await api("POST","/api/users/"+id,body);userCache=r.users;}
    else{const r=await api("POST","/api/users",{username:$("#us_name").value.trim(),password:$("#us_pw").value,role:$("#us_role").value,allowed_models:readModelPick($("#us_models"))});userCache=r.users;}
    $("#user-edit").style.display="none";renderUsers();toast("saved");}catch(e){toast(e.message);}}
// account
async function changePassword(){const o=$("#ac_old").value,n=$("#ac_new").value,n2=$("#ac_new2").value;
  if(n!==n2){toast("passwords do not match");return;}
  try{await api("POST","/api/account/password",{old:o,new:n});$("#ac_old").value=$("#ac_new").value=$("#ac_new2").value="";toast("password changed");}catch(e){toast(e.message);}}
async function deleteAccount(){
  if(!await uiConfirm("Delete your account and EVERYTHING you own (chats, folders, private characters)? This cannot be undone.",{danger:true,ok:"Delete account"}))return;
  const c=await uiPrompt("Type your username to confirm:","",{title:"Confirm deletion",ok:"Delete forever"});
  if(c===null)return;if(c!==CFG.me.username){toast("username did not match");return;}
  try{await api("DELETE","/api/account");location.href="/login";}catch(e){toast(e.message);}}

// ---------------- overlays
function showOverlay(p){$("#backdrop").classList.add("show");p.classList.add("show");}
function closeOverlay(p){p.classList.remove("show");if(!$("#modal").classList.contains("show"))$("#backdrop").classList.remove("show");}
function showModal(){$("#backdrop").classList.add("show");$("#modal").classList.add("show");}
function closeModal(){$("#modal").classList.remove("show");if(!$("#drawer").classList.contains("show"))$("#backdrop").classList.remove("show");}
function openSidebar(){$("#sidebar").classList.add("show");$("#backdrop").classList.add("show");}
function closeSidebar(){$("#sidebar").classList.remove("show");if(!$("#modal").classList.contains("show")&&!$("#drawer").classList.contains("show"))$("#backdrop").classList.remove("show");}
function closeAll(){$("#drawer").classList.remove("show");closeModal();$("#sidebar").classList.remove("show");$("#backdrop").classList.remove("show");hideMenu();}

// ---------------- wire up
$("#newbtn").onclick=newChat;
$("#menubtn").onclick=openSidebar;
$("#revealbtn").onclick=()=>toggleCollapse(false);
$("#collapsebtn").onclick=()=>toggleCollapse(true);
$("#searchbox").addEventListener("input",renderTree);
$("#foldernew").onclick=async()=>{const n=await uiPrompt("Name your new folder:","",{title:"New folder",ok:"Create"});if(n!=null){await api("POST","/api/folders",{name:(n.trim()||"New folder")});refreshFolders();toast("folder created");}};
$("#selbtn").onclick=()=>setSelMode(!selMode);
$("#seldone").onclick=()=>setSelMode(false);
$("#selmove").onclick=bulkMove;
$("#seldel").onclick=bulkDelete;
$("#title").onclick=editTitle;
$("#charchip").onclick=()=>openSettings("characters");
$("#tunebtn").onclick=openDrawer;
$("#exportbtn").onclick=e=>{if(!current){toast("open a chat first");return;}showMenu(e,'<button data-f="md">export markdown</button><button data-f="json">export json</button>',m=>{m.querySelectorAll("[data-f]").forEach(b=>b.onclick=()=>{hideMenu();exportChat(current.id,b.dataset.f);});});};
$("#settingsbtn").onclick=()=>openSettings(isAdmin()?"endpoints":"account");
$("#whobtn").onclick=()=>openSettings("account");
$("#themebtn").onclick=()=>applyTheme(curTheme()==="dark"?"light":"dark");
$("#logoutbtn").onclick=async()=>{try{await api("POST","/api/logout");}catch(_){}; location.href="/login";};
$("#modelsel").onchange=async()=>{if(current){try{current=await api("POST","/api/conversations/"+current.id+"/settings",{model:$("#modelsel").value});syncBar();toast("model: "+$("#modelsel").value);}catch(e){toast(e.message);syncBar();}}};
$("#send").onclick=()=>busy?stopStream():send();
$("#d_save").onclick=saveDrawer;
$("#d_defaults").onclick=()=>fillDefaults($("#d_params"));
$("#d_clear").onclick=()=>buildParamsGrid($("#d_params"),{});
$$("[data-close-drawer]").forEach(b=>b.onclick=()=>closeOverlay($("#drawer")));
$$("[data-close-modal]").forEach(b=>b.onclick=closeModal);
$("#backdrop").onclick=closeAll;
$("#ep-add").onclick=addEndpoint;
$("#settings-save").onclick=saveSettings;
$("#char-add").onclick=()=>editCharacter(null);
$("#ch_save").onclick=saveCharacter;
$("#ch_cancel").onclick=()=>$("#char-edit").style.display="none";
$("#def_defaults").onclick=()=>fillDefaults($("#def_params"));
$("#def_clear").onclick=()=>buildParamsGrid($("#def_params"),{});
$("#user-add").onclick=()=>editUser(null);
$("#us_save").onclick=saveUser;
$("#us_cancel").onclick=()=>$("#user-edit").style.display="none";
$("#ac_save").onclick=changePassword;
$("#acctdel").onclick=deleteAccount;
$("#exportall").onclick=()=>{const a=document.createElement("a");a.href="/api/export";a.click();toast("exporting…");};
$$("#themeseg button").forEach(b=>b.onclick=()=>applyTheme(b.dataset.th));
$$("#fontseg button").forEach(b=>b.onclick=()=>applyFont(b.dataset.f));
$("#fs_range").addEventListener("input",e=>applyFS(parseFloat(e.target.value)));
$("#fs_minus").onclick=()=>applyFS(curFS()-0.05);
$("#fs_plus").onclick=()=>applyFS(curFS()+0.05);
$("#fs_reset").onclick=()=>applyFS(1);
$("#cw_range").addEventListener("input",e=>applyCW(parseInt(e.target.value)));
$("#cw_reset").onclick=()=>applyCW(840);
const inp=$("#input");
inp.addEventListener("keydown",e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();if(!busy)send();}});
inp.addEventListener("input",()=>{inp.style.height="auto";inp.style.height=Math.min(inp.scrollHeight,230)+"px";});
document.addEventListener("keydown",e=>{if(e.key==="Escape"){if($("#dlgwrap").classList.contains("show"))return;closeAll();}});

(async function init(){
  applyTheme(localStorage.getItem("oracle_theme")||"dark");
  applyFont(curFont());
  applyFS(parseFloat(localStorage.getItem("oracle_fs"))||1);
  applyCW(parseInt(localStorage.getItem("oracle_cw"))||840);
  applySidebar();initResize();
  try{await loadConfig();}catch(e){toast("load failed: "+e.message);return;}
  await refreshList();renderEmpty();
  if(convoCache.length){try{await openConvo(convoCache[0].id);}catch(_){}}
})();
</script></body></html>"""

PAGE = PAGE_HEAD + PAGE_BODY + PAGE_JS1 + PAGE_JS2 + PAGE_JS3


if __name__ == "__main__":
    init_db()
    bootstrap_admin()
    server = ThreadedHTTPServer(("0.0.0.0", PORT), Handler)
    print("ORACLE chat on http://localhost:%d" % PORT)
    print("database: %s" % DB_PATH)
    if user_count() == 0:
        print("no users yet -> open the page to create the first admin (or set KENOSIS_ADMIN_USER/PASS)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass






