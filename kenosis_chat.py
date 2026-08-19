"""ORACLE a multi-user, authenticated chat front-end for OpenAI-compatible models.

Run locally:   python kenosis_chat.py   then open http://localhost:8770
Production:    Docker + Cloudflare tunnel (see Dockerfile / docker-compose.yml / DEPLOY_CHAT.md).

State lives in one SQLite database (KENOSIS_DB, default chat.db next to this file).
On first run, conversations in ./chat_conversations/*.json are imported to the admin.

Env: KENOSIS_PORT, KENOSIS_DB, KENOSIS_ADMIN_USER/PASS, KENOSIS_SESSION_SECRET,
     KENOSIS_COOKIE_SECURE (1 behind HTTPS), KENOSIS_SESSION_DAYS.
Dependencies: requests + pypdf for PDF attachments (everything else is the standard library).
"""

import ast
import functools as _functools
import http.server
import math
import operator
import socketserver
import json
import os
import re
import time
import uuid
import base64
import gzip
import hmac
import hashlib
import glob
import random
import logging
import sqlite3
import threading
import socket
import ipaddress
import requests
import html as _html_mod
from datetime import datetime
from html.parser import HTMLParser
from urllib.parse import urlparse, parse_qs, urlunparse, urljoin

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
# Sliding-window context management: when a conversation outgrows the model's window we drop the
# oldest turns from what we *send* (the saved chat is untouched) so generation can keep rolling over.
CTX_REPLY_RESERVE = 512   # always keep at least this much room for the answer
CTX_MARGIN = 256          # slack for prompt-token estimation error
REQUEST_TIMEOUT = 1800
PBKDF2_ITERS = 240000

# Brute-force protection: after LOGIN_MAX_FAILS failed sign-ins from one client IP within
# LOGIN_WINDOW seconds, further attempts are refused (429) until the oldest failure ages out.
LOGIN_MAX_FAILS = int(os.environ.get("KENOSIS_LOGIN_MAX_FAILS", "8"))
LOGIN_WINDOW = int(os.environ.get("KENOSIS_LOGIN_WINDOW", "900"))

# Automated SQLite backups (VACUUM INTO). Set KENOSIS_BACKUP_HOURS=0 to disable.
# Backups contain the full database — point KENOSIS_BACKUP_DIR at a path *outside* anything
# web-served or shared.
BACKUP_HOURS = float(os.environ.get("KENOSIS_BACKUP_HOURS", "24"))
BACKUP_KEEP = int(os.environ.get("KENOSIS_BACKUP_KEEP", "7"))
BACKUP_DIR = os.environ.get("KENOSIS_BACKUP_DIR") or os.path.join(os.path.dirname(os.path.abspath(DB_PATH)), "backups")

# Hard cap on any JSON request body read into memory, applied before auth. Attachments arrive
# base64-inside-JSON, so this sits above ATTACH_MAX_BYTES * 4/3 with headroom.
MAX_BODY_BYTES = int(os.environ.get("KENOSIS_MAX_BODY_MB", "32")) * 1024 * 1024

# Trust CF-Connecting-IP / X-Forwarded-For for the login throttle key. Correct behind the
# Cloudflare tunnel; set KENOSIS_TRUST_PROXY=0 if clients can reach this port directly,
# because those headers are spoofable by a direct client.
TRUST_PROXY = os.environ.get("KENOSIS_TRUST_PROXY", "1") not in ("0", "false", "no")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("oracle")

# Sent on every HTML document. Inline <style> stays 'unsafe-inline' (style attributes everywhere),
# but inline <script> blocks are allowed by SHA-256 hash instead of 'unsafe-inline' — injected
# script text that isn't byte-identical to our own blocks won't execute. Pages are static
# constants, so hashes are computed once per page and cached.
CSP_TMPL = ("default-src 'self'; base-uri 'self'; object-src 'none'; frame-ancestors 'none'; "
            "form-action 'self'; img-src 'self' data:; font-src 'self' https://fonts.gstatic.com; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "script-src 'self' %s; connect-src 'self'")


@_functools.lru_cache(maxsize=64)
def csp_for(html):
    """CSP with script hashes. MUST only ever be called on static templates/constants — never on
    HTML containing user data, or an injected <script> would be hashed and thereby allowed.
    Non-executable script elements (e.g. type="application/json" data islands) are skipped, so a
    template's CSP stays valid for rendered pages whose data island varies."""
    hashes = []
    for attrs, body in re.findall(r"<script([^>]*)>(.*?)</script>", html, re.S):
        mtype = re.search(r'type\s*=\s*["\']([^"\']+)', attrs)
        if mtype and mtype.group(1).strip().lower() not in ("text/javascript", "module", "application/javascript"):
            continue   # data island, not executable
        if body.strip():
            hashes.append("'sha256-%s'" % base64.b64encode(hashlib.sha256(body.encode("utf-8")).digest()).decode("ascii"))
    return CSP_TMPL % (" ".join(hashes) or "'none'")

# Attachments (uploaded files folded into a user turn) and the web-fetch tool.
DEFAULT_CONTEXT = 8192               # fallback context window when /v1/models doesn't report one
ATTACH_MAX_BYTES = 12 * 1024 * 1024  # 12 MB raw upload cap
ATTACH_MAX_CHARS = 500000            # ~125k tokens of extracted text per file (truncated beyond)
TEXT_EXTS = (".txt", ".md", ".markdown", ".text", ".csv", ".tsv", ".json", ".log", ".rst", ".yaml", ".yml")
FETCH_MAX_BYTES = 2 * 1024 * 1024    # cap on a fetched page
FETCH_MAX_CHARS = 60000              # ~15k tokens of extracted page text returned to the model
FETCH_TIMEOUT = 12
TOOL_MAX_ITERS = 4                   # max tool round-trips per assistant turn (loop guard)

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
DEFAULT_MODEL = "kenosislight"

# The model server's own sampler defaults (X_105): the groupchat-voice winner at coherence 8.78.
# Mirrored below as each spec's 'default' and seeded as default_params so a fresh DB matches live.
SERVER_DEFAULT_PARAMS = {
    "temperature": 1.05, "top_p": 0.99, "min_p": 0.03,
    "xtc_probability": 0.4, "xtc_threshold": 0.1,
    "frequency_penalty": 0.4, "presence_penalty": 0.4,
}

# Per-request sampler params. 'default' = the server's own default (what the per-param reset
# clears toward). repetition_penalty is accepted per-request by current oMLX (verified: no reload,
# applied at inference time) and matters a lot: model files that bake one in will progressively
# starve function words on long outputs. top_k is honored per-request by vLLM/llama.cpp and newer
# oMLX; older oMLX builds silently ignore it (verify with a temp=0 A/B if unsure).
PARAM_SPECS = [
    {"key": "temperature",       "label": "temperature",       "type": "float", "min": 0,  "max": 4,  "step": 0.05, "ph": "server default", "slider": True,  "default": 1.05,
     "tip": "Controls randomness. Higher values (e.g. 1.2) make replies more creative and varied; lower values (e.g. 0.7) make them more focused and predictable. 0 always picks the single most likely token."},
    {"key": "top_p",             "label": "top_p",             "type": "float", "min": 0,  "max": 1,  "step": 0.01, "ph": "server default", "slider": True,  "default": 0.99,
     "tip": "Nucleus sampling. Each step only considers the most likely tokens whose probabilities add up to this fraction. 0.9 keeps the top 90% of the probability mass; lower means more focused. Often tuned instead of temperature."},
    {"key": "min_p",             "label": "min_p",             "type": "float", "min": 0,  "max": 1,  "step": 0.01, "ph": "server default", "slider": True,  "default": 0.03,
     "tip": "Minimum probability, relative to the top token. Discards any token less likely than this fraction of the most likely one. Higher values (e.g. 0.1) prune unlikely tokens harder, keeping output coherent even at high temperature."},
    {"key": "top_k",             "label": "top_k",             "type": "int",   "min": 0,  "max": 200, "step": 1,   "ph": "server default", "slider": True,  "default": 0,
     "tip": "Each step only considers the K most likely tokens. 0 disables the cap. A blunter cousin of top_p/min_p — mostly useful as a hard safety rail at very high temperatures. Note: some backends (older oMLX) ignore this per-request."},
    {"key": "xtc_probability",   "label": "xtc_probability",   "type": "float", "min": 0,  "max": 1,  "step": 0.01, "ph": "server default", "slider": True,  "default": 0.4,
     "tip": "Exclude Top Choices: the chance, per token, of applying XTC. XTC removes the most-probable tokens (those above the threshold), always leaving at least one, to reduce clichés and boost creativity. 0 disables it."},
    {"key": "xtc_threshold",     "label": "xtc_threshold",     "type": "float", "min": 0,  "max": 1,  "step": 0.01, "ph": "server default", "slider": True,  "default": 0.1,
     "tip": "Exclude Top Choices threshold. Only tokens more probable than this are eligible to be dropped by XTC. Lower thresholds let XTC act on more tokens; a typical value is around 0.1."},
    {"key": "frequency_penalty", "label": "frequency_penalty", "type": "float", "min": -2, "max": 2,  "step": 0.05, "ph": "server default", "slider": True,  "default": 0.4,
     "tip": "Penalizes tokens by how many times they have already appeared, discouraging the model from repeating the same words. Positive values reduce repetition; negative values encourage it. Range -2 to 2."},
    {"key": "presence_penalty",  "label": "presence_penalty",  "type": "float", "min": -2, "max": 2,  "step": 0.05, "ph": "server default", "slider": True,  "default": 0.4,
     "tip": "Penalizes tokens that have appeared at all (regardless of how often), nudging the model toward new words and topics. Positive values increase novelty; negative values keep it on-topic. Range -2 to 2."},
    {"key": "repetition_penalty", "label": "repetition_penalty", "type": "float", "min": 0.8, "max": 2, "step": 0.01, "ph": "server default", "slider": True, "default": 1.0,
     "tip": "Multiplicative penalty on every token that already appeared, applied over the whole output. Values above 1 suppress loops — but on long generations they also starve common words (articles, prepositions), degrading grammar toward the end. 1.0 disables it, overriding any penalty baked into the model file. Prefer frequency_penalty for gentler repetition control."},
    {"key": "max_tokens",        "label": "max_tokens",        "type": "int",   "min": 1,             "step": 1,    "ph": str(MAX_TOKENS),  "slider": False, "default": MAX_TOKENS,
     "tip": "The maximum number of tokens to generate in a single reply. Generation stops here even if the model is not finished. This does not limit the length of the prompt or conversation."},
]
PARAM_KEYS = [p["key"] for p in PARAM_SPECS]

_init_lock = threading.Lock()
_local = threading.local()


# ---------------------------------------------------------------- small utils
def _now():
    # timezone-aware (absolute instant); the browser renders it in each viewer's local zone
    return datetime.now().astimezone().isoformat(timespec="seconds")


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
        # Writes open BEGIN IMMEDIATE transactions: a deferred read->write upgrade returns
        # "database is locked" instantly (busy_timeout can't retry it); immediate writers queue.
        c.isolation_level = "IMMEDIATE"
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=NORMAL")   # the standard WAL pairing; durable enough, much faster
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
                allowed_models TEXT, disabled INTEGER NOT NULL DEFAULT 0, persona TEXT,
                session_version INTEGER NOT NULL DEFAULT 0, created TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE IF NOT EXISTS folders(
                id TEXT PRIMARY KEY, owner_id INTEGER NOT NULL, name TEXT NOT NULL,
                position INTEGER NOT NULL DEFAULT 0, created TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS conversations(
                id TEXT PRIMARY KEY, owner_id INTEGER NOT NULL, folder_id TEXT, title TEXT,
                system TEXT, model TEXT, endpoint_id TEXT, params TEXT, character_id TEXT,
                active_leaf_id TEXT, tools INTEGER NOT NULL DEFAULT 0, think INTEGER, mode TEXT, created TEXT NOT NULL, updated TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS messages(
                id TEXT PRIMARY KEY, convo_id TEXT NOT NULL, parent_id TEXT, position INTEGER NOT NULL,
                role TEXT NOT NULL, content TEXT, reasoning TEXT, model TEXT, meta TEXT, ts TEXT, edited TEXT,
                rating INTEGER, attachments TEXT, tool TEXT);
            CREATE TABLE IF NOT EXISTS characters(
                id TEXT PRIMARY KEY, owner_id INTEGER, scope TEXT NOT NULL DEFAULT 'private',
                name TEXT, avatar TEXT, model TEXT, params TEXT, system TEXT, created TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS presets(
                id TEXT PRIMARY KEY, owner_id INTEGER, scope TEXT NOT NULL DEFAULT 'private',
                model TEXT, name TEXT, params TEXT, created TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS invites(
                token TEXT PRIMARY KEY, created_by INTEGER, role TEXT NOT NULL DEFAULT 'user',
                allowed_models TEXT, max_uses INTEGER, uses INTEGER NOT NULL DEFAULT 0,
                expires INTEGER, note TEXT, created TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS push_subs(
                endpoint TEXT PRIMARY KEY, user_id INTEGER NOT NULL, sub TEXT NOT NULL, created TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS shares(
                token TEXT PRIMARY KEY, convo_id TEXT NOT NULL, owner_id INTEGER NOT NULL,
                title TEXT, data TEXT NOT NULL, views INTEGER NOT NULL DEFAULT 0,
                created TEXT NOT NULL, updated TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS idx_shares_convo ON shares(convo_id);
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
        if "attachments" not in mcols:
            c.execute("ALTER TABLE messages ADD COLUMN attachments TEXT")
        if "tool" not in mcols:
            c.execute("ALTER TABLE messages ADD COLUMN tool TEXT")
        ccols = [r["name"] for r in c.execute("PRAGMA table_info(conversations)")]
        if "active_leaf_id" not in ccols:
            c.execute("ALTER TABLE conversations ADD COLUMN active_leaf_id TEXT")
        if "tools" not in ccols:
            c.execute("ALTER TABLE conversations ADD COLUMN tools INTEGER NOT NULL DEFAULT 0")
        if "think" not in ccols:   # NULL = model default; 1 = thinking on; 0 = thinking off
            c.execute("ALTER TABLE conversations ADD COLUMN think INTEGER")
        if "mode" not in ccols:    # NULL/'chat' = a normal conversation; 'compose' = one continuable text
            c.execute("ALTER TABLE conversations ADD COLUMN mode TEXT")
        ucols = [r["name"] for r in c.execute("PRAGMA table_info(users)")]
        if "persona" not in ucols:
            c.execute("ALTER TABLE users ADD COLUMN persona TEXT")
        if "totp_secret" not in ucols:   # base32 TOTP secret; NULL = 2FA off
            c.execute("ALTER TABLE users ADD COLUMN totp_secret TEXT")
        if "session_version" not in ucols:   # bumped on password change to revoke old session cookies
            c.execute("ALTER TABLE users ADD COLUMN session_version INTEGER NOT NULL DEFAULT 0")
        if "ctx_summary" not in ccols:      # rolling summary of turns dropped by the sliding window
            c.execute("ALTER TABLE conversations ADD COLUMN ctx_summary TEXT")
        if "ctx_summary_upto" not in ccols:  # id of the last message that summary covers
            c.execute("ALTER TABLE conversations ADD COLUMN ctx_summary_upto TEXT")
        c.commit()
        init_fts()
        seed_settings()
        if not get_setting("tree_migrated"):
            migrate_tree()
            set_setting("tree_migrated", "1")


FTS_OK = False   # set by init_fts(); when False, search falls back to LIKE and knowledge retrieval is off


def init_fts():
    """FTS5 index over message bodies (instant ranked search) and character knowledge chunks
    (BM25 retrieval). Kept in sync by triggers. SQLite builds without FTS5 degrade gracefully."""
    global FTS_OK
    c = db()
    try:
        c.executescript(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                content, convo_id UNINDEXED, msg_id UNINDEXED, tokenize='unicode61');
            CREATE TRIGGER IF NOT EXISTS msg_fts_ins AFTER INSERT ON messages BEGIN
                INSERT INTO messages_fts(content, convo_id, msg_id)
                VALUES (COALESCE(new.content,''), new.convo_id, new.id);
            END;
            CREATE TRIGGER IF NOT EXISTS msg_fts_del AFTER DELETE ON messages BEGIN
                DELETE FROM messages_fts WHERE msg_id = old.id;
            END;
            CREATE TRIGGER IF NOT EXISTS msg_fts_upd AFTER UPDATE OF content ON messages BEGIN
                DELETE FROM messages_fts WHERE msg_id = old.id;
                INSERT INTO messages_fts(content, convo_id, msg_id)
                VALUES (COALESCE(new.content,''), new.convo_id, new.id);
            END;
            CREATE TABLE IF NOT EXISTS knowledge(
                id TEXT PRIMARY KEY, character_id TEXT NOT NULL, name TEXT,
                chars INTEGER NOT NULL DEFAULT 0, created TEXT NOT NULL);
            CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
                chunk, character_id UNINDEXED, doc_id UNINDEXED, name UNINDEXED, tokenize='unicode61');
            CREATE TABLE IF NOT EXISTS knowledge_vec(
                doc_id TEXT NOT NULL, idx INTEGER NOT NULL, character_id TEXT NOT NULL,
                chunk TEXT, vec TEXT, PRIMARY KEY(doc_id, idx));
            """
        )
        if not get_setting("fts_built"):
            with c:
                c.execute("DELETE FROM messages_fts")
                c.execute("INSERT INTO messages_fts(content, convo_id, msg_id) "
                          "SELECT COALESCE(content,''), convo_id, id FROM messages")
            set_setting("fts_built", "1")
        FTS_OK = True
    except sqlite3.OperationalError as e:
        log.warning("FTS5 unavailable (%s) — search uses LIKE scans; character knowledge disabled", e)
        FTS_OK = False


def _fts_match_query(q, all_terms=True):
    """User text -> FTS5 MATCH string. Quoted terms (last as prefix) ANDed for search;
    OR-joined for knowledge recall."""
    terms = re.findall(r"\w+", q or "", re.UNICODE)[:12]
    if all_terms:
        if not terms:
            return None
        quoted = ['"%s"' % t for t in terms[:-1]] + ['"%s"*' % terms[-1]]
        return " ".join(quoted)               # implicit AND
    terms = [t for t in terms if len(t) > 2]
    return " OR ".join('"%s"' % t for t in terms) or None


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
        "default_params": dict(SERVER_DEFAULT_PARAMS), "user_models": [DEFAULT_MODEL], "thinking_models": [],
        "search_url": "", "utility_model": "", "vision_models": [], "embed_model": "",
        # {model: params} layered between default_params and the conversation's own params; a model
        # with no entry behaves exactly as it did before this existed. Seeded empty because the
        # values are per-checkpoint and come from the sweeps, not from source.
        "model_defaults": {},
    }
    for k, v in defaults.items():
        if get_setting(k) is None:
            set_setting(k, v)


# Endpoint API keys are never sent to the browser; saved settings echo this sentinel instead,
# and the save/test handlers swap the stored key back in when they see it.
KEY_SENTINEL = "__stored__"


def redacted_endpoints():
    out = []
    for ep in get_setting("endpoints", []):
        ep = dict(ep)
        if ep.get("key"):
            ep["key"] = KEY_SENTINEL
        out.append(ep)
    return out


def restore_endpoint_keys(eps):
    """Replace KEY_SENTINEL placeholders in a submitted endpoint list with the stored keys."""
    stored = {e.get("id"): e for e in get_setting("endpoints", [])}
    out = []
    for ep in (eps or []):
        ep = dict(ep)
        if ep.get("key") == KEY_SENTINEL:
            ep["key"] = (stored.get(ep.get("id")) or {}).get("key", "")
        out.append(ep)
    return out


def admin_settings():
    s = {k: get_setting(k) for k in
         ("endpoints", "active_endpoint", "default_model", "default_system", "default_params",
          "model_defaults", "user_models", "thinking_models", "search_url", "utility_model",
          "vision_models", "embed_model")}
    s["endpoints"] = redacted_endpoints()
    return s


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


_SESSION_SECRET = None


def secret_bytes():
    """The cookie-signing secret. Preference: KENOSIS_SESSION_SECRET (never persisted) > a legacy
    copy in the settings table (pre-hardening installs) > a fresh ephemeral value. The secret is
    no longer written to the database, so DB copies and backups can't be used to forge sessions."""
    global _SESSION_SECRET
    if _SESSION_SECRET is None:
        env = os.environ.get("KENOSIS_SESSION_SECRET")
        legacy = get_setting("session_secret")
        if env:
            _SESSION_SECRET = env
            if legacy is not None:
                with db():
                    db().execute("DELETE FROM settings WHERE key='session_secret'")
                log.info("scrubbed legacy session secret from the database (env var is authoritative)")
        elif legacy:
            _SESSION_SECRET = legacy
            log.warning("session secret lives in the database (legacy install); set KENOSIS_SESSION_SECRET "
                        "and restart to keep it out of DB backups")
        else:
            _SESSION_SECRET = b64(os.urandom(32))
            log.warning("KENOSIS_SESSION_SECRET is not set — using an ephemeral secret; "
                        "sessions will not survive a restart")
    return _SESSION_SECRET.encode("utf-8")


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


# ---- TOTP two-factor (stdlib only; RFC 6238, SHA-1, 6 digits, 30s step)
def totp_gen_secret():
    return base64.b32encode(os.urandom(20)).decode("ascii").rstrip("=")


def totp_code(secret_b32, t=None, step=30, digits=6):
    key = base64.b32decode(secret_b32.upper() + "=" * (-len(secret_b32) % 8))
    counter = int((time.time() if t is None else t) // step)
    h = hmac.new(key, counter.to_bytes(8, "big"), hashlib.sha1).digest()
    o = h[-1] & 0xF
    return str((int.from_bytes(h[o:o + 4], "big") & 0x7FFFFFFF) % (10 ** digits)).zfill(digits)


def totp_verify(secret_b32, code, window=1):
    code = re.sub(r"\D", "", code or "")
    if len(code) != 6 or not secret_b32:
        return False
    now = time.time()
    try:
        return any(hmac.compare_digest(totp_code(secret_b32, now + i * 30), code)
                   for i in range(-window, window + 1))
    except Exception:
        return False


def user_totp_secret(u):
    return (u["totp_secret"] if "totp_secret" in u.keys() else None) or ""


def user_session_version(u):
    return (u["session_version"] if "session_version" in u.keys() else 0) or 0


def sign_session(username, version=0):
    exp = int(time.time()) + SESSION_DAYS * 86400
    payload = b64(json.dumps({"u": username, "exp": exp, "v": version}).encode("utf-8"))
    sig = b64(hmac.new(secret_bytes(), payload.encode("ascii"), hashlib.sha256).digest())
    return payload + "." + sig


def parse_session(token):
    """Valid token -> (username, session_version); anything else -> None. Tokens minted before
    versioning carry an implicit version 0, which matches the column default."""
    try:
        payload, sig = token.split(".")
        good = b64(hmac.new(secret_bytes(), payload.encode("ascii"), hashlib.sha256).digest())
        if not hmac.compare_digest(sig, good):
            return None
        data = json.loads(ub64(payload))
        if data.get("exp", 0) < time.time():
            return None
        return (data.get("u"), data.get("v", 0))
    except Exception:
        return None


# ---- login brute-force throttle (per client IP, in-memory; resets on restart)
_login_fails = {}
_login_lock = threading.Lock()


def login_retry_after(key):
    """Seconds the caller must wait, or 0 if not currently throttled."""
    now = time.time()
    with _login_lock:
        arr = [t for t in _login_fails.get(key, []) if now - t < LOGIN_WINDOW]
        if arr:
            _login_fails[key] = arr
        else:
            _login_fails.pop(key, None)
        if len(arr) >= LOGIN_MAX_FAILS:
            return int(LOGIN_WINDOW - (now - arr[0])) + 1
    return 0


def login_record_fail(key):
    now = time.time()
    with _login_lock:
        arr = [t for t in _login_fails.get(key, []) if now - t < LOGIN_WINDOW]
        arr.append(now)
        _login_fails[key] = arr
        # opportunistic prune so the table can't grow without bound
        if len(_login_fails) > 4096:
            for k in [k for k, v in _login_fails.items() if not v or now - v[-1] >= LOGIN_WINDOW]:
                _login_fails.pop(k, None)


def login_clear(key):
    with _login_lock:
        _login_fails.pop(key, None)


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
            "persona": (u["persona"] or "") if "persona" in u.keys() else "",
            "totp": bool(user_totp_secret(u)),
            "disabled": bool(u["disabled"]), "created": u["created"]}


def allowed_models_for(u):
    if u["role"] == "admin":
        return None
    per = json.loads(u["allowed_models"]) if u["allowed_models"] else []
    return per if per else get_setting("user_models", [])


def model_allowed(u, model):
    allow = allowed_models_for(u)
    return allow is None or model in allow


# ---------------------------------------------------------------- invite links
def gen_token():
    return base64.urlsafe_b64encode(os.urandom(18)).decode("ascii").rstrip("=")


def invite_by_token(tok):
    return db().execute("SELECT * FROM invites WHERE token=?", (tok,)).fetchone()


def invite_valid(inv):
    if inv is None:
        return False
    if inv["expires"] is not None and time.time() >= inv["expires"]:
        return False
    if inv["max_uses"] is not None and inv["uses"] >= inv["max_uses"]:
        return False
    return True


def invite_status(inv):
    if inv["expires"] is not None and time.time() >= inv["expires"]:
        return "expired"
    if inv["max_uses"] is not None and inv["uses"] >= inv["max_uses"]:
        return "used up"
    return "active"


def invite_error(inv):
    if inv is None:
        return "This invite link is not valid."
    if inv["expires"] is not None and time.time() >= inv["expires"]:
        return "This invite link has expired."
    if inv["max_uses"] is not None and inv["uses"] >= inv["max_uses"]:
        return "This invite link has already been used up."
    return "This invite link is not valid."


def invite_public(inv):
    return {"token": inv["token"], "role": inv["role"],
            "allowed_models": json.loads(inv["allowed_models"]) if inv["allowed_models"] else [],
            "max_uses": inv["max_uses"], "uses": inv["uses"], "expires": inv["expires"],
            "note": inv["note"] or "", "status": invite_status(inv), "created": inv["created"]}


def list_invites():
    return [invite_public(r) for r in db().execute("SELECT * FROM invites ORDER BY created DESC").fetchall()]


def delete_user_cascade(uid):
    c = db()
    with c:
        c.execute("DELETE FROM shares WHERE owner_id=?", (uid,))
        c.execute("DELETE FROM messages WHERE convo_id IN (SELECT id FROM conversations WHERE owner_id=?)", (uid,))
        c.execute("DELETE FROM conversations WHERE owner_id=?", (uid,))
        c.execute("DELETE FROM folders WHERE owner_id=?", (uid,))
        if FTS_OK:
            c.execute("DELETE FROM knowledge_fts WHERE doc_id IN (SELECT id FROM knowledge WHERE character_id IN"
                      " (SELECT id FROM characters WHERE owner_id=? AND scope='private'))", (uid,))
            c.execute("DELETE FROM knowledge WHERE character_id IN"
                      " (SELECT id FROM characters WHERE owner_id=? AND scope='private')", (uid,))
        c.execute("DELETE FROM characters WHERE owner_id=? AND scope='private'", (uid,))
        c.execute("DELETE FROM users WHERE id=?", (uid,))


# ---------------------------------------------------------------- characters
def knowledge_list(char_id):
    try:
        rows = db().execute("SELECT id,name,chars FROM knowledge WHERE character_id=? ORDER BY created", (char_id,)).fetchall()
    except sqlite3.OperationalError:   # FTS5-less build never created the table
        return []
    return [{"id": r["id"], "name": r["name"], "chars": r["chars"]} for r in rows]


def visible_characters(u):
    rows = db().execute(
        "SELECT * FROM characters WHERE scope='site' OR owner_id=? ORDER BY scope DESC, name COLLATE NOCASE", (u["id"],)
    ).fetchall()
    return [{"id": r["id"], "name": r["name"], "avatar": r["avatar"] or "", "model": r["model"],
             "scope": r["scope"], "owner_id": r["owner_id"], "system": r["system"] or "",
             "params": json.loads(r["params"]) if r["params"] else None,
             "knowledge": knowledge_list(r["id"]),
             "editable": (r["owner_id"] == u["id"]) or (u["role"] == "admin")} for r in rows]


def character_by_id(cid):
    return db().execute("SELECT * FROM characters WHERE id=?", (cid,)).fetchone()


def _preset_models(raw):
    """The model column holds a JSON list of model ids; '' / [] means it applies to all models.
    Also tolerates a legacy bare-string single model from earlier builds."""
    if not raw:
        return []
    try:
        v = json.loads(raw)
        if isinstance(v, list):
            return [str(m) for m in v if m]
    except (ValueError, TypeError):
        pass
    return [raw]


def visible_presets(u):
    """Sampler-parameter presets the user can see: every site-wide one plus their own."""
    rows = db().execute(
        "SELECT * FROM presets WHERE scope='site' OR owner_id=? ORDER BY scope DESC, name COLLATE NOCASE", (u["id"],)
    ).fetchall()
    return [{"id": r["id"], "name": r["name"], "models": _preset_models(r["model"]),
             "scope": r["scope"], "owner_id": r["owner_id"],
             "params": json.loads(r["params"]) if r["params"] else {},
             "editable": (r["owner_id"] == u["id"]) or (u["role"] == "admin")} for r in rows]


def preset_by_id(pid):
    return db().execute("SELECT * FROM presets WHERE id=?", (pid,)).fetchone()


def _norm_params(d):
    return {k: str(d[k]) for k in (d or {}) if k in PARAM_KEYS and d[k] not in (None, "")}


def matching_preset_name(u, model, conv_params):
    """Name of the visible preset whose sampler values equal the conversation's overrides for this
    model, or None. Mirrors the client-side preset match so a recorded message can name its preset."""
    target = _norm_params(conv_params)
    if not target:
        return None
    for p in visible_presets(u):
        models = p.get("models") or []
        if models and model not in models:
            continue
        if _norm_params(p["params"]) == target:
            return p["name"]
    return None


# ---------------------------------------------------------------- folders / convos
def list_folders(u):
    rows = db().execute("SELECT * FROM folders WHERE owner_id=? ORDER BY position, name COLLATE NOCASE", (u["id"],)).fetchall()
    return [{"id": r["id"], "name": r["name"], "position": r["position"]} for r in rows]


def list_convos(u):
    rows = db().execute(
        "SELECT c.id,c.title,c.updated,c.created,c.model,c.character_id,c.folder_id,c.mode,"
        "(SELECT COUNT(*) FROM messages m WHERE m.convo_id=c.id AND m.role IN('user','assistant')) turns "
        "FROM conversations c WHERE c.owner_id=? ORDER BY c.updated DESC", (u["id"],)).fetchall()
    return [dict(r) for r in rows]


def _snippet(content, q, span=140):
    """A short excerpt of `content` centred on the first case-insensitive match of `q`."""
    content = " ".join((content or "").split())   # collapse whitespace for a tidy one-liner
    i = content.lower().find(q.lower())
    if i < 0:
        return content[:span]
    start = max(0, i - 48)
    end = min(len(content), i + len(q) + span - 48)
    return ("…" if start > 0 else "") + content[start:end] + ("…" if end < len(content) else "")


def search_convo_content(u, q, limit=60):
    """Conversations of `u` with a message body matching `q`, each with one snippet. Title-only
    matching is handled client-side; this is the full-text half. Uses the FTS5 index (instant,
    ranked) with the old LIKE scan as fallback."""
    q = (q or "").strip()
    if len(q) < 2:
        return []
    rows = None
    if FTS_OK:
        match = _fts_match_query(q)
        if match:
            try:
                rows = db().execute(
                    "SELECT c.id,c.title,c.updated,c.created,c.model,c.character_id,c.folder_id,m.content "
                    "FROM messages_fts f JOIN messages m ON m.id=f.msg_id "
                    "JOIN conversations c ON c.id=f.convo_id "
                    "WHERE messages_fts MATCH ? AND c.owner_id=? AND m.role IN('user','assistant') "
                    "ORDER BY c.updated DESC LIMIT 400", (match, u["id"])).fetchall()
            except sqlite3.OperationalError:
                rows = None   # e.g. query text FTS can't parse — fall through to LIKE
    if rows is None:
        like = "%" + q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        # LIMIT bounds the scan: enough rows to fill `limit` distinct conversations on any realistic
        # data, without ever walking the whole message table for a common word.
        rows = db().execute(
            "SELECT c.id,c.title,c.updated,c.created,c.model,c.character_id,c.folder_id,m.content "
            "FROM messages m JOIN conversations c ON c.id=m.convo_id "
            "WHERE c.owner_id=? AND m.role IN('user','assistant') AND m.content LIKE ? ESCAPE '\\' "
            "ORDER BY c.updated DESC LIMIT 400", (u["id"], like)).fetchall()
    out, seen = [], set()
    for r in rows:
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        d = {k: r[k] for k in ("id", "title", "updated", "created", "model", "character_id", "folder_id")}
        d["snippet"] = _snippet(r["content"], q)
        out.append(d)
        if len(out) >= limit:
            break
    return out


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
    if r["attachments"]:
        m["attachments"] = json.loads(r["attachments"])
    if r["tool"]:
        m["tool"] = json.loads(r["tool"])
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


def chain_content(cid, node_id, by=None):
    if node_id is None:
        return []
    if by is None:   # callers with a fresh _tree() in hand pass it to skip the re-query
        _, by, _ = _tree(cid)
    seq, cur = [], node_id
    while cur is not None and cur in by:
        seq.append(by[cur])
        cur = by[cur]["parent_id"]
    seq.reverse()
    return [{"id": r["id"], "role": r["role"], "content": r["content"] or "",
             "attachments": json.loads(r["attachments"]) if r["attachments"] else None,
             "tool": json.loads(r["tool"]) if r["tool"] else None} for r in seq]


def get_convo(cid, u=None):
    r = db().execute("SELECT * FROM conversations WHERE id=?", (cid,)).fetchone()
    if r is None or (u is not None and r["owner_id"] != u["id"]):
        return None
    return {"id": r["id"], "owner_id": r["owner_id"], "folder_id": r["folder_id"],
            "title": r["title"] or "", "system": r["system"] or "", "model": r["model"],
            "endpoint_id": r["endpoint_id"], "params": json.loads(r["params"]) if r["params"] else {},
            "character_id": r["character_id"], "active_leaf_id": r["active_leaf_id"],
            "tools": bool(r["tools"]), "think": r["think"],
            "mode": (r["mode"] if "mode" in r.keys() else None) or "chat",
            "ctx_summary": r["ctx_summary"] if "ctx_summary" in r.keys() else None,
            "ctx_summary_upto": r["ctx_summary_upto"] if "ctx_summary_upto" in r.keys() else None,
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


def insert_message(cid, parent, role, content, reasoning=None, model=None, meta=None,
                   attachments=None, tool=None):
    mid = _mid()
    # position is computed inside the INSERT so two threads writing to the same conversation
    # can't read the same MAX() and produce duplicate positions
    db().execute(
        "INSERT INTO messages(id,convo_id,parent_id,position,role,content,reasoning,model,meta,ts,attachments,tool)"
        " SELECT ?,?,?,COALESCE(MAX(position),-1)+1,?,?,?,?,?,?,?,? FROM messages WHERE convo_id=?",
        (mid, cid, parent, role, content, reasoning or None, model,
         json.dumps(meta) if meta else None, _now(),
         json.dumps(attachments) if attachments else None,
         json.dumps(tool) if tool else None, cid))
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
    c = db()
    with c:
        # one recursive DELETE inside the transaction, so messages inserted concurrently can't be
        # orphaned between a separate tree read and the delete
        cur = c.execute(
            "WITH RECURSIVE sub(id) AS ("
            " SELECT id FROM messages WHERE id=? AND convo_id=?"
            " UNION ALL"
            " SELECT m.id FROM messages m JOIN sub s ON m.parent_id=s.id) "
            "DELETE FROM messages WHERE convo_id=? AND id IN sub", (mid, cid, cid))
        if cur.rowcount == 0:
            return
        row = c.execute("SELECT active_leaf_id FROM conversations WHERE id=?", (cid,)).fetchone()
        leaf = row["active_leaf_id"] if row else None
        if leaf and not c.execute("SELECT 1 FROM messages WHERE id=? AND convo_id=?", (leaf, cid)).fetchone():
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


# ---------------------------------------------------------------- backups
def backup_db():
    """Write a consistent, defragmented copy of the DB via VACUUM INTO, then prune old copies."""
    os.makedirs(BACKUP_DIR, exist_ok=True)
    dest = os.path.join(BACKUP_DIR, "chat-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".db")
    con = sqlite3.connect(DB_PATH, timeout=30)
    try:
        con.execute("VACUUM INTO ?", (dest,))
    finally:
        con.close()
    # Verify the copy actually opens and passes an integrity check — a backup that can't restore
    # is worse than none, because it looks like insurance. A bad copy is deleted and logged loudly.
    try:
        chk = sqlite3.connect(dest)
        try:
            ok = chk.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            users = chk.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        finally:
            chk.close()
        if not ok or users < 1:
            raise ValueError("integrity check failed" if not ok else "no users in backup")
    except Exception as e:
        log.error("backup verification FAILED for %s: %s — deleting the bad copy", dest, e)
        try:
            os.remove(dest)
        except OSError:
            pass
        raise
    if BACKUP_KEEP > 0:
        old = sorted(glob.glob(os.path.join(BACKUP_DIR, "chat-*.db")))[:-BACKUP_KEEP]
        for f in old:
            try:
                os.remove(f)
            except OSError:
                pass
    return dest


def _latest_backup_age():
    files = glob.glob(os.path.join(BACKUP_DIR, "chat-*.db"))
    if not files:
        return None
    return time.time() - max(os.path.getmtime(f) for f in files)


def _backup_loop():
    interval = max(0.1, BACKUP_HOURS) * 3600
    while True:
        try:
            con = sqlite3.connect(DB_PATH, timeout=30)
            try:
                con.execute("PRAGMA wal_checkpoint(TRUNCATE)")   # keep the WAL from growing unbounded
            finally:
                con.close()
        except Exception:
            log.exception("wal checkpoint failed")
        # back up only when the newest copy is older than the interval, so container restarts
        # don't each burn one of the BACKUP_KEEP retention slots
        try:
            age = _latest_backup_age()
        except OSError:
            age = None
        if age is None or age >= interval:
            try:
                dest = backup_db()
                log.info("backup written: %s", dest)
            except Exception:
                log.exception("backup failed")
        time.sleep(min(interval, 3600))


def start_backups():
    if BACKUP_HOURS <= 0:
        log.info("automated backups disabled (KENOSIS_BACKUP_HOURS=0)")
        return
    threading.Thread(target=_backup_loop, daemon=True).start()
    log.info("automated backups: every %gh -> %s (keep %d)", BACKUP_HOURS, BACKUP_DIR, BACKUP_KEEP)


# ---------------------------------------------------------------- model calls
def est_tokens(text):
    return max(1, round(len(text or "") / 4))


def _msg_tokens(m):
    t = est_tokens(m.get("content"))
    for a in (m.get("attachments") or []):
        t += est_tokens(a.get("text"))
        if a.get("image"):
            t += 1024   # rough per-image cost; varies by model/resolution
    return t


# Our token estimate is a crude chars/4. After each reply the server reports the *real* prompt token
# count, so we learn a per-model correction factor (real / estimated) and apply it to future context
# trim / cap decisions — they then track the model's actual tokenizer instead of a fixed guess.
# Persisted to the settings table so calibration survives restarts instead of resetting to 1.0.
_TOK_CAL = None   # model -> factor; lazy-loaded under _TOK_CAL_LOCK
_TOK_CAL_LOCK = threading.Lock()


def _tok_cal():
    global _TOK_CAL
    if _TOK_CAL is None:
        try:
            _TOK_CAL = {str(k): float(v) for k, v in (get_setting("tok_factors") or {}).items()}
        except Exception:
            _TOK_CAL = {}
    return _TOK_CAL


def tok_factor(model):
    with _TOK_CAL_LOCK:
        return _tok_cal().get(model, 1.0)


def update_tok_factor(model, real_tokens, est):
    if not model or not real_tokens or not est:
        return
    obs = max(0.4, min(4.0, real_tokens / est))   # clamp wild outliers
    with _TOK_CAL_LOCK:
        cal = _tok_cal()
        cur = cal.get(model)
        cal[model] = obs if cur is None else (0.7 * cur + 0.3 * obs)   # EMA toward the latest reading
        snapshot = dict(cal)
    try:
        set_setting("tok_factors", snapshot)   # one tiny upsert per reply; WAL makes this cheap
    except Exception:
        log.warning("could not persist token calibration", exc_info=True)


def _attach_block(attachments):
    # Fence each file so the model can cleanly tell document text apart from the user's instruction.
    parts = []
    for a in attachments or []:
        name = str(a.get("name", "file")).replace('"', "'")
        parts.append('<file name="%s">\n%s\n</file>' % (name, a.get("text", "")))
    return "\n\n".join(parts)


def _roll_dice(spec):
    spec = spec.strip().lower().replace(" ", "")
    m = re.fullmatch(r"(\d*)d(\d+)", spec)
    if m:
        n = min(int(m.group(1) or "1"), 100)
        sides = max(1, min(int(m.group(2)), 1000))
        return sum(random.randint(1, sides) for _ in range(max(0, n)))
    if spec.isdigit() and int(spec) > 0:
        return random.randint(1, int(spec))
    return 0


def apply_macros(text, ctx):
    """SillyTavern-style {{macros}} for the system prompt. ctx provides user/char/model; the date
    is resolved at send time. Deliberately no time-of-day macros: they'd change the system prompt
    (position 0 of the prompt) every request and defeat server prefix caching; everything here is
    day-granular or per-conversation-stable. Unknown macros are left untouched."""
    if not text or "{{" not in text:
        return text
    now = datetime.now().astimezone()
    simple = {
        "user": ctx.get("user") or "User",
        "char": ctx.get("char") or "Assistant",
        "character": ctx.get("char") or "Assistant",
        "model": ctx.get("model") or "",
        "newline": "\n",
        "date": now.strftime("%Y-%m-%d"),
        "isodate": now.strftime("%Y-%m-%d"),
        "weekday": now.strftime("%A"),
        "day": now.strftime("%d"),
        "month": now.strftime("%B"),
        "year": now.strftime("%Y"),
    }

    def repl(m):
        raw = m.group(1).strip()
        key = raw.lower()
        if key in simple:
            return simple[key]
        if raw.startswith("//"):          # {{// comment}}
            return ""
        head, sep, rest = raw.partition(":")
        if sep:
            head = head.strip().lower()
            if head in ("random", "pick"):
                opts = [o.strip() for o in rest.split(",") if o.strip()]
                return random.choice(opts) if opts else ""
            if head == "roll":
                return str(_roll_dice(rest))
        return m.group(0)                 # unknown -> leave as typed

    return re.sub(r"\{\{\s*([^{}]+?)\s*\}\}", repl, text)


def build_api_messages(system, messages, vision=False):
    api = []
    if system and system.strip():
        api.append({"role": "system", "content": system})
    for m in messages:
        role = m.get("role")
        if role == "tool":
            t = m.get("tool") or {}
            api.append({"role": "tool", "tool_call_id": t.get("tool_call_id", ""),
                        "content": m.get("content", "")})
        elif role == "assistant" and (m.get("tool") or {}).get("tool_calls"):
            api.append({"role": "assistant", "content": m.get("content") or "",
                        "tool_calls": m["tool"]["tool_calls"]})
        elif role in ("user", "assistant"):
            content = m.get("content", "")
            if role == "assistant" and content and "<tool_call>" in content:
                content = strip_tool_calls(content)  # don't feed leaked tool-call text back as an example
            atts = m.get("attachments") or []
            block = _attach_block(atts)
            if block:
                content = block + ("\n\n" + content if content else "")
            images = [a.get("image") for a in atts if isinstance(a, dict) and a.get("image")]
            if images and vision and role == "user":
                # OpenAI-style multimodal parts; only for models the admin marked vision-capable
                parts = ([{"type": "text", "text": content}] if content else [])
                parts += [{"type": "image_url", "image_url": {"url": img}} for img in images]
                api.append({"role": role, "content": parts})
                continue
            if images:   # non-vision model (or assistant turn): note the images textually
                note = "\n".join("[attached image: %s]" % (a.get("name") or "image")
                                 for a in atts if isinstance(a, dict) and a.get("image"))
                content = (note + "\n\n" + content) if content else note
            api.append({"role": role, "content": content})
    return api


# Appended to the system prompt only when web tools are on. Stable text per admin config → part of
# the cacheable prefix (it only changes when the admin toggles search on/off), so it never costs a
# per-turn cache hit. Targets the exact failure modes the recovery code handles.
def tool_guide():
    g = ("# Tools\n"
         "You can call the fetch_url function to read a public web page or a plain-text/JSON URL. "
         "Use it when the user shares a link, or when a good answer needs current or online information you "
         "don't reliably know. Pass the full absolute http(s) URL. ")
    if (get_setting("search_url") or "").strip():
        g += ("You can call web_search(query) to find pages, then fetch_url the promising results. ")
    g += ("You can call calculate(expression) for exact arithmetic instead of computing it yourself. "
          "Ground your answers only in what you actually fetched — quote or summarize it — and never invent "
          "URLs, page contents, or citations. If a tool fails, say so plainly instead of guessing.")
    return g

# Default-on; admins can set KENOSIS_SITUATION=0 to stop injecting the date/no-internet note.
SITUATION_NOTE = os.environ.get("KENOSIS_SITUATION", "1") not in ("0", "false", "no", "")


def _today_str():
    now = datetime.now().astimezone()
    return now.strftime("%A, ") + now.strftime("%B ") + str(int(now.strftime("%d"))) + now.strftime(", %Y")


def compose_system(system, tools_on):
    """Assemble the final system prompt from the user's text plus situational addenda.

    Cache-aware by construction: the user's prompt and the (static) tool guidance form a stable
    prefix, and the only volatile piece — the date — is deliberately day-granular (never time-of-day),
    so vLLM prefix-caching holds across every turn within a day and misses at most once at midnight.
    """
    parts = []
    s = (system or "").strip()
    if s:
        parts.append(s)
    if tools_on:
        parts.append(tool_guide())
    if SITUATION_NOTE:
        line = "Current date: " + _today_str() + "."
        if not tools_on:
            line += (" You have no live internet access; if a question depends on current information "
                     "you can't be sure of, say so rather than guessing.")
        parts.append(line)
    return "\n\n".join(parts)


def model_for(convo, model_override=None):
    return model_override or convo.get("model") or get_setting("default_model") or DEFAULT_MODEL


def effective_params(convo, model=None):
    """Sampler params for a request, layered global -> per-model -> per-conversation.

    The middle layer is `model_defaults`, a {model: params} map, and it exists because a good
    sampler config is a property of the checkpoint, not of the site: the sweeps put centostron1's
    best all-round settings at t0.95/xtc 0.50 and centosbolt2's at t0.70, and no single global can
    be right for both. A conversation that has never touched the tune drawer carries `params: {}`,
    so before this layer existed it got the global config whatever model it was pointed at.

    Each layer MERGES over the one below rather than replacing it, which is what keeps the older,
    partially-specified presets behaving as they always have -- their unset keys still fall through
    to the same global values they fell through to before. A model entry only has to state the keys
    where that checkpoint differs, though the entries written from a full sweep state all of them so
    nothing leaks in from the global.
    """
    merged = dict(get_setting("default_params") or {})
    merged.update((get_setting("model_defaults") or {}).get(model_for(convo, model)) or {})
    merged.update(convo.get("params") or {})
    out = {k: v for k, v in merged.items() if k in PARAM_KEYS and v not in (None, "")}
    out.setdefault("max_tokens", MAX_TOKENS)
    return out


def resolve_request(convo, model_override=None):
    """The model override is applied BEFORE the params are computed, not after. Once defaults are
    per-model, resolving params against the conversation's model and then swapping the model out
    would send one checkpoint's sampler settings to a different one -- which is exactly what
    'regenerate with...' does."""
    ep = endpoint_by_id(convo.get("endpoint_id")) if convo.get("endpoint_id") else active_endpoint()
    model = model_for(convo, model_override)
    return ep, model, convo.get("system", ""), effective_params(convo, model)


def _model_post(url, headers, body, stream):
    """POST to the model server with a (connect, read) timeout split and connect-phase retries.
    The split makes a dead endpoint fail in seconds instead of holding the request for the full
    generation budget. The retries matter because a busy single-process model server (oMLX swapping
    models under memory pressure) can stall its accept queue for tens of seconds — a connect failure
    there is transient, and since nothing has been sent yet, retrying is always safe."""
    for wait in (4, 12):
        try:
            return requests.post(url, headers=headers, json=body, stream=stream, timeout=(10, REQUEST_TIMEOUT))
        except (requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError) as e:
            log.warning("connect to %s failed (%s); retrying in %ss", url, type(e).__name__, wait)
            time.sleep(wait)
    return requests.post(url, headers=headers, json=body, stream=stream, timeout=(10, REQUEST_TIMEOUT))


def _open_stream(ep, body):
    headers = {"Content-Type": "application/json"}
    if ep.get("key"):
        headers["Authorization"] = "Bearer " + ep["key"]
    return _model_post(ep["url"], headers, body, stream=True)


def stream_model(ep, model, system, messages, params, tools=None, extra=None, tool_choice="auto", vision=False):
    base = {"model": model, "messages": build_api_messages(system, messages, vision=vision), "stream": True}
    base.update(params)
    if tools:
        base["tools"] = tools
        base["tool_choice"] = tool_choice
    so = {"include_usage": True}
    # Graduated fallback on 4xx, shedding one risky field per step so a picky server never
    # silently loses usage reporting (which feeds token calibration) along with the hints:
    #   1. base + best-effort hints (continue_final_message etc.) + stream_options
    #   2. base + stream_options                      (only if hints were present)
    #   3. tools/tool_choice stripped + stream_options (only for non-auto tool_choice)
    #   4. bare base
    first = dict(base)
    if extra:
        first.update(extra)
    first["stream_options"] = so
    attempts = [first]
    if extra:
        attempts.append(dict(base, stream_options=so))
    if tools and tool_choice != "auto":
        nt = {k: v for k, v in base.items() if k not in ("tools", "tool_choice")}
        attempts.append(dict(nt, stream_options=so))
        attempts.append(nt)
    else:
        attempts.append(base)
    r = _open_stream(ep, attempts[0])
    for b in attempts[1:]:
        if r.status_code < 400:
            break
        try:
            r.close()
        except Exception:
            pass
        r = _open_stream(ep, b)
    try:
        r.raise_for_status()
    except Exception:
        r.close()   # streamed response: without this the pooled connection leaks
        raise
    tcalls = {}
    finish = None
    r.encoding = "utf-8"   # SSE bodies are UTF-8; without a charset in Content-Type, requests would decode as ISO-8859-1
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
                if choices[0].get("finish_reason"):
                    finish = choices[0]["finish_reason"]
                delta = choices[0].get("delta") or {}
                if delta.get("content"):
                    yield {"delta": delta["content"]}
                rc = delta.get("reasoning_content") or delta.get("reasoning")
                if rc:
                    yield {"reasoning": rc}
                for tc in (delta.get("tool_calls") or []):
                    idx = tc.get("index", 0)
                    slot = tcalls.setdefault(idx, {"id": None, "type": "function",
                                                   "function": {"name": "", "arguments": ""}})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["function"]["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["function"]["arguments"] += fn["arguments"]
            if obj.get("usage"):
                yield {"usage": obj["usage"]}
    if tcalls:
        ordered = [tcalls[k] for k in sorted(tcalls)]
        for i, s in enumerate(ordered):
            if not s.get("id"):
                s["id"] = "call_%d" % i
        yield {"tool_calls": ordered}
    yield {"finish": finish}


def call_model_nonstream(ep, model, system, messages, params, tools, vision=False):
    """One non-streaming completion. The oMLX server returns structured tool_calls reliably this
    way (its streaming tool parser can silently drop a call), so it's the recovery path."""
    body = {"model": model, "messages": build_api_messages(system, messages, vision=vision), "stream": False}
    body.update(params)
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    headers = {"Content-Type": "application/json"}
    if ep.get("key"):
        headers["Authorization"] = "Bearer " + ep["key"]
    r = _model_post(ep["url"], headers, body, stream=False)
    r.raise_for_status()
    d = r.json()
    ch = (d.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    return (msg.get("content") or "",
            msg.get("reasoning_content") or msg.get("reasoning") or "",
            msg.get("tool_calls") or None,
            d.get("usage"), ch.get("finish_reason"))


def clean_title(s):
    s = (s or "").strip()
    if not s:
        return ""
    s = s.splitlines()[0].strip()                                                  # first line only
    s = re.sub(r'^(?:title|chat title|conversation title)\s*[:\-–]\s*', '', s, flags=re.I)
    s = s.strip().strip('"\'`“”‘’').strip()                     # unwrap quotes
    s = re.sub(r'[\s.]+$', '', s)                                                   # trailing space/period
    words = s.split()
    if len(words) > 8:
        s = " ".join(words[:8])
    return s[:64].strip()


def utility_model(fallback):
    """Model for background utility calls (titles, context summaries). Admin-configurable so a
    small always-loaded model can do the chores instead of the (possibly huge) chat model."""
    return (get_setting("utility_model") or "").strip() or fallback


def generate_title(ep, model, user_text, assistant_text=""):
    """Ask the model for a short Title-Case name. Clean (no persona) prompt + low temp so even a
    creative model returns just a title. Returns "" on any failure (caller falls back to first line)."""
    model = utility_model(model)
    sys = ("You write a concise 3-5 word title in Title Case for a conversation. "
           "Reply with ONLY the title — no quotes, no punctuation, no preamble or explanation.")
    prompt = "Write a title for a conversation that begins with this message:\n\n" + (user_text or "")[:1500]
    if assistant_text:
        prompt += "\n\nThe reply began:\n" + assistant_text[:400]
    body = {"model": model, "stream": False, "max_tokens": 24, "temperature": 0.3,
            "messages": [{"role": "system", "content": sys}, {"role": "user", "content": prompt}]}
    headers = {"Content-Type": "application/json"}
    if ep.get("key"):
        headers["Authorization"] = "Bearer " + ep["key"]
    try:
        r = requests.post(ep["url"], headers=headers, json=body, timeout=25)
        r.raise_for_status()
        out = ((r.json().get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        return clean_title(out)
    except Exception:
        return ""


# ---------------------------------------------------------------- web push (optional)
# Requires pywebpush (pulls `cryptography`); when absent the feature simply stays hidden.
try:
    from pywebpush import webpush, WebPushException
    from cryptography.hazmat.primitives.asymmetric import ec as _ec
    PUSH_OK = True
except Exception:
    PUSH_OK = False


def _b64url(b):
    return base64.urlsafe_b64encode(b).decode("ascii").rstrip("=")


def vapid_keys():
    """(private_b64url, public_b64url) VAPID pair, generated once and stored in settings."""
    if not PUSH_OK:
        return None, None
    kp = get_setting("vapid_keys")
    if not kp:
        priv = _ec.generate_private_key(_ec.SECP256R1())
        d = priv.private_numbers().private_value.to_bytes(32, "big")
        pub = priv.public_key().public_numbers()
        raw = b"\x04" + pub.x.to_bytes(32, "big") + pub.y.to_bytes(32, "big")
        kp = {"private": _b64url(d), "public": _b64url(raw)}
        set_setting("vapid_keys", kp)
    return kp["private"], kp["public"]


# Hosts the big browser engines actually use for Web Push. The server POSTs to whatever endpoint
# a client registers, so anything outside this list is refused at subscribe time (SSRF guard).
_PUSH_HOST_SUFFIXES = (
    "fcm.googleapis.com",                  # Chrome / Chromium / Brave / Edge (new)
    "updates.push.services.mozilla.com",   # Firefox
    "push.services.mozilla.com",
    "web.push.apple.com",                  # Safari
    "notify.windows.com",                  # Edge (WNS)
)


def push_endpoint_allowed(endpoint):
    try:
        host = (urlparse(endpoint).hostname or "").lower().rstrip(".")
    except Exception:
        return False
    return bool(host) and any(host == s or host.endswith("." + s) for s in _PUSH_HOST_SUFFIXES)


def push_notify(user_id, title, body, cid=None):
    """Send a web-push to every subscription of a user; dead subscriptions are pruned. Runs in a
    background thread — never on the request path."""
    if not PUSH_OK:
        return

    def worker():
        priv, _ = vapid_keys()
        subs = db().execute("SELECT endpoint, sub FROM push_subs WHERE user_id=?", (user_id,)).fetchall()
        payload = json.dumps({"title": title, "body": (body or "")[:160], "cid": cid})
        for row in subs:
            if not push_endpoint_allowed(row["endpoint"]):   # also covers rows stored pre-allowlist
                with db():
                    db().execute("DELETE FROM push_subs WHERE endpoint=?", (row["endpoint"],))
                continue
            try:
                webpush(subscription_info=json.loads(row["sub"]), data=payload,
                        vapid_private_key=priv, vapid_claims={"sub": "mailto:admin@" + urlparse(PUBLIC_URL).netloc},
                        ttl=600)
            except WebPushException as e:
                code = getattr(getattr(e, "response", None), "status_code", None)
                if code in (404, 410):   # subscription expired/revoked
                    with db():
                        db().execute("DELETE FROM push_subs WHERE endpoint=?", (row["endpoint"],))
                else:
                    log.warning("push failed (%s): %s", code, e)
            except Exception:
                log.exception("push failed")
    threading.Thread(target=worker, daemon=True).start()


# Automatic context compression: when the sliding window drops turns, fold them into a rolling
# per-conversation summary (stored on the conversation) that rides along in the system prompt.
# KENOSIS_COMPRESS=0 restores plain dropping.
COMPRESS = os.environ.get("KENOSIS_COMPRESS", "1") not in ("0", "false", "no", "")


def summarize_dropped(ep, model, prior_summary, msgs):
    """One non-streaming call folding newly-dropped turns into the rolling summary. Returns the
    new summary text, or the prior one on failure/empty."""
    model = utility_model(model)
    lines = []
    for m in msgs:
        if m.get("role") not in ("user", "assistant"):
            continue
        content = (m.get("content") or "").strip()
        if content:
            lines.append(("User: " if m["role"] == "user" else "Assistant: ") + content[:4000])
    if not lines:
        return prior_summary
    sys = ("You maintain a running summary of a long conversation. Fold the prior summary and the new "
           "excerpt into ONE updated summary of at most 250 words. Keep names, facts, decisions, "
           "preferences, story/argument state, and open questions; drop pleasantries and repetition. "
           "Reply with ONLY the summary text.")
    parts = []
    if prior_summary:
        parts.append("Prior summary:\n" + prior_summary[:4000])
    parts.append("New excerpt:\n" + "\n\n".join(lines)[:24000])
    body = {"model": model, "stream": False, "max_tokens": 400, "temperature": 0.3,
            "messages": [{"role": "system", "content": sys},
                         {"role": "user", "content": "\n\n---\n\n".join(parts)}]}
    headers = {"Content-Type": "application/json"}
    if ep.get("key"):
        headers["Authorization"] = "Bearer " + ep["key"]
    r = requests.post(ep["url"], headers=headers, json=body, timeout=(10, 180))
    r.raise_for_status()
    out = ((r.json().get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    return out.strip() or prior_summary


def ensure_summary(convo, ep, model, dropped_msgs):
    """Rolling summary covering `dropped_msgs` (the window's dropped prefix, oldest first).
    Reuses the stored summary without any model call when it already covers everything dropped —
    which, thanks to trim hysteresis, is the common case — so the system prompt stays byte-stable
    (and cacheable) between trim events."""
    ids = [m.get("id") for m in dropped_msgs if m.get("id")]
    if not ids:
        return None
    stored, upto = convo.get("ctx_summary"), convo.get("ctx_summary_upto")
    if stored and upto in ids:
        new_msgs = dropped_msgs[ids.index(upto) + 1:]
        if not new_msgs:
            return stored
    else:
        stored, new_msgs = None, dropped_msgs   # first trim, or a branch the stored summary doesn't cover
    s = summarize_dropped(ep, model, stored, new_msgs)
    if s and s != stored:
        with db():
            db().execute("UPDATE conversations SET ctx_summary=?, ctx_summary_upto=? WHERE id=?",
                         (s, ids[-1], convo["id"]))
    return s


def embed_texts(texts):
    """Embed texts via the active endpoint's /v1/embeddings using the admin-configured embed model.
    Returns a list of vectors, or raises. Only called when `embed_model` is set."""
    ep = active_endpoint()
    model = (get_setting("embed_model") or "").strip()
    if not model:
        raise ValueError("no embed model configured")
    url = ep.get("url", "")
    url = url.replace("/chat/completions", "/embeddings") if "/chat/completions" in url else url.rstrip("/") + "/embeddings"
    headers = {"Content-Type": "application/json"}
    if ep.get("key"):
        headers["Authorization"] = "Bearer " + ep["key"]
    r = requests.post(url, headers=headers, json={"model": model, "input": texts}, timeout=(6, 60))
    r.raise_for_status()
    data = sorted(r.json().get("data") or [], key=lambda d: d.get("index", 0))
    return [d["embedding"] for d in data]


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (na * nb)


def knowledge_snippets(char_id, query, limit=4, budget=4000):
    """Best knowledge chunks for a character this turn. BM25 always; when an embed model is
    configured (and chunks have stored vectors) the two rankings are fused (reciprocal rank),
    so paraphrased questions still hit. Returns display-ready strings."""
    if not FTS_OK or not char_id:
        return []
    ranked = []   # list of (chunk, name) in preference order

    match = _fts_match_query(query, all_terms=False)
    bm25_rows = []
    if match:
        try:
            bm25_rows = db().execute(
                "SELECT chunk, name FROM knowledge_fts WHERE knowledge_fts MATCH ? AND character_id=? "
                "ORDER BY bm25(knowledge_fts) LIMIT 8", (match, char_id)).fetchall()
        except sqlite3.OperationalError:
            bm25_rows = []

    vec_rows = []
    if (get_setting("embed_model") or "").strip():
        try:
            cand = db().execute("SELECT chunk, vec, doc_id FROM knowledge_vec WHERE character_id=?",
                                (char_id,)).fetchall()
            if cand:
                qv = embed_texts([query[:2000]])[0]
                names = {k["id"]: k["name"] for k in
                         db().execute("SELECT id,name FROM knowledge WHERE character_id=?", (char_id,)).fetchall()}
                scored = sorted(((_cosine(qv, json.loads(r["vec"])), r) for r in cand),
                                key=lambda t: -t[0])[:8]
                vec_rows = [{"chunk": r["chunk"], "name": names.get(r["doc_id"], "notes")} for _, r in scored]
        except Exception as e:
            log.warning("embedding retrieval failed (BM25 only): %s", e)

    # reciprocal-rank fusion of the two lists, deduped by chunk text
    scores = {}
    for rank_list in (bm25_rows, vec_rows):
        for i, r in enumerate(rank_list):
            key = (r["chunk"] or "").strip()
            if not key:
                continue
            e = scores.setdefault(key, {"score": 0.0, "name": r["name"] or "notes"})
            e["score"] += 1.0 / (i + 3)
    ranked = sorted(scores.items(), key=lambda kv: -kv[1]["score"])

    out, used = [], 0
    for chunk, info in ranked[:limit * 2]:
        if len(out) >= limit or used + len(chunk) > budget:
            continue
        out.append("(%s) %s" % (info["name"], chunk))
        used += len(chunk)
    return out


def _chunk_text(text, target=1200):
    """Split document text into ~target-char chunks on paragraph boundaries."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if len(p) > target * 2:   # giant paragraph: hard-split
            for i in range(0, len(p), target):
                chunks.append(p[i:i + target])
            continue
        if cur and len(cur) + len(p) + 2 > target:
            chunks.append(cur)
            cur = p
        else:
            cur = (cur + "\n\n" + p) if cur else p
    if cur:
        chunks.append(cur)
    return chunks[:2000]


def _models_data(ep):
    headers = {}
    if ep.get("key"):
        headers["Authorization"] = "Bearer " + ep["key"]
    r = requests.get(models_url_for(ep), headers=headers, timeout=8)
    r.raise_for_status()
    return r.json().get("data", [])


_MODELS_CACHE = {}   # endpoint key -> (fetched_at, [ids]); keeps /api/config from blocking on /v1/models
_MODELS_LOCK = threading.Lock()


def fetch_models(ep, ttl=60):
    """Model ids from the endpoint, cached briefly per endpoint. On upstream failure the last
    known list is served (stale beats phantom); an endpoint that was never reachable yields []
    so the UI can say so honestly instead of listing models that don't exist."""
    key = ep.get("id") or ep.get("url") or "?"
    now = time.time()
    with _MODELS_LOCK:
        ent = _MODELS_CACHE.get(key)
        if ent and now - ent[0] < ttl:
            return list(ent[1])
    try:
        ids = [m["id"] for m in _models_data(ep)]
        ids.sort(key=lambda x: (x != DEFAULT_MODEL, not x.startswith("kenosis"), x))
    except Exception as e:
        log.warning("model list fetch failed for %s: %s", key, e)
        ids = list(ent[1]) if ent else []
    with _MODELS_LOCK:
        _MODELS_CACHE[key] = (now, ids)
    return list(ids)


def model_contexts(ep):
    """Map model id -> context window (from /v1/models max_model_len, with fallbacks)."""
    out = {}
    try:
        for m in _models_data(ep):
            ctx = (m.get("max_model_len") or m.get("max_context_length")
                   or m.get("context_length") or m.get("n_ctx"))
            if ctx:
                try:
                    out[m["id"]] = int(ctx)
                except (TypeError, ValueError):
                    pass
    except Exception:
        pass
    return out


_CTX_CACHE = {}      # endpoint id -> (fetched_at, {model: context_window}); 5-min TTL
_CTX_LOCK = threading.Lock()


def _ctx_data(ep):
    """Cached model -> context-window map for an endpoint. The staleness check happens under the
    lock and stamps a placeholder, so concurrent requests don't all refetch (thundering herd)."""
    key = ep.get("id") or ep.get("url") or "?"
    now = time.time()
    with _CTX_LOCK:
        ent = _CTX_CACHE.get(key)
        if ent and now - ent[0] < 300:
            return ent[1]
        _CTX_CACHE[key] = (now, ent[1] if ent else {})   # this thread refreshes; others use the old map
    try:
        m = model_contexts(ep)
    except Exception:
        m = ent[1] if ent else {}
    with _CTX_LOCK:
        _CTX_CACHE[key] = (now, m)
    return m


def context_for(ep, model, default=DEFAULT_CONTEXT):
    """Context window for a model. Returns `default` when the window is unknown — pass default=0
    to distinguish 'unknown' from a real value."""
    return _ctx_data(ep).get(model) or default


def shown_models(u):
    allm = fetch_models(active_endpoint())
    if u["role"] == "admin":
        return allm
    wl = allowed_models_for(u) or []
    return [m for m in allm if m in set(wl)] or list(wl)


def usage_stats():
    """Aggregate per-model usage from stored message meta (nothing new is instrumented).
    30-day and all-time buckets: replies, tokens, avg tok/s, avg ttft, avg cache-hit rate."""
    cutoff = (datetime.now().astimezone().timestamp()) - 30 * 86400
    agg = {}
    rows = db().execute("SELECT model, meta, ts FROM messages WHERE role='assistant' AND meta IS NOT NULL").fetchall()
    for r in rows:
        try:
            m = json.loads(r["meta"])
        except Exception:
            continue
        recent = False
        try:
            recent = datetime.fromisoformat(r["ts"]).timestamp() >= cutoff
        except Exception:
            pass
        b = agg.setdefault(r["model"] or "?", {"model": r["model"] or "?", "replies": 0, "tokens": 0,
                                               "tps": [], "ttft": [], "cache": [], "replies_30d": 0, "tokens_30d": 0})
        b["replies"] += 1
        b["tokens"] += m.get("completion_tokens") or 0
        if recent:
            b["replies_30d"] += 1
            b["tokens_30d"] += m.get("completion_tokens") or 0
        if m.get("tps"):
            b["tps"].append(m["tps"])
        if m.get("ttft_ms") is not None:
            b["ttft"].append(m["ttft_ms"])
        if m.get("cached_tokens") is not None and m.get("prompt_tokens"):
            b["cache"].append(m["cached_tokens"] / m["prompt_tokens"])
    out = []
    for b in agg.values():
        out.append({"model": b["model"], "replies": b["replies"], "tokens": b["tokens"],
                    "replies_30d": b["replies_30d"], "tokens_30d": b["tokens_30d"],
                    "avg_tps": round(sum(b["tps"]) / len(b["tps"]), 1) if b["tps"] else None,
                    "avg_ttft_ms": round(sum(b["ttft"]) / len(b["ttft"])) if b["ttft"] else None,
                    "avg_cache_pct": round(100 * sum(b["cache"]) / len(b["cache"])) if b["cache"] else None})
    out.sort(key=lambda x: -x["replies"])
    return {"models": out, "totals": {"replies": sum(x["replies"] for x in out),
                                      "tokens": sum(x["tokens"] for x in out)}}


def build_meta(t0, t_first, t1, usage, reply):
    usage = usage or {}
    comp = usage.get("completion_tokens")
    est = comp is None
    if est:
        comp = max(1, round(len(reply) / 4))
    gen = (t1 - t_first) if t_first else (t1 - t0)
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    return {"elapsed_ms": round((t1 - t0) * 1000),
            "ttft_ms": round((t_first - t0) * 1000) if t_first else None,
            "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": comp,
            "cached_tokens": cached,   # server-reported prompt-cache hit (None if not reported)
            "tokens_est": est, "tps": round((comp / gen) if gen > 0 else 0, 1)}


# Cooperative stop: a running generation polls this set each token and, when its conversation id
# appears, wraps up gracefully — persisting whatever has streamed so far and emitting its normal
# `done` event over the still-open stream (so the client keeps the partial, no disconnect race).
_STOP = set()
_STOP_LOCK = threading.Lock()


def request_stop(cid):
    with _STOP_LOCK:
        _STOP.add(cid)


def stop_requested(cid):
    with _STOP_LOCK:
        return cid in _STOP


def clear_stop(cid):
    with _STOP_LOCK:
        _STOP.discard(cid)


# ---------------------------------------------------------------- public sharing
def share_by_token(tok):
    return db().execute("SELECT * FROM shares WHERE token=?", (tok,)).fetchone()


def share_for_convo(cid):
    return db().execute("SELECT * FROM shares WHERE convo_id=?", (cid,)).fetchone()


def share_public_url(tok):
    return PUBLIC_URL + "/s/" + tok


def build_share_snapshot(cid):
    """A stripped-down, frozen copy of a conversation's visible thread for public viewing:
    user/assistant prose only — no system prompt, reasoning, tool steps, params, or file bodies."""
    convo = get_convo(cid, None)
    if convo is None:
        return None
    msgs = []
    for m in convo.get("messages", []):
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        content = (m.get("content") or "")
        if role == "assistant" and "<tool_call>" in content:
            content = strip_tool_calls(content)
        atts = [a.get("name") or "file" for a in (m.get("attachments") or []) if a.get("name") or a.get("text")]
        if not content.strip() and not atts:
            continue  # e.g. an intermediate tool-only assistant turn
        entry = {"role": role, "content": content, "ts": m.get("ts")}
        if atts:
            entry["files"] = atts
        msgs.append(entry)
    return {"title": convo.get("title") or "Untitled conversation", "messages": msgs}


def upsert_share(cid, owner_id):
    snap = build_share_snapshot(cid)
    if snap is None:
        return None
    data = json.dumps(snap, ensure_ascii=False)
    existing = share_for_convo(cid)
    now = _now()
    with db():
        if existing:
            db().execute("UPDATE shares SET title=?, data=?, updated=? WHERE token=?",
                         (snap["title"], data, now, existing["token"]))
            tok = existing["token"]
        else:
            tok = gen_token()
            db().execute("INSERT INTO shares(token,convo_id,owner_id,title,data,views,created,updated)"
                         " VALUES(?,?,?,?,?,0,?,?)",
                         (tok, cid, owner_id, snap["title"], data, now, now))
    return share_for_convo(cid)


# ---------------------------------------------------------------- tools / fetch
class _TextExtractor(HTMLParser):
    """Very small HTML -> readable-text reducer (stdlib only)."""
    _SKIP = {"script", "style", "noscript", "template", "svg", "head", "nav", "footer"}
    _BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
              "section", "article", "header", "blockquote", "pre"}

    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = 0
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self.skip += 1
        if tag == "title":
            self._in_title = True
        if tag in self._BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self._SKIP and self.skip:
            self.skip -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data
            return
        if self.skip:
            return
        t = data.strip()
        if t:
            self.parts.append(t)

    def text(self):
        s = " ".join(self.parts)
        s = re.sub(r"[ \t]+", " ", s)
        s = re.sub(r" *\n *", "\n", s)
        s = re.sub(r"\n{3,}", "\n\n", s)
        return s.strip()


class _PinnedHTTPSAdapter(requests.adapters.HTTPAdapter):
    """Connect to a pre-validated IP while verifying TLS against the real hostname (SNI + cert)."""
    def __init__(self, hostname):
        self._hostname = hostname
        super().__init__(max_retries=0)

    def init_poolmanager(self, *a, **kw):
        kw["assert_hostname"] = self._hostname
        kw["server_hostname"] = self._hostname
        return super().init_poolmanager(*a, **kw)


def _resolve_public_ip(host):
    """Resolve a host and return one public IP, refusing any private/internal address (SSRF guard)."""
    ip = None
    for info in socket.getaddrinfo(host, None):
        addr = ipaddress.ip_address(info[4][0])
        if (addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved
                or addr.is_multicast or addr.is_unspecified):
            raise ValueError("refusing to fetch a private/internal address (%s)" % addr)
        if ip is None:
            ip = str(addr)
    if ip is None:
        raise ValueError("could not resolve host")
    return ip


def safe_fetch(url):
    """Fetch a public http(s) page with SSRF guards. DNS is pinned to a validated IP (defeats
    rebinding) and every redirect hop is validated before it is followed. Returns (title, text, final_url)."""
    sess = requests.Session()
    hops = 0
    raw, ctype = b"", ""
    encoding = "utf-8"
    while True:
        p = urlparse(url)
        if p.scheme not in ("http", "https"):
            raise ValueError("only http/https URLs are allowed")
        if not p.hostname:
            raise ValueError("missing host in URL")
        ip = _resolve_public_ip(p.hostname)                      # validate + pin
        port = p.port or (443 if p.scheme == "https" else 80)
        defport = port in (80, 443)
        host_hdr = p.hostname if defport else "%s:%d" % (p.hostname, port)
        ipnet = ("[%s]" % ip) if ":" in ip else ip
        ip_url = urlunparse(p._replace(netloc=(ipnet if defport else "%s:%d" % (ipnet, port))))
        if p.scheme == "https":
            sess.mount("https://", _PinnedHTTPSAdapter(p.hostname))
        headers = {"Host": host_hdr, "User-Agent": "ORACLE-fetch/1.0",
                   "Accept": "text/html,text/plain;q=0.9,*/*;q=0.5"}
        r = sess.get(ip_url, headers=headers, timeout=FETCH_TIMEOUT, stream=True, allow_redirects=False)
        if r.is_redirect or r.is_permanent_redirect:
            loc = r.headers.get("Location", ""); r.close()
            hops += 1
            if hops > 5:
                raise ValueError("too many redirects")
            if not loc:
                raise ValueError("redirect without a location")
            url = urljoin(url, loc)                              # next hop re-validated at loop top
            continue
        ctype = r.headers.get("Content-Type", "")
        try:
            r.raise_for_status()
            for chunk in r.iter_content(8192):
                raw += chunk
                if len(raw) > FETCH_MAX_BYTES:
                    break
            encoding = r.encoding or "utf-8"
        finally:
            r.close()
        break
    body = raw.decode(encoding, errors="replace")
    title = ""
    if "html" in ctype or body.lstrip()[:1] == "<":
        ex = _TextExtractor()
        try:
            ex.feed(body)
        except Exception:
            pass
        title, text = ex.title.strip(), ex.text()
    else:
        text = body
    if len(text) > FETCH_MAX_CHARS:
        text = text[:FETCH_MAX_CHARS] + "\n\n[...truncated...]"
    return title, text, url


# Some models are tool-trained and emit a tool call as plain text (e.g. "<tool_call><function=fetch__fetch>
# <parameter=url>...</parameter></function></tool_call>") instead of via the structured tool_calls channel,
# especially the MLX server doesn't parse it when no tools were offered. Parse/strip those so they never leak.
_TOOLCALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
_FUNC_RE = re.compile(r"<function=([^>\s]+)\s*>(.*?)</function>", re.S)
_PARAM_RE = re.compile(r"<parameter=([^>\s]+)\s*>\s*(.*?)\s*</parameter>", re.S)
_FETCH_ALIASES = {"fetch_url", "fetch", "fetch__fetch", "browse", "open_url", "get_url", "web_fetch", "url_fetch"}


def parse_text_tool_calls(text):
    """Recover tool calls a model leaked into plain text (XML-tag or JSON forms)."""
    calls = []
    for i, block in enumerate(_TOOLCALL_RE.findall(text or "")):
        block = block.strip()
        name, args = None, {}
        mf = _FUNC_RE.search(block)
        if mf:
            name = mf.group(1).strip()
            for pk, pv in _PARAM_RE.findall(mf.group(2)):
                args[pk.strip()] = pv.strip()
        else:
            try:
                obj = json.loads(block)
                name = obj.get("name")
                args = obj.get("arguments") or obj.get("parameters") or {}
                if isinstance(args, str):
                    args = json.loads(args)
            except Exception:
                pass
        if name:
            calls.append({"id": "call_%d" % i, "type": "function",
                          "function": {"name": name,
                                       "arguments": json.dumps(args if isinstance(args, dict) else {})}})
    return calls


def strip_tool_calls(text):
    if not text or "<tool_call>" not in text:
        return text
    text = _TOOLCALL_RE.sub("", text)
    text = re.sub(r"<tool_call>.*$", "", text, flags=re.S)  # drop a dangling/unclosed block
    return text.strip()


FETCH_TOOL = {
    "type": "function",
    "function": {
        "name": "fetch_url",
        "description": ("Fetch a public web page (or plain-text/JSON URL) over http(s) and return its "
                        "readable text. Use this to look up current information or to read a link the "
                        "user shares. Only public internet addresses are allowed."),
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "Absolute http(s) URL to fetch."}},
            "required": ["url"],
        },
    },
}
SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": ("Search the web and return the top results (title, URL, snippet). Use this to "
                        "find current information or pages to read; follow up with fetch_url on a "
                        "promising result when you need the full text."),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The search query."}},
            "required": ["query"],
        },
    },
}
CALC_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": ("Evaluate an arithmetic expression exactly (+ - * / // % ** parentheses, and "
                        "functions like sqrt, sin, cos, log, exp, floor, ceil; constants pi and e). "
                        "Use this instead of doing nontrivial arithmetic yourself."),
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "e.g. (17.5*365)/sqrt(2)"}},
            "required": ["expression"],
        },
    },
}


def tools_spec():
    """The tool list offered to the model. web_search appears only when an admin has configured a
    SearXNG endpoint, so the model is never offered a tool that can't work."""
    spec = [FETCH_TOOL, CALC_TOOL]
    if (get_setting("search_url") or "").strip():
        spec.insert(1, SEARCH_TOOL)
    return spec


# ---- calculator: whitelisted-AST arithmetic — no names, attributes, or subscripts are reachable
_CALC_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
             ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
             ast.Mod: operator.mod, ast.Pow: operator.pow}
_CALC_FUNCS = {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
               "asin": math.asin, "acos": math.acos, "atan": math.atan, "atan2": math.atan2,
               "log": math.log, "log10": math.log10, "log2": math.log2, "exp": math.exp,
               "abs": abs, "round": round, "floor": math.floor, "ceil": math.ceil,
               "min": min, "max": max, "degrees": math.degrees, "radians": math.radians,
               "factorial": math.factorial}
_CALC_CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau}


def safe_calc(expr):
    expr = (expr or "").strip()[:400].replace("^", "**")   # models often write ^ for power
    tree = ast.parse(expr, mode="eval")

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return n.value
        if isinstance(n, ast.Name) and n.id in _CALC_CONSTS:
            return _CALC_CONSTS[n.id]
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            v = ev(n.operand)
            return v if isinstance(n.op, ast.UAdd) else -v
        if isinstance(n, ast.BinOp) and type(n.op) in _CALC_OPS:
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and (abs(b) > 4096 or abs(a) > 1e100):
                raise ValueError("exponent out of range")
            return _CALC_OPS[type(n.op)](a, b)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _CALC_FUNCS and not n.keywords:
            args = [ev(a) for a in n.args]
            if n.func.id == "factorial" and (args and (args[0] > 5000 or args[0] < 0)):
                raise ValueError("factorial out of range")
            return _CALC_FUNCS[n.func.id](*args)
        raise ValueError("unsupported expression element: %s" % type(n).__name__)

    return ev(tree)


def run_web_search(query):
    """Query the admin-configured SearXNG instance; returns readable top results for the model."""
    base = (get_setting("search_url") or "").strip().rstrip("/")
    if not base:
        raise ValueError("web search is not configured on this server")
    r = requests.get(base + "/search", params={"q": query, "format": "json"},
                     headers={"User-Agent": "oracle-chat/1.0"}, timeout=(6, FETCH_TIMEOUT))
    r.raise_for_status()
    results = (r.json().get("results") or [])[:6]
    if not results:
        return "No results found for: %s" % query, 0
    lines = []
    for i, res in enumerate(results, 1):
        lines.append("%d. %s\n   %s\n   %s" % (i, (res.get("title") or "").strip()[:200],
                                               (res.get("url") or "").strip()[:400],
                                               " ".join(((res.get("content") or "").split()))[:400]))
    return "Search results for \"%s\":\n\n%s" % (query, "\n\n".join(lines)), len(results)


_URL_RE = re.compile(r'https?://[^\s<>"\')\]]+')
_FETCHY = ("fetch", "browse", "scrape", "crawl", "url", "http", "web", "visit", "page", "read", "open", "get", "curl", "wget", "request")


def _url_from_args(args):
    for key in ("url", "uri", "link", "href", "address", "page", "target", "input", "query", "q", "text"):
        v = args.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    for v in args.values():  # any value that looks like a URL
        if isinstance(v, str) and v.strip().lower().startswith(("http://", "https://")):
            return v.strip()
    return ""


def execute_tool(name, args, fallback_url=None):
    """Run one tool call. Tolerant of the names tool-trained models invent (fetch__fetch, fetch__fetcher,
    browse, …) and recovers the URL from conversation context when the model omits it. Returns (text, ui)."""
    nm = (name or "").lower()
    if not isinstance(args, dict):
        args = {}
    if nm in ("calculate", "calc", "calculator", "math", "evaluate"):
        expr = str(args.get("expression") or args.get("expr") or args.get("input") or "")
        try:
            val = safe_calc(expr)
            out = repr(val) if isinstance(val, int) else ("%.12g" % val)
            return ("%s = %s" % (expr.strip(), out), {"ok": True, "summary": "%s = %s" % (expr.strip()[:60], out[:40])})
        except Exception as e:
            return ("Error evaluating '%s': %s" % (expr[:200], e), {"ok": False, "summary": "calc error"})
    if nm in ("web_search", "search", "websearch", "search_web", "google"):
        query = str(args.get("query") or args.get("q") or args.get("input") or "").strip()
        if not query:
            return ("Error: no query given. Call web_search again with a \"query\" argument.",
                    {"ok": False, "summary": "no query"})
        try:
            text, n = run_web_search(query)
            return text, {"ok": True, "summary": "%d results: %s" % (n, query[:80])}
        except Exception as e:
            return "Error searching for '%s': %s" % (query[:120], e), {"ok": False, "summary": str(e)}
    url = _url_from_args(args)
    looks_fetch = (nm in _FETCH_ALIASES) or any(k in nm for k in _FETCHY)
    if looks_fetch or url:
        if not url:
            url = (fallback_url or "").strip()
        if not url:
            return ("Error: no URL was provided. Call the tool again with the user's link in the \"url\" argument.",
                    {"ok": False, "url": "", "summary": "no url provided"})
        try:
            title, text, final = safe_fetch(url)
            head = ("Title: %s\n" % title) if title else ""
            return ("Fetched %s\n%s\n%s" % (final, head, text),
                    {"ok": True, "url": final, "title": title, "chars": len(text), "summary": title or final})
        except Exception as e:
            return "Error fetching %s: %s" % (url, e), {"ok": False, "url": url, "summary": str(e)}
    return ("Error: unknown tool '%s'. Available tools: %s." % (name, ", ".join(t["function"]["name"] for t in tools_spec())),
            {"ok": False, "summary": "unknown tool: %s" % name})


def extract_text_file(name, raw):
    """Extract plain text from an uploaded file (txt/md/etc. or PDF). Raises ValueError on failure."""
    lower = (name or "").lower()
    if lower.endswith(".pdf"):
        try:
            import pypdf
        except Exception:
            raise ValueError("PDF support is not installed on the server (pypdf).")
        import io
        try:
            reader = pypdf.PdfReader(io.BytesIO(raw))
        except Exception as e:
            raise ValueError("could not read PDF: %s" % e)
        pages = []
        for pg in reader.pages:
            try:
                pages.append(pg.extract_text() or "")
            except Exception:
                pages.append("")
        text = "\n\n".join(pages).strip()
        if not text:
            raise ValueError("no extractable text (scanned/image PDF? OCR is not supported)")
        return text
    if lower.endswith(TEXT_EXTS):
        return raw.decode("utf-8", errors="replace")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("unsupported file type")


# ---------------------------------------------------------------- HTTP handler
class _BodyTooLarge(Exception):
    pass


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"   # keep-alive: reuse the TCP connection (and its thread) across requests
    timeout = 60                    # drop connections whose client stalls mid-request

    def log_message(self, format, *args):   # noqa: A002 (matches the base-class signature)
        log.info("%s %s", self._client_ip(), format % args)

    def _send(self, code, ctype, body, extra=None):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "strict-origin-when-cross-origin")
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
        self._send(200, "text/html; charset=utf-8", body,
                   [("Content-Security-Policy", csp_for(body)), ("X-Frame-Options", "DENY")])

    def _html_static(self, body_bytes, gz_bytes, etag):
        """Serve a startup-precompressed HTML page with ETag revalidation (304s + gzip)."""
        if self.headers.get("If-None-Match") == etag:
            self.send_response(304)
            self.send_header("ETag", etag)
            self.send_header("Cache-Control", "private, no-cache")   # 304 must echo the 200's cache headers
            self.send_header("Vary", "Accept-Encoding")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        extra = [("Content-Security-Policy", PAGE_CSP), ("X-Frame-Options", "DENY"),
                 ("ETag", etag), ("Cache-Control", "private, no-cache"), ("Vary", "Accept-Encoding")]
        use_gz = gz_bytes is not None and "gzip" in (self.headers.get("Accept-Encoding") or "")
        if use_gz:
            extra.append(("Content-Encoding", "gzip"))
        self._send(200, "text/html; charset=utf-8", gz_bytes if use_gz else body_bytes, extra)

    def _client_ip(self):
        # Behind the documented Cloudflare tunnel the real client IP arrives in CF-Connecting-IP;
        # fall back to X-Forwarded-For, then the socket peer. Only used for best-effort throttling
        # and logging. KENOSIS_TRUST_PROXY=0 ignores the (spoofable) headers entirely.
        if TRUST_PROXY:
            for h in ("CF-Connecting-IP", "X-Forwarded-For"):
                v = self.headers.get(h)
                if v:
                    return v.split(",")[0].strip()
        return self.client_address[0] if self.client_address else "?"

    def _redirect(self, location):
        self.send_response(302)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _read_chunked(self):
        """Decode a Transfer-Encoding: chunked request body. BaseHTTPRequestHandler never does
        this itself, and Cloudflare forwards some client uploads chunked (e.g. Safari over
        HTTP/2/3 with no upfront length); without decoding, the handler saw an empty body and
        the unread chunk bytes were misparsed as the next request line on the kept-alive
        connection ("Bad request syntax ('28')")."""
        data = bytearray()
        while True:
            line = self.rfile.readline(1024)
            if not line.endswith(b"\n"):
                raise ValueError("malformed chunk size line")
            size = int(line.split(b";", 1)[0].strip() or b"0", 16)
            if size == 0:
                while True:   # consume optional trailers up to the terminating blank line
                    t = self.rfile.readline(1024)
                    if t in (b"\r\n", b"\n", b""):
                        break
                return bytes(data)
            if len(data) + size > MAX_BODY_BYTES:
                raise _BodyTooLarge()
            remaining = size
            while remaining:
                piece = self.rfile.read(min(remaining, 65536))
                if not piece:
                    raise ValueError("truncated chunk")
                data.extend(piece)
                remaining -= len(piece)
            self.rfile.read(2)   # CRLF after each chunk

    def _read(self):
        if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
            try:
                raw = self._read_chunked()
            except _BodyTooLarge:
                raise
            except Exception:
                self.close_connection = True   # partially-read body would poison the keep-alive stream
                return {}
            try:
                return json.loads(raw.decode("utf-8"))
            except Exception:
                return {}
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except (TypeError, ValueError):
            n = 0
        if n <= 0:   # also rejects a negative Content-Length, which would read until EOF
            return {}
        if n > MAX_BODY_BYTES:
            raise _BodyTooLarge()
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
        sess = parse_session(tok)
        if not sess or not sess[0]:
            return None
        u = user_by_name(sess[0])
        if u is None or u["disabled"]:
            return None
        if (sess[1] or 0) != user_session_version(u):   # password changed -> old cookies are dead
            return None
        return u

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
            "presets": visible_presets(u),
            "folders": list_folders(u), "param_specs": PARAM_SPECS, "max_tokens": MAX_TOKENS,
            "default_system": get_setting("default_system", DEFAULT_SYSTEM),
            "default_params": get_setting("default_params", {}),
            "model_defaults": get_setting("model_defaults", {}),
            "default_model": get_setting("default_model", DEFAULT_MODEL),
            "model_contexts": _ctx_data(active_endpoint()),
            "default_context": DEFAULT_CONTEXT,
            "thinking_models": get_setting("thinking_models", []),
            "vision_models": get_setting("vision_models", []),
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
        if path == "/healthz":   # for container orchestration; no auth, reveals nothing
            try:
                db().execute("SELECT 1")
                return self._json(200, {"ok": True})
            except Exception:
                return self._json(500, {"ok": False})
        if path == "/manifest.webmanifest":
            return self._send(200, "application/manifest+json", MANIFEST_JSON,
                              [("Cache-Control", "public, max-age=86400")])
        if path == "/sw.js":
            return self._send(200, "application/javascript; charset=utf-8", SW_JS,
                              [("Cache-Control", "no-cache")])
        if path == "/favicon.svg":
            return self._send(200, "image/svg+xml; charset=utf-8", FAVICON_SVG,
                              [("Cache-Control", "public, max-age=604800")])
        if path == "/og-image.png":
            return self._send(200, "image/png", og_image_png(),
                              [("Cache-Control", "public, max-age=86400")])
        if path in ("/apple-touch-icon.png", "/apple-touch-icon-precomposed.png"):
            return self._send(200, "image/png", apple_icon_png(),
                              [("Cache-Control", "public, max-age=604800")])
        if path == "/api/setup-status":
            return self._json(200, {"needs_setup": user_count() == 0})
        if path == "/login":
            if self.current_user():
                return self._redirect("/")
            return self._redirect("/setup") if user_count() == 0 else self._html(LOGIN_PAGE)
        if path == "/setup":
            return self._redirect("/login") if user_count() > 0 else self._html(SETUP_PAGE)
        if re.fullmatch(r"/invite/([A-Za-z0-9_-]+)", path):
            if self.current_user():
                return self._redirect("/")
            return self._html(INVITE_PAGE)
        m = re.fullmatch(r"/api/invite/([A-Za-z0-9_-]+)", path)
        if m:
            inv = invite_by_token(m.group(1))
            if not invite_valid(inv):
                return self._json(200, {"valid": False, "error": invite_error(inv)})
            return self._json(200, {"valid": True, "role": inv["role"]})

        # --- public, unauthenticated share viewer (frozen snapshot; no system prompt / private data)
        m = re.fullmatch(r"/s/([A-Za-z0-9_-]+)", path)
        if m:
            sh = share_by_token(m.group(1))
            if sh is None:
                return self._send(404, "text/html; charset=utf-8", SHARE_404,
                                  [("Content-Security-Policy", csp_for(SHARE_404))])
            try:
                with db():
                    db().execute("UPDATE shares SET views=views+1 WHERE token=?", (sh["token"],))
            except Exception:
                log.exception("share view-count update failed")
            # CSP comes from the static TEMPLATE, never the rendered page: hashing rendered output
            # would auto-allow any script an attacker managed to inject into share content
            return self._send(200, "text/html; charset=utf-8", render_share_page(sh),
                              [("Content-Security-Policy", SHARE_CSP), ("X-Robots-Tag", "noindex, nofollow")])

        u = self.current_user()
        if path == "/":
            if user_count() == 0:
                return self._redirect("/setup")
            return self._html_static(PAGE_BYTES, PAGE_GZ, PAGE_ETAG) if u else self._redirect("/login")
        if not u:
            return self._json(401, {"error": "auth required"})

        if path == "/api/config":
            return self._json(200, self.config_payload(u))
        if path == "/api/me":
            return self._json(200, {"me": user_public(u), "is_admin": u["role"] == "admin"})
        if path == "/api/push/key":
            _, pub = vapid_keys()
            return self._json(200, {"key": pub})   # None -> push unavailable on this server
        if path == "/api/stats":
            if u["role"] != "admin":
                return self._json(403, {"error": "admin only"})
            return self._json(200, usage_stats())
        if path == "/api/models":
            qs = parse_qs(parsed.query)
            if u["role"] == "admin":
                eid = (qs.get("endpoint") or [None])[0]
                return self._json(200, {"models": fetch_models(endpoint_by_id(eid) if eid else active_endpoint())})
            return self._json(200, {"models": shown_models(u)})
        if path == "/api/characters":
            return self._json(200, {"characters": visible_characters(u)})
        if path == "/api/presets":
            return self._json(200, {"presets": visible_presets(u)})
        if path == "/api/folders":
            return self._json(200, {"folders": list_folders(u)})
        if path == "/api/conversations":
            return self._json(200, {"conversations": list_convos(u)})
        if path == "/api/search":
            q = (parse_qs(parsed.query).get("q") or [""])[0]
            return self._json(200, {"results": search_convo_content(u, q)})
        if path == "/api/export":
            rows = db().execute("SELECT id FROM conversations WHERE owner_id=? ORDER BY updated DESC", (u["id"],)).fetchall()
            convos = [convo_export(r["id"]) for r in rows]
            fname = "oracle-export-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".json"
            return self._send(200, "application/json; charset=utf-8",
                              json.dumps({"generated": _now(), "user": u["username"], "conversations": convos}, indent=2, ensure_ascii=False),
                              [("Content-Disposition", 'attachment; filename="%s"' % fname)])

        # full-tree export of one conversation (all branches, ratings, reasoning, tool steps)
        m = re.fullmatch(r"/api/conversations/([^/]+)/export", path)
        if m and valid_id(m.group(1)):
            if get_convo(m.group(1), u) is None:
                return self._json(404, {"error": "not found"})
            return self._json(200, convo_export(m.group(1)))
        if path == "/api/users":
            if u["role"] != "admin":
                return self._json(403, {"error": "admin only"})
            return self._json(200, {"users": [user_public(r) for r in db().execute("SELECT * FROM users ORDER BY id").fetchall()]})
        if path == "/api/invites":
            if u["role"] != "admin":
                return self._json(403, {"error": "admin only"})
            return self._json(200, {"invites": list_invites()})

        # --- admin read-only peek at another user's chats + characters (no write endpoints exist)
        m = re.fullmatch(r"/api/admin/users/([0-9]+)", path)
        if m:
            if u["role"] != "admin":
                return self._json(403, {"error": "admin only"})
            tu = user_by_id(int(m.group(1)))
            if tu is None:
                return self._json(404, {"error": "not found"})
            chars = [{"id": r["id"], "name": r["name"], "avatar": r["avatar"] or "", "model": r["model"],
                      "scope": r["scope"], "system": r["system"] or ""}
                     for r in db().execute("SELECT * FROM characters WHERE owner_id=? ORDER BY name COLLATE NOCASE", (tu["id"],)).fetchall()]
            return self._json(200, {"user": user_public(tu), "conversations": list_convos(tu), "characters": chars})
        m = re.fullmatch(r"/api/admin/conversations/([^/]+)", path)
        if m and valid_id(m.group(1)):
            if u["role"] != "admin":
                return self._json(403, {"error": "admin only"})
            c = get_convo(m.group(1), None)   # None owner = bypass ownership (read-only view)
            return self._json(200, c) if c else self._json(404, {"error": "not found"})

        m = re.fullmatch(r"/api/conversations/([^/]+)/share", path)
        if m and valid_id(m.group(1)):
            if get_convo(m.group(1), u) is None:
                return self._json(404, {"error": "not found"})
            sh = share_for_convo(m.group(1))
            if sh is None:
                return self._json(200, {"shared": False})
            return self._json(200, {"shared": True, "token": sh["token"], "url": share_public_url(sh["token"]),
                                    "views": sh["views"], "updated": sh["updated"]})

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

        m = re.fullmatch(r"/api/conversations/([^/]+)/share", path)
        if m:
            cid = m.group(1)
            if not get_convo(cid, u):
                return self._json(404, {"error": "not found"})
            with db():
                db().execute("DELETE FROM shares WHERE convo_id=?", (cid,))
            return self._json(200, {"shared": False})

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
                c.execute("DELETE FROM shares WHERE convo_id=?", (cid,))
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

        m = re.fullmatch(r"/api/characters/([^/]+)/knowledge/([^/]+)", path)
        if m:
            ch = character_by_id(m.group(1))
            if ch is None or not (ch["owner_id"] == u["id"] or u["role"] == "admin"):
                return self._json(404, {"error": "not found"})
            with db():
                db().execute("DELETE FROM knowledge WHERE id=? AND character_id=?", (m.group(2), ch["id"]))
                db().execute("DELETE FROM knowledge_fts WHERE doc_id=?", (m.group(2),))
                db().execute("DELETE FROM knowledge_vec WHERE doc_id=?", (m.group(2),))
            return self._json(200, {"ok": True, "knowledge": knowledge_list(ch["id"])})

        m = re.fullmatch(r"/api/characters/([^/]+)", path)
        if m:
            ch = character_by_id(m.group(1))
            if ch is None:
                return self._json(404, {"error": "not found"})
            if not (ch["owner_id"] == u["id"] or u["role"] == "admin"):
                return self._json(403, {"error": "not yours"})
            with db():
                db().execute("DELETE FROM characters WHERE id=?", (m.group(1),))
                if FTS_OK:
                    db().execute("DELETE FROM knowledge WHERE character_id=?", (m.group(1),))
                    db().execute("DELETE FROM knowledge_fts WHERE character_id=?", (m.group(1),))
                    db().execute("DELETE FROM knowledge_vec WHERE character_id=?", (m.group(1),))
            return self._json(200, {"characters": visible_characters(u)})

        m = re.fullmatch(r"/api/presets/([^/]+)", path)
        if m:
            ps = preset_by_id(m.group(1))
            if ps is None:
                return self._json(404, {"error": "not found"})
            if not (ps["owner_id"] == u["id"] or u["role"] == "admin"):
                return self._json(403, {"error": "not yours"})
            with db():
                db().execute("DELETE FROM presets WHERE id=?", (m.group(1),))
            return self._json(200, {"presets": visible_presets(u)})

        m = re.fullmatch(r"/api/invites/([A-Za-z0-9_-]+)", path)
        if m:
            if u["role"] != "admin":
                return self._json(403, {"error": "admin only"})
            with db():
                db().execute("DELETE FROM invites WHERE token=?", (m.group(1),))
            return self._json(200, {"invites": list_invites()})

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
        try:
            payload = self._read()
        except _BodyTooLarge:
            self.close_connection = True   # the unread body would poison the keep-alive stream
            return self._json(413, {"error": "request body too large"})

        if path == "/api/login":
            if not self._csrf_ok():
                return self._json(403, {"error": "bad origin"})
            name = (payload.get("username") or "").strip()
            # throttle per source IP *and* per target account, so a spoofed/rotating
            # forwarded-IP header still can't brute-force one username
            keys = [self._client_ip()] + (["u:" + name.lower()] if name else [])
            retry = max(login_retry_after(k) for k in keys)
            if retry:
                return self._json(429, {"error": "too many failed attempts; try again in %d seconds" % retry},
                                  extra=[("Retry-After", str(retry))])
            u = user_by_name(name)
            if u is None or u["disabled"] or not verify_pw(payload.get("password") or "", u["pw_hash"]):
                for k in keys:
                    login_record_fail(k)
                return self._json(401, {"error": "invalid username or password"})
            secret = user_totp_secret(u)
            if secret:
                if not (payload.get("totp") or "").strip():
                    return self._json(200, {"ok": False, "totp_required": True})
                if not totp_verify(secret, payload.get("totp")):
                    for k in keys:   # wrong codes count toward the same throttle as wrong passwords
                        login_record_fail(k)
                    return self._json(401, {"error": "invalid authentication code", "totp_required": True})
            for k in keys:
                login_clear(k)
            return self._json(200, {"ok": True, "me": user_public(u)},
                              extra=[self._set_cookie(sign_session(u["username"], user_session_version(u)))])

        if path == "/api/setup":
            if not self._csrf_ok():
                return self._json(403, {"error": "bad origin"})
            name = (payload.get("username") or "").strip()
            pw = payload.get("password") or ""
            if not re.fullmatch(r"[A-Za-z0-9_.-]{2,32}", name):
                return self._json(400, {"error": "username must be 2-32 chars: letters, digits, _ . -"})
            if len(pw) < 8:
                return self._json(400, {"error": "password must be at least 8 characters"})
            with _init_lock:   # check + create atomically: two racing setups can't both make an admin
                if user_count() > 0:
                    return self._json(403, {"error": "already set up"})
                u = create_user(name, pw, role="admin")
            import_legacy(u["id"])
            return self._json(200, {"ok": True, "me": user_public(u)}, extra=[self._set_cookie(sign_session(name))])

        if path == "/api/invite/register":
            if not self._csrf_ok():
                return self._json(403, {"error": "bad origin"})
            tok = (payload.get("token") or "").strip()
            name = (payload.get("username") or "").strip()
            pw = payload.get("password") or ""
            inv = invite_by_token(tok)
            if not invite_valid(inv):
                return self._json(400, {"error": invite_error(inv)})
            if not re.fullmatch(r"[A-Za-z0-9_.-]{2,32}", name):
                return self._json(400, {"error": "username must be 2-32 chars: letters, digits, _ . -"})
            if len(pw) < 8:
                return self._json(400, {"error": "password must be at least 8 characters"})
            if user_by_name(name):
                return self._json(400, {"error": "that username is already taken"})
            role = "admin" if inv["role"] == "admin" else "user"
            allowed = inv["allowed_models"]  # already JSON or None; store as-is
            conn = db()
            with conn:
                # re-check inside the transaction so concurrent sign-ups can't overrun max_uses
                cur = conn.execute("SELECT uses, max_uses, expires FROM invites WHERE token=?", (tok,)).fetchone()
                if cur is None or (cur["expires"] is not None and time.time() >= cur["expires"]) \
                        or (cur["max_uses"] is not None and cur["uses"] >= cur["max_uses"]):
                    return self._json(400, {"error": "This invite link is no longer valid."})
                if user_by_name(name):
                    return self._json(400, {"error": "that username is already taken"})
                conn.execute("INSERT INTO users(username,pw_hash,role,allowed_models,created) VALUES(?,?,?,?,?)",
                             (name, hash_pw(pw), role, allowed, _now()))
                conn.execute("UPDATE invites SET uses=uses+1 WHERE token=?", (tok,))
            nu = user_by_name(name)
            return self._json(200, {"ok": True, "me": user_public(nu)},
                              extra=[self._set_cookie(sign_session(name, user_session_version(nu)))])

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
            newv = user_session_version(u) + 1   # revoke every other signed-in device
            with db():
                db().execute("UPDATE users SET pw_hash=?, session_version=? WHERE id=?",
                             (hash_pw(payload["new"]), newv, u["id"]))
            # re-issue this session at the new version so the current device stays signed in
            return self._json(200, {"ok": True}, extra=[self._set_cookie(sign_session(u["username"], newv))])

        if path == "/api/account":
            persona = (payload.get("persona") or "").strip()[:60]
            with db():
                db().execute("UPDATE users SET persona=? WHERE id=?", (persona or None, u["id"]))
            return self._json(200, {"me": user_public(user_by_id(u["id"]))})

        # ---- TOTP 2FA. Setup is stateless: the server hands out a candidate secret, and confirm
        # only persists it after the user proves their authenticator produces matching codes.
        if path == "/api/account/totp/setup":
            secret = totp_gen_secret()
            uri = "otpauth://totp/ORACLE:%s?secret=%s&issuer=ORACLE" % (u["username"], secret)
            return self._json(200, {"secret": secret, "uri": uri})
        if path == "/api/account/totp/confirm":
            secret = (payload.get("secret") or "").strip()
            if not re.fullmatch(r"[A-Z2-7]{16,64}", secret):
                return self._json(400, {"error": "bad secret"})
            if not totp_verify(secret, payload.get("code")):
                return self._json(400, {"error": "that code doesn't match — check your authenticator app"})
            with db():
                db().execute("UPDATE users SET totp_secret=? WHERE id=?", (secret, u["id"]))
            return self._json(200, {"ok": True, "me": user_public(user_by_id(u["id"]))})
        if path == "/api/account/totp/disable":
            if not verify_pw(payload.get("password") or "", u["pw_hash"]):
                return self._json(403, {"error": "password is wrong"})
            with db():
                db().execute("UPDATE users SET totp_secret=NULL WHERE id=?", (u["id"],))
            return self._json(200, {"ok": True, "me": user_public(user_by_id(u["id"]))})

        # ---- web-push subscriptions (per browser, per user)
        if path == "/api/push/subscribe":
            sub = payload.get("subscription")
            if not (isinstance(sub, dict) and str(sub.get("endpoint", "")).startswith("https://")):
                return self._json(400, {"error": "bad subscription"})
            if not push_endpoint_allowed(sub["endpoint"]):
                # SSRF guard: the server later POSTs to this URL, so only real browser push
                # services are accepted — never arbitrary (or internal) hosts
                return self._json(400, {"error": "unrecognized push service"})
            with db():
                db().execute("INSERT INTO push_subs(endpoint,user_id,sub,created) VALUES(?,?,?,?)"
                             " ON CONFLICT(endpoint) DO UPDATE SET user_id=excluded.user_id, sub=excluded.sub",
                             (sub["endpoint"][:2000], u["id"], json.dumps(sub), _now()))
            return self._json(200, {"ok": True})
        if path == "/api/push/unsubscribe":
            with db():
                db().execute("DELETE FROM push_subs WHERE endpoint=? AND user_id=?",
                             (str(payload.get("endpoint") or "")[:2000], u["id"]))
            return self._json(200, {"ok": True})

        if path == "/api/extract":
            name = (payload.get("name") or "file").strip()
            try:
                raw = base64.b64decode(payload.get("data") or "")
            except Exception:
                return self._json(400, {"error": "bad file data"})
            if not raw:
                return self._json(400, {"error": "empty file"})
            if len(raw) > ATTACH_MAX_BYTES:
                return self._json(413, {"error": "file too large (max %d MB)" % (ATTACH_MAX_BYTES // (1024 * 1024))})
            try:
                text = extract_text_file(name, raw)
            except ValueError as e:
                return self._json(400, {"error": str(e)})
            truncated = len(text) > ATTACH_MAX_CHARS
            if truncated:
                text = text[:ATTACH_MAX_CHARS]
            return self._json(200, {"name": name, "text": text, "chars": len(text),
                                    "tokens_est": est_tokens(text), "truncated": truncated})

        ms = re.fullmatch(r"/api/conversations/([^/]+)/stream", path)
        if ms and valid_id(ms.group(1)):
            return self._stream(ms.group(1), payload, u)

        mstop = re.fullmatch(r"/api/conversations/([^/]+)/stop", path)
        if mstop and valid_id(mstop.group(1)):
            if get_convo(mstop.group(1), u) is None:
                return self._json(404, {"error": "not found"})
            request_stop(mstop.group(1))
            return self._json(200, {"ok": True})

        # create or re-snapshot a public share link for this conversation
        mshare = re.fullmatch(r"/api/conversations/([^/]+)/share", path)
        if mshare and valid_id(mshare.group(1)):
            cid = mshare.group(1)
            if get_convo(cid, u) is None:
                return self._json(404, {"error": "not found"})
            sh = upsert_share(cid, u["id"])
            if sh is None:
                return self._json(400, {"error": "nothing to share"})
            return self._json(200, {"shared": True, "token": sh["token"], "url": share_public_url(sh["token"]),
                                    "views": sh["views"], "updated": sh["updated"]})

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
            # tools/think are accepted here, not just in /settings, so a new chat can be created with
            # the whole drawer state in one request — the client carries the current chat's setup
            # forward and would otherwise need a follow-up write to reinstate these two.
            think = payload.get("think")
            mode = "compose" if payload.get("mode") == "compose" else "chat"
            # compose mode is seeded at creation: the whole document is one assistant message that
            # "continue" extends, so the conversation is only worth creating once there is text.
            seed = payload.get("seed") if mode == "compose" else None
            with db():
                db().execute("INSERT INTO conversations(id,owner_id,folder_id,title,system,model,endpoint_id,params,character_id,active_leaf_id,tools,think,mode,created,updated)"
                             " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                             (cid, u["id"], payload.get("folder_id"), payload.get("title", ""),
                              payload.get("system", get_setting("default_system", DEFAULT_SYSTEM)), model,
                              payload.get("endpoint_id") if u["role"] == "admin" else None,
                              json.dumps(payload.get("params") or {}), payload.get("character_id"), None,
                              1 if payload.get("tools") else 0,
                              None if think is None else (1 if think else 0), mode, _now(), _now()))
                if seed is not None:
                    set_leaf(cid, insert_message(cid, None, "assistant", str(seed)))
                    # maybe_title() only ever looks at user messages, and a composition has none
                    if not payload.get("title"):
                        db().execute("UPDATE conversations SET title=? WHERE id=?", (title_from(str(seed)), cid))
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

        m = re.fullmatch(r"/api/characters/([^/]+)/knowledge", path)
        if m:
            if not FTS_OK:
                return self._json(400, {"error": "knowledge needs SQLite FTS5, which this server's build lacks"})
            ch = character_by_id(m.group(1))
            if ch is None or not (ch["owner_id"] == u["id"] or u["role"] == "admin"):
                return self._json(404, {"error": "not found"})
            name = str(payload.get("name") or "file")[:200]
            try:
                raw = ub64(payload.get("data") or "")
            except Exception:
                return self._json(400, {"error": "bad file data"})
            if len(raw) > ATTACH_MAX_BYTES:
                return self._json(413, {"error": "file too large"})
            try:
                text = extract_text_file(name, raw)[:ATTACH_MAX_CHARS]
            except ValueError as e:
                return self._json(400, {"error": str(e)})
            kid = "k-" + uuid.uuid4().hex[:10]
            chunks = _chunk_text(text)
            vecs = []
            if (get_setting("embed_model") or "").strip():
                try:
                    vecs = embed_texts(chunks)
                except Exception as e:
                    log.warning("knowledge embedding failed (doc stays keyword-searchable): %s", e)
            with db():
                db().execute("INSERT INTO knowledge(id,character_id,name,chars,created) VALUES(?,?,?,?,?)",
                             (kid, ch["id"], name, len(text), _now()))
                for i, chunk in enumerate(chunks):
                    db().execute("INSERT INTO knowledge_fts(chunk,character_id,doc_id,name) VALUES(?,?,?,?)",
                                 (chunk, ch["id"], kid, name))
                    if i < len(vecs):
                        db().execute("INSERT INTO knowledge_vec(doc_id,idx,character_id,chunk,vec) VALUES(?,?,?,?,?)",
                                     (kid, i, ch["id"], chunk, json.dumps(vecs[i])))
            return self._json(200, {"ok": True, "knowledge": knowledge_list(ch["id"])})

        if path == "/api/import":
            convs = payload.get("conversations")
            if not isinstance(convs, list) or not convs:
                return self._json(400, {"error": "no conversations to import"})
            made = 0
            for cv in convs[:500]:
                if not isinstance(cv, dict):
                    continue
                msgs = [msg for msg in (cv.get("messages") or [])
                        if isinstance(msg, dict) and msg.get("role") in ("system", "user", "assistant")
                        and isinstance(msg.get("content"), str) and msg["content"].strip()]
                if not msgs:
                    continue
                imp_system = ""
                if msgs[0]["role"] == "system":
                    imp_system = msgs.pop(0)["content"]
                if not msgs:
                    continue
                title = str(cv.get("title") or "").strip()[:120] or title_from(
                    next((msg["content"] for msg in msgs if msg["role"] == "user"), "imported"))
                icid = _cid()
                now = _now()
                c = db()
                with c:
                    c.execute("INSERT INTO conversations(id,owner_id,title,system,model,created,updated)"
                              " VALUES(?,?,?,?,?,?,?)",
                              (icid, u["id"], title, imp_system,
                               get_setting("default_model") or DEFAULT_MODEL, now, now))
                    prev = None
                    for msg in msgs[:4000]:
                        prev = insert_message(icid, prev, msg["role"], msg["content"].strip()[:ATTACH_MAX_CHARS])
                    c.execute("UPDATE conversations SET active_leaf_id=? WHERE id=?", (prev, icid))
                made += 1
            return self._json(200, {"ok": True, "imported": made})

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

        if path == "/api/presets":
            ps = payload.get("preset") or payload
            scope = ps.get("scope", "private")
            if scope == "site" and u["role"] != "admin":
                return self._json(403, {"error": "only admins can create site-wide presets"})
            pid = ps.get("id")
            if pid:
                existing = preset_by_id(pid)
                if existing is None:
                    pid = None
                elif not (existing["owner_id"] == u["id"] or u["role"] == "admin"):
                    return self._json(403, {"error": "not yours to edit"})
            if not pid:
                pid = "p-" + uuid.uuid4().hex[:8]
            params = ps.get("params") or {}
            params = {k: v for k, v in params.items() if k in PARAM_KEYS and v not in (None, "")}
            models = ps.get("models")
            if models is None and ps.get("model"):
                models = [ps["model"]]
            models = [str(m) for m in (models or []) if m]
            model_col = json.dumps(models) if models else ""
            with db():
                db().execute("INSERT INTO presets(id,owner_id,scope,model,name,params,created)"
                             " VALUES(?,?,?,?,?,?,?)"
                             " ON CONFLICT(id) DO UPDATE SET scope=excluded.scope,model=excluded.model,"
                             "name=excluded.name,params=excluded.params",
                             (pid, u["id"], scope, model_col, ps.get("name", "Untitled"),
                              json.dumps(params), _now()))
            return self._json(200, {"presets": visible_presets(u), "id": pid})

        if path == "/api/settings":
            if u["role"] != "admin":
                return self._json(403, {"error": "admin only"})
            if "endpoints" in payload:
                set_setting("endpoints", restore_endpoint_keys(payload["endpoints"]))
            for k in ("active_endpoint", "default_model", "default_system", "default_params",
                      "model_defaults", "user_models", "thinking_models", "search_url",
                      "utility_model", "vision_models", "embed_model"):
                if k in payload:
                    set_setting(k, payload[k])
            return self._json(200, {"settings": admin_settings(), "all_models": fetch_models(active_endpoint(), ttl=0)})

        # connectivity test for an endpoint as edited in the form — nothing is persisted
        if path == "/api/settings/test":
            if u["role"] != "admin":
                return self._json(403, {"error": "admin only"})
            ep = restore_endpoint_keys([payload.get("endpoint") or {}])[0]
            if not ep.get("url"):
                return self._json(200, {"ok": False, "error": "no endpoint url"})
            try:
                ids = [m["id"] for m in _models_data(ep)]
                return self._json(200, {"ok": True, "models": ids})
            except Exception as e:
                return self._json(200, {"ok": False, "error": str(e)})

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

        if path == "/api/invites":
            if u["role"] != "admin":
                return self._json(403, {"error": "admin only"})
            role = "admin" if payload.get("role") == "admin" else "user"
            allowed = payload.get("allowed_models") or None
            max_uses = payload.get("max_uses")
            try:
                max_uses = int(max_uses) if max_uses not in (None, "") else None
            except (ValueError, TypeError):
                max_uses = None
            if max_uses is not None and max_uses < 1:
                max_uses = None
            days = payload.get("days")
            try:
                days = float(days) if days not in (None, "") else None
            except (ValueError, TypeError):
                days = None
            expires = int(time.time() + days * 86400) if (days and days > 0) else None
            note = (payload.get("note") or "").strip()[:200]
            tok = gen_token()
            with db():
                db().execute("INSERT INTO invites(token,created_by,role,allowed_models,max_uses,uses,expires,note,created)"
                             " VALUES(?,?,?,?,?,0,?,?,?)",
                             (tok, u["id"], role, json.dumps(allowed) if allowed else None, max_uses, expires, note, _now()))
            return self._json(200, {"invites": list_invites(), "token": tok})

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
                    # admin reset also revokes the user's existing sessions
                    c.execute("UPDATE users SET pw_hash=?, session_version=session_version+1 WHERE id=?",
                              (hash_pw(payload["password"]), uid))
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
            if "tools" in payload:
                sets.append("tools=?"); args.append(1 if payload["tools"] else 0)
            if "think" in payload:   # None -> model default; else 1/0
                sets.append("think=?"); args.append(None if payload["think"] is None else (1 if payload["think"] else 0))
            if "params" in payload:
                sets.append("params=?"); args.append(json.dumps(payload["params"] or {}))
            if "ctx_summary" in payload:   # user-edited rolling summary; clearing resets the window anchor too
                txt = (payload["ctx_summary"] or "").strip()
                sets.append("ctx_summary=?"); args.append(txt or None)
                if not txt:
                    sets.append("ctx_summary_upto=?"); args.append(None)
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
    @staticmethod
    def _sanitize_attachments(raw):
        out = []
        for a in (raw or [])[:12]:
            if not isinstance(a, dict):
                continue
            if a.get("text"):
                out.append({"name": str(a.get("name") or "file")[:200], "text": a["text"][:ATTACH_MAX_CHARS]})
            elif isinstance(a.get("image"), str) and a["image"].startswith("data:image/") \
                    and ";base64," in a["image"][:64] and len(a["image"]) <= 8 * 1024 * 1024:
                out.append({"name": str(a.get("name") or "image")[:200], "image": a["image"]})
        return out

    def _stream(self, cid, payload, u):
        convo = get_convo(cid, u)
        if convo is None:
            return self._json(404, {"error": "not found"})
        # the per-turn override ("regenerate with…") goes in here, not after: params are per-model
        ep, model, system, params = resolve_request(
            convo, str(payload["model"]) if payload.get("model") else None)
        if not model_allowed(u, model):
            return self._json(403, {"error": "model not permitted"})
        self.close_connection = True   # the ndjson stream has no Content-Length, so it can't be kept alive
        char_row = character_by_id(convo.get("character_id")) if convo.get("character_id") else None
        system = apply_macros(system, {"user": (u["persona"] if "persona" in u.keys() else None) or u["username"],
                                       "char": char_row["name"] if char_row else None, "model": model})
        tools = tools_spec() if convo.get("tools") else None
        vision_on = model in set(get_setting("vision_models") or [])
        clear_stop(cid)   # drop any stale stop flag from a prior run on this conversation

        _, by, _ = _tree(cid)
        content = (payload.get("content") or "").strip()
        attachments = self._sanitize_attachments(payload.get("attachments"))
        regenerate_id = payload.get("regenerate_id")
        edit_user_id = payload.get("edit_user_id")
        continue_id = payload.get("continue_id")
        # branch: keep the message being continued and write the continuation to a *sibling* of it
        # instead of extending it in place, so one prefix can carry several alternative endings and
        # the existing ‹ 1/2 › switcher navigates them.
        cont_branch = bool(payload.get("branch")) and bool(continue_id)
        cont_prefix = None   # set in continue mode: the existing text we resume from

        # mode -> (parent node, ctx sent to model, optional leading user message, whether to title)
        if continue_id:
            # "janky prefill": resume generation from an existing assistant message. The model is
            # fed the partial reply as a trailing assistant turn and continues it; the new tokens are
            # appended in place to the same message. Tools are off — this is plain text continuation.
            tc = by.get(continue_id)
            if tc is None or tc["role"] != "assistant":
                return self._json(400, {"error": "can only continue an assistant message"})
            # NOT the stripped `content`: a continuation resumes at the exact character the text
            # ends on, and stripping the trailing newline would restart mid-paragraph.
            cont_prefix = payload["content"] if payload.get("content") is not None else (tc["content"] or "")
            parent = tc["parent_id"]
            ctx = chain_content(cid, parent, by) + [{"role": "assistant", "content": cont_prefix}]
            lead, title_after = None, False
            tools = None
        elif regenerate_id or payload.get("regenerate"):
            target = regenerate_id or convo.get("active_leaf_id")
            tr = by.get(target)
            if tr is None or tr["role"] != "assistant":
                return self._json(400, {"error": "nothing to regenerate"})
            parent = tr["parent_id"]
            ctx = chain_content(cid, parent, by)
            lead, title_after = None, False
        elif edit_user_id:
            tu = by.get(edit_user_id)
            if tu is None or tu["role"] != "user":
                return self._json(400, {"error": "cannot edit/resend that message"})
            new_content = content or tu["content"]
            # Branching an edited message carries its attachments over. The edit UI sends content
            # only, so without this the pasted text / files / images that the original turn was
            # built around silently vanish from the new branch — the model would answer the same
            # question with the evidence removed. Absent key = inherit; an explicit [] still clears.
            edit_atts = attachments
            if payload.get("attachments") is None and tu["attachments"]:
                edit_atts = json.loads(tu["attachments"])   # stored as TEXT; insert_message re-encodes
            parent = tu["parent_id"]
            ctx = chain_content(cid, parent, by) + [{"role": "user", "content": new_content, "attachments": edit_atts or None}]
            lead, title_after = ("user", new_content, edit_atts), True
        else:
            if not content and not attachments:
                return self._json(400, {"error": "empty message"})
            parent = convo.get("active_leaf_id")
            ctx = chain_content(cid, parent, by) + [{"role": "user", "content": content, "attachments": attachments or None}]
            lead, title_after = ("user", content, attachments), True

        # Finalize the system prompt against the conversation's own tool setting — NOT the effective
        # one (continue mode drops the tools *spec* from the API call, but the system text must stay
        # byte-identical to normal turns or the server's prefix/KV cache misses on the whole prompt).
        # Date stays day-granular so this whole block remains a stable, cacheable prefix within a day.
        system = compose_system(system, bool(convo.get("tools")))

        # Character knowledge (BM25 over uploaded files): retrieved per-turn and appended to the
        # LAST user message in the sent copy — the tail of the prompt — so the cacheable prefix
        # (system + earlier turns) is untouched. The stored chat never contains the injection.
        if char_row and FTS_OK:
            try:
                last_user = next((m for m in reversed(ctx) if m.get("role") == "user"), None)
                if last_user and (last_user.get("content") or "").strip():
                    snips = knowledge_snippets(convo.get("character_id"), last_user["content"])
                    if snips:
                        last_user["content"] += ("\n\n[Reference notes from this character's knowledge "
                                                 "files — use if relevant:\n" + "\n---\n".join(snips) + "\n]")
            except Exception:
                log.exception("knowledge retrieval failed (convo %s)", cid)

        # Context-window management. When the conversation outgrows the model's window we slide it:
        # drop the oldest turns from what we *send* (keeping the system prompt + most recent turns)
        # so there's always room to answer and generation can roll over indefinitely. The stored chat
        # is untouched; a non-fatal notice tells the user what the model can no longer see. max_tokens
        # is a body param only, so none of this affects prefix-cache hits. We act only when the real
        # window is known, so a missing/flaky /v1/models never trims or throttles a big model.
        ctx_notice = None
        sent_est_raw = None   # uncalibrated chars/4 estimate of what we sent (for post-reply learning)
        ctx_win, budget, want, factor, sys_est = 0, 0, MAX_TOKENS, 1.0, 0
        raw_of = lambda msgs: sys_est + sum(_msg_tokens(m) for m in msgs)
        est_of = lambda msgs: round(raw_of(msgs) * factor)

        def retrim(msgs):
            """Drop oldest turns until msgs fits the budget (with hysteresis — see below). Returns
            the number dropped. Never drops the final message; never leaves an orphaned non-user
            turn at the window start."""
            n = 0
            if est_of(msgs) > budget:
                low_water = int(budget * 0.75)
                while len(msgs) > 1 and est_of(msgs) > low_water:
                    msgs.pop(0); n += 1
                while len(msgs) > 1 and msgs[0].get("role") != "user":
                    msgs.pop(0); n += 1
            return n

        try:
            ctx_win = context_for(ep, model, default=0)
            if ctx_win > 0:
                want = int(params.get("max_tokens", MAX_TOKENS) or MAX_TOKENS)
                factor = tok_factor(model)   # learned correction toward this model's real tokenizer
                sys_est = est_tokens(system)
                budget = ctx_win - CTX_REPLY_RESERVE - CTX_MARGIN   # tokens left for the history
                # Trim with hysteresis: once over budget, cut down to ~75% of it in one go. Popping
                # exactly to the budget would shift the window start every turn, changing the prompt
                # prefix each time and defeating the server's prefix/KV cache for the whole rest of
                # the conversation; a deeper cut keeps the prefix stable for many turns between trims.
                pre_trim = list(ctx)
                dropped = 0
                compressed = None
                # Anchored window: once a conversation has overflowed, the previous trim's boundary
                # (ctx_summary_upto = last summarized message) is reused as long as the window after
                # it still fits. This keeps the sent prompt byte-identical at the front across turns
                # (prefix-cache hits) and reuses the stored summary with no extra model call. Only
                # when the window outgrows the budget again does the boundary advance — one fresh
                # trim + one incremental summary fold, then stable again.
                SUM_HDR = "\n\n# Earlier conversation (rolling summary)\n"
                anchor = convo.get("ctx_summary_upto") if COMPRESS else None
                stored_sum = convo.get("ctx_summary") if COMPRESS else None
                if anchor and stored_sum:
                    ids0 = [m.get("id") for m in ctx]
                    if anchor in ids0:
                        cut = ids0.index(anchor) + 1
                        window = ctx[cut:]
                        se = est_tokens(system + SUM_HDR + stored_sum)
                        if window and round((se + sum(_msg_tokens(m) for m in window)) * factor) <= budget:
                            system = system + SUM_HDR + stored_sum
                            sys_est = se
                            del ctx[:cut]
                            dropped, compressed = cut, stored_sum
                if not dropped:
                    dropped = retrim(ctx)
                    if dropped and COMPRESS:
                        # Fold the newly dropped turns into the rolling summary (incremental: the
                        # stored summary is extended, not rebuilt, when the boundary just advanced).
                        try:
                            compressed = ensure_summary(convo, ep, model, pre_trim[:dropped])
                        except Exception:
                            log.exception("context compression failed; dropping plainly (convo %s)", cid)
                        if compressed:
                            system = system + SUM_HDR + compressed
                            sys_est = est_tokens(system)
                            retrim(ctx)   # the summary itself consumes budget; re-check
                sent_est_raw = raw_of(ctx)
                prompt_est = est_of(ctx)
                room = ctx_win - prompt_est - CTX_MARGIN
                params["max_tokens"] = max(64, min(want, room if room > 64 else 64))
                if dropped and compressed:
                    ctx_notice = (
                        "This conversation outgrew {model}'s ~{cw:,}-token context window; the {n} oldest "
                        "message{s} compressed into a rolling summary the model sees instead — your "
                        "saved chat is unchanged.").format(
                            model=model, cw=ctx_win, n=dropped, s=" was" if dropped == 1 else "s were")
                elif dropped:
                    ctx_notice = (
                        "This conversation outgrew {model}'s ~{cw:,}-token context window, so the {n} "
                        "oldest message{s} dropped from what the model sees this turn — your saved chat "
                        "is unchanged.").format(
                            model=model, cw=ctx_win, n=dropped, s=" was" if dropped == 1 else "s were")
                elif room < CTX_REPLY_RESERVE:   # a single huge final turn we can't trim around
                    ctx_notice = (
                        "This message nearly fills {model}'s ~{cw:,}-token context window; the reply may "
                        "be cut short or fail.").format(model=model, cw=ctx_win)
        except Exception:
            log.exception("context-window management failed; sending untrimmed (convo %s)", cid)

        def persist(collected, reply, reasoning, meta, allow_empty=True):
            with db():
                if continue_id:
                    row = by.get(continue_id)
                    new_content = (cont_prefix or "") + (reply or "")
                    old_reason = (row["reasoning"] if row else "") or ""
                    merged_reason = (old_reason + reasoning) if reasoning else (old_reason or None)
                    if cont_branch:
                        # sibling: a second ending for the same prefix, left alongside the original
                        set_leaf(cid, insert_message(cid, parent, "assistant", new_content,
                                                     merged_reason, model, meta))
                    else:
                        # append the continuation to the existing assistant message, in place
                        db().execute("UPDATE messages SET content=?, reasoning=?, model=?, meta=?, edited=? WHERE id=? AND convo_id=?",
                                     (new_content, merged_reason, model,
                                      json.dumps(meta) if meta else (row["meta"] if row else None), _now(), continue_id, cid))
                        _, _, kids2 = _tree(cid)
                        set_leaf(cid, _default_leaf(continue_id, kids2))
                    touch_convo(cid)
                    return
                node = parent
                if lead:
                    node = insert_message(cid, node, "user", lead[1], attachments=lead[2] or None)
                for item in collected:
                    node = insert_message(cid, node, item["role"], item["content"],
                                          reasoning=item.get("reasoning"),
                                          model=model if item["role"] == "assistant" else None,
                                          tool=item.get("tool"))
                if reply or reasoning or (allow_empty and not collected):
                    node = insert_message(cid, node, "assistant", reply, reasoning, model, meta)
                set_leaf(cid, node)
                touch_convo(cid)
                if title_after:
                    maybe_title(cid)

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        gone = [False]

        def emit(obj):
            if gone[0]:
                return
            try:
                self.wfile.write((json.dumps(obj) + "\n").encode("utf-8"))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                gone[0] = True

        if ctx_notice:
            emit({"notice": ctx_notice})

        work = list(ctx)
        collected = []
        reply, reasoning, usage = "", "", None
        t0 = time.time(); t_first = [0.0]
        # Per-request body extras (dropped by stream_model's fallback if a server rejects them).
        # Everything here rides in ONE chat_template_kwargs dict because that is the only channel
        # the server feeds to apply_chat_template. The continuation hints used to be sent as
        # top-level body fields, where they were silently ignored: the trailing assistant message
        # was then rendered as a *finished* turn with a fresh generation prompt after it, so
        # "continue" glued a brand-new reply (with its own thinking block) onto the old text
        # instead of extending it. Setting them as two keys on separate dicts is equally broken —
        # the second assignment would drop the first, which is why they are merged, not stacked.
        ctk = {}
        if continue_id:
            ctk.update({"add_generation_prompt": False, "continue_final_message": True})
        if model in set(get_setting("thinking_models", []) or []) and convo.get("think") is not None:
            # tri-state: NULL = model default (send nothing); 1/0 = explicit per-chat on/off
            ctk["enable_thinking"] = convo.get("think") == 1
        cont_extra = {"chat_template_kwargs": ctk} if ctk else None

        seg_state = {}   # live buffers of the current run_stream pass, for partial-persist on error

        def run_stream(cur_tools, cur_choice="auto"):
            # Stream a turn. Leading whitespace is held back so a *dropped* tool call (which the oMLX
            # server emits as a lone "\n") never spawns an empty bubble — the thinking dots stay up.
            parts, rparts, tcalls, used, finish = [], [], None, None, None
            seg_state["parts"], seg_state["rparts"] = parts, rparts
            flushed = [False]

            def push(d):
                parts.append(d)
                if flushed[0]:
                    if not t_first[0]:
                        t_first[0] = time.time()
                    emit({"delta": d})
                elif "".join(parts).strip():
                    flushed[0] = True
                    if not t_first[0]:
                        t_first[0] = time.time()
                    emit({"delta": "".join(parts)})

            for ev in stream_model(ep, model, system, work, params, tools=cur_tools, extra=cont_extra,
                                   tool_choice=cur_choice, vision=vision_on):
                if "delta" in ev:
                    push(ev["delta"])
                elif "reasoning" in ev:
                    if not t_first[0]:
                        t_first[0] = time.time()
                    rparts.append(ev["reasoning"]); emit({"reasoning": ev["reasoning"]})
                elif "usage" in ev:
                    used = ev["usage"]
                elif "tool_calls" in ev:
                    tcalls = ev["tool_calls"]
                elif "finish" in ev:
                    finish = ev["finish"]
                if gone[0] or stop_requested(cid):
                    break
            return "".join(parts), "".join(rparts), tcalls, used, finish

        # the user's most recent link — recovered into a tool call when the model omits the url
        fallback_url = None
        for m in reversed(work):
            if m.get("role") == "user":
                found = _URL_RE.findall(m.get("content") or "")
                if found:
                    fallback_url = found[-1]
                    break

        seen_calls = set()
        force_answer = False
        try:
            for it in range(TOOL_MAX_ITERS + 1):
                # Final round (or a repeated identical call): force a text answer via tool_choice
                # "none" while KEEPING the tools schema in the request — the schema is rendered into
                # the templated prompt, so dropping it would invalidate the KV cache built by this
                # very turn's earlier rounds. stream_model falls back to a no-tools request for
                # servers that reject a non-auto tool_choice.
                answer_only = (it >= TOOL_MAX_ITERS or force_answer)
                cur_tools = tools
                cur_choice = "none" if answer_only else "auto"
                if cur_tools:
                    emit({"status": "reading the results…" if collected else "thinking…"})
                if it and ctx_win > 0:
                    # fetched tool results grew the prompt beyond what the pre-turn budget saw;
                    # re-trim and re-clamp max_tokens so this pass can't overflow the window
                    retrim(work)
                    room2 = ctx_win - est_of(work) - CTX_MARGIN
                    params["max_tokens"] = max(64, min(want, room2 if room2 > 64 else 64))
                # Always stream first — a real text answer streams cleanly token-by-token.
                seg_reply, seg_reason, tcalls, used, finish = run_stream(cur_tools, cur_choice)
                if used:
                    usage = used
                if cur_tools and not answer_only and not tcalls:
                    if seg_reply and "<tool_call>" in seg_reply:           # tool call leaked as text
                        parsed = parse_text_tool_calls(seg_reply)
                        if parsed:
                            tcalls = parsed; seg_reply = strip_tool_calls(seg_reply)
                    # the streaming tool parser dropped the call -> recover via reliable non-streaming
                    if not tcalls and not gone[0] and not stop_requested(cid) and (finish == "tool_calls" or not seg_reply.strip()):
                        try:
                            c2, r2, t2, u2, _ = call_model_nonstream(ep, model, system, work, params, cur_tools,
                                                                     vision=vision_on)
                        except Exception:
                            c2, r2, t2, u2 = "", "", None, None
                        if u2:
                            usage = u2
                        if t2:
                            tcalls = t2; seg_reply = strip_tool_calls(c2); seg_reason = seg_reason or r2
                        elif c2 and "<tool_call>" in c2 and parse_text_tool_calls(c2):
                            tcalls = parse_text_tool_calls(c2); seg_reply = strip_tool_calls(c2); seg_reason = seg_reason or r2
                        elif c2:                                            # actually a text answer; show it
                            seg_reply, seg_reason = c2, (seg_reason or r2)
                            if not t_first[0]:
                                t_first[0] = time.time()
                            emit({"delta": c2})
                if gone[0] or stop_requested(cid) or not tcalls or answer_only:
                    reply, reasoning = seg_reply, seg_reason
                    break
                emit({"tool_turn": True})
                collected.append({"role": "assistant", "content": seg_reply, "reasoning": seg_reason,
                                  "tool": {"tool_calls": tcalls}})
                work.append({"role": "assistant", "content": seg_reply, "tool": {"tool_calls": tcalls}})
                new_call = False
                for tc in tcalls:
                    fn = tc.get("function") or {}
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except ValueError:
                        args = {}
                    if not isinstance(args, dict):
                        args = {}
                    sig = (fn.get("name"), json.dumps(args, sort_keys=True))
                    if sig not in seen_calls:
                        new_call = True
                    seen_calls.add(sig)
                    emit({"tool_call": {"name": fn.get("name"), "args": args, "id": tc.get("id")}})
                    result, ui = execute_tool(fn.get("name"), args, fallback_url=fallback_url)
                    emit({"tool_result": dict(ui, name=fn.get("name"), id=tc.get("id"))})
                    collected.append({"role": "tool", "content": result,
                                      "tool": {"tool_call_id": tc.get("id"), "name": fn.get("name"), "ui": ui}})
                    work.append({"role": "tool", "content": result, "tool": {"tool_call_id": tc.get("id")}})
                if not new_call:
                    force_answer = True  # model is repeating the same call — make the next pass answer
        except Exception as e:
            # Persist what made it through before failing, so the user's message and any partial
            # reply survive a mid-stream error instead of evaporating on reload.
            log.exception("stream failed (convo %s)", cid)
            partial = strip_tool_calls("".join(seg_state.get("parts") or []))
            partial_r = "".join(seg_state.get("rparts") or [])
            persisted = False
            if partial or partial_r or collected or lead:
                try:
                    meta = (build_meta(t0, t_first[0] or None, time.time(), usage, partial)
                            if (partial or partial_r) else None)
                    persist(collected, partial, partial_r, meta, allow_empty=False)
                    persisted = True
                except Exception:
                    log.exception("failed to persist partial reply (convo %s)", cid)
            emit({"error": "model error: " + str(e)})
            if persisted and not gone[0]:
                try:
                    emit({"done": True, "convo": get_convo(cid, u)})
                except Exception:
                    pass
            return

        # never leak raw <tool_call> syntax into a final answer; if tools are off but the model
        # tried to call one, replace the dead-end with an actionable hint.
        if reply and "<tool_call>" in reply:
            attempted = parse_text_tool_calls(reply)
            reply = strip_tool_calls(reply)
            if not tools and attempted:
                hint = "_(I tried to use a web tool, but web tools are off for this chat — turn on **web tools** in *tune* to let me fetch links.)_"
                reply = (reply + "\n\n" + hint).strip() if reply else hint

        if not reply and not reasoning and not collected:
            if gone[0]:
                return
            if stop_requested(cid):       # stopped before anything streamed — resolve cleanly
                try:
                    emit({"done": True, "convo": get_convo(cid, u)})
                except Exception:
                    pass
                return
            emit({"error": "the model returned an empty response — please try again"})
            if lead:   # still keep the user's turn so it isn't lost on reload
                try:
                    persist(collected, "", "", None, allow_empty=False)
                    emit({"done": True, "convo": get_convo(cid, u)})
                except Exception:
                    log.exception("failed to persist user turn after empty response (convo %s)", cid)
            return
        meta = build_meta(t0, t_first[0] or None, time.time(), usage, reply)
        push_notify(u["id"], convo.get("title") or "reply finished", reply, cid)
        meta["params"] = {k: params[k] for k in params if k in PARAM_KEYS}   # what was actually sent
        pn = matching_preset_name(u, model, convo.get("params") or {})
        if pn:
            meta["preset"] = pn
        # Learn this model's real tokens-per-estimate from the server's reported prompt size. Only on
        # clean text turns (no tool steps, which inflate the real prompt beyond what we measured).
        if usage and usage.get("prompt_tokens") and not collected and sent_est_raw:
            update_tok_factor(model, usage["prompt_tokens"], sent_est_raw)
        persist(collected, reply, reasoning, meta)

        # On the very first exchange, summarize a short title (replacing the first-line fallback).
        # Runs in a background thread AFTER `done` is emitted: the extra title-gen model call
        # (seconds on a warm model, much longer on a cold one) must not delay the finished reply.
        # The client re-fetches the title shortly after `done`. Skipped when the user stopped.
        if lead and parent is None and not gone[0] and not stop_requested(cid):
            utext = lead[1]
            atts = lead[2] or []
            if atts and atts[0].get("text"):
                utext = (utext + "\n\n" + atts[0]["text"]) if utext else atts[0]["text"]

            def _title_bg(utext=utext, reply=reply):
                title = generate_title(ep, model, utext, reply)
                if title:
                    with db():
                        db().execute("UPDATE conversations SET title=? WHERE id=?", (title, cid))
            threading.Thread(target=_title_bg, daemon=True).start()

        if not gone[0]:
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

# ============================================================ branding / icons
# A procedurally-generated alchemical sigil: a {7/3} heptagram (seven points for
# the seven models on the endpoint) ringed around an all-seeing, serpent-slit
# oracle eye. Gold-on-black to match the ORACLE theme. The favicon ships as crisp
# vector SVG; the Open Graph card (1200x630) and Apple touch icon (180x180) are
# rendered to PNG in pure stdlib (zlib + struct) — no image libs, no build step —
# computed once on first request and cached.

PUBLIC_URL = os.environ.get("KENOSIS_PUBLIC_URL", "https://delphi.disinfo.zone").rstrip("/")

FAVICON_SVG = r"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="64" height="64">
<defs>
<radialGradient id="disc" cx="50%" cy="46%" r="62%">
<stop offset="0%" stop-color="#241a0f"/><stop offset="60%" stop-color="#140f09"/><stop offset="100%" stop-color="#0b0805"/>
</radialGradient>
<linearGradient id="gold" x1="0" y1="0" x2="0" y2="1">
<stop offset="0%" stop-color="#f7e3a1"/><stop offset="50%" stop-color="#d9a64c"/><stop offset="100%" stop-color="#a9772a"/>
</linearGradient>
<radialGradient id="eyeglow" cx="50%" cy="50%" r="60%">
<stop offset="0%" stop-color="#fff6dd"/><stop offset="55%" stop-color="#f3cd84"/><stop offset="100%" stop-color="#d9a84e"/>
</radialGradient>
<radialGradient id="iris" cx="50%" cy="50%" r="58%">
<stop offset="0%" stop-color="#e9b65e"/><stop offset="60%" stop-color="#b07e30"/><stop offset="100%" stop-color="#5e3f16"/>
</radialGradient>
</defs>
<circle cx="32" cy="32" r="31" fill="url(#disc)"/>
<circle cx="32" cy="32" r="30" fill="none" stroke="url(#gold)" stroke-width="1" opacity="0.5"/>
<circle cx="32" cy="32" r="27.5" fill="none" stroke="#a9772a" stroke-width="0.6" stroke-dasharray="0.6 3" opacity="0.6"/>
<g fill="none" stroke-linejoin="round">
<path d="M32 8 L42.41 53.62 L13.24 17.04 L55.40 37.34 L8.60 37.34 L50.76 17.04 L21.59 53.62 Z" stroke="#e9b85e" stroke-width="3" opacity="0.16"/>
<path d="M32 8 L42.41 53.62 L13.24 17.04 L55.40 37.34 L8.60 37.34 L50.76 17.04 L21.59 53.62 Z" stroke="url(#gold)" stroke-width="1.4"/>
</g>
<g fill="#f7e3a1">
<circle cx="32" cy="8" r="1.4"/><circle cx="50.76" cy="17.04" r="1.4"/><circle cx="55.40" cy="37.34" r="1.4"/><circle cx="42.41" cy="53.62" r="1.4"/><circle cx="21.59" cy="53.62" r="1.4"/><circle cx="8.60" cy="37.34" r="1.4"/><circle cx="13.24" cy="17.04" r="1.4"/>
</g>
<g>
<path d="M20.5 32 Q32 23 43.5 32 Q32 41 20.5 32 Z" fill="url(#eyeglow)" stroke="url(#gold)" stroke-width="0.8"/>
<circle cx="32" cy="32" r="4" fill="url(#iris)"/>
<ellipse cx="32" cy="32" rx="1" ry="3.3" fill="#0c0905"/>
<circle cx="30.6" cy="30.6" r="0.8" fill="#fffaf0" opacity="0.85"/>
</g>
</svg>"""

_OG_DESC = "A private oracle - authenticated chat with local language models. Pneuma, kenosis, prophecy. Entry by invitation only."
_OG_ALT = "A glowing golden seven-pointed star with a serpent-eyed oracle at its center."
META_TAGS = (
    '<meta name="description" content="' + _OG_DESC + '">' +
    '<meta name="keywords" content="ORACLE, pneuma, kenosis, Delphi, oracle, LLM chat, local AI, disinfo.zone">' +
    '<meta name="author" content="disinfo.zone">' +
    '<meta name="theme-color" content="#100c08">' +
    '<meta name="color-scheme" content="dark light">' +
    '<meta name="robots" content="noindex, nofollow">' +
    '<link rel="canonical" href="' + PUBLIC_URL + '/">' +
    '<link rel="icon" type="image/svg+xml" href="/favicon.svg">' +
    '<link rel="alternate icon" type="image/png" sizes="180x180" href="/apple-touch-icon.png">' +
    '<link rel="apple-touch-icon" href="/apple-touch-icon.png">' +
    '<link rel="manifest" href="/manifest.webmanifest">' +
    '<meta property="og:type" content="website">' +
    '<meta property="og:site_name" content="ORACLE">' +
    '<meta property="og:title" content="ORACLE">' +
    '<meta property="og:description" content="' + _OG_DESC + '">' +
    '<meta property="og:url" content="' + PUBLIC_URL + '/">' +
    '<meta property="og:image" content="' + PUBLIC_URL + '/og-image.png">' +
    '<meta property="og:image:secure_url" content="' + PUBLIC_URL + '/og-image.png">' +
    '<meta property="og:image:type" content="image/png">' +
    '<meta property="og:image:width" content="1200">' +
    '<meta property="og:image:height" content="630">' +
    '<meta property="og:image:alt" content="' + _OG_ALT + '">' +
    '<meta name="twitter:card" content="summary_large_image">' +
    '<meta name="twitter:title" content="ORACLE">' +
    '<meta name="twitter:description" content="' + _OG_DESC + '">' +
    '<meta name="twitter:image" content="' + PUBLIC_URL + '/og-image.png">' +
    '<meta name="twitter:image:alt" content="' + _OG_ALT + '">'
)

# ---------------------------------------------------------------- PWA (installable app shell)
MANIFEST_JSON = json.dumps({
    "name": "ORACLE", "short_name": "ORACLE", "start_url": "/", "scope": "/",
    "display": "standalone", "background_color": "#100c08", "theme_color": "#100c08",
    "icons": [{"src": "/favicon.svg", "sizes": "any", "type": "image/svg+xml", "purpose": "any"},
              {"src": "/apple-touch-icon.png", "sizes": "180x180", "type": "image/png"}],
})

# Network-first with cache fallback for the shell and icons; API, auth, and share pages are never
# touched so nothing dynamic or private is served stale.
SW_JS = r"""
const C="oracle-shell-v1";
self.addEventListener("install",e=>self.skipWaiting());
self.addEventListener("activate",e=>{e.waitUntil(caches.keys()
  .then(ks=>Promise.all(ks.filter(k=>k!==C).map(k=>caches.delete(k)))).then(()=>self.clients.claim()));});
self.addEventListener("fetch",e=>{
  const u=new URL(e.request.url);
  if(e.request.method!=="GET"||u.origin!==location.origin)return;
  const p=u.pathname;
  if(p.startsWith("/api/")||p.startsWith("/s/")||p==="/login"||p==="/setup"||p.startsWith("/invite"))return;
  e.respondWith(fetch(e.request).then(r=>{
    if(r.ok){const cp=r.clone();caches.open(C).then(c=>c.put(e.request,cp));}
    return r;
  }).catch(()=>caches.match(e.request).then(m=>m||Response.error())));
});
self.addEventListener("push",e=>{
  let d={};try{d=e.data?e.data.json():{};}catch(_){}
  e.waitUntil((async()=>{
    // suppress the banner when the app is already visible — the in-page toast covers that case
    const cs=await self.clients.matchAll({type:"window",includeUncontrolled:true});
    if(cs.some(c=>c.visibilityState==="visible"))return;
    await self.registration.showNotification(d.title||"ORACLE",{body:d.body||"reply finished",tag:"oracle-push",data:d});
  })());
});
self.addEventListener("notificationclick",e=>{
  e.notification.close();
  e.waitUntil((async()=>{
    const cs=await self.clients.matchAll({type:"window",includeUncontrolled:true});
    if(cs.length){cs[0].focus();return;}
    await self.clients.openWindow("/");
  })());
});
"""

_BRAND_LOCK = threading.Lock()
_BRAND_CACHE = {}


def _sigil_png(W, H, R, eye_w, eye_h, iris_r, slit_rx, slit_ry, point_r,
               ring1, ring2, line_half, glow_r, stars):
    """Render the ORACLE sigil to PNG bytes using only the standard library."""
    import math, zlib, struct
    cx, cy = W / 2, H / 2
    BG_TOP, BG_BOT = (20, 15, 9), (10, 8, 5)
    GLOW = (70, 48, 18)
    GOLD, GOLD_HI, GOLD_LO = (233, 184, 94), (247, 227, 161), (169, 119, 42)
    EYE_PALE, EYE_MID, EYE_EDGE = (255, 246, 221), (243, 205, 132), (217, 168, 78)
    IRIS_HI, IRIS_MID, IRIS_LO = (233, 182, 94), (176, 126, 48), (94, 63, 22)
    PUPIL, HILITE, STAR = (12, 9, 5), (255, 250, 240), (200, 170, 110)
    buf = bytearray(W * H * 3)

    def lerp(a, b, t):
        return (a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t, a[2]+(b[2]-a[2])*t)

    def blend(x, y, c, a):
        if a <= 0 or x < 0 or y < 0 or x >= W or y >= H:
            return
        i = (int(y) * W + int(x)) * 3
        if a >= 1:
            buf[i], buf[i+1], buf[i+2] = int(c[0]), int(c[1]), int(c[2])
        else:
            ia = 1 - a
            buf[i]   = int(buf[i]   * ia + c[0] * a)
            buf[i+1] = int(buf[i+1] * ia + c[1] * a)
            buf[i+2] = int(buf[i+2] * ia + c[2] * a)

    def addlight(x, y, c, f):
        if f <= 0 or x < 0 or y < 0 or x >= W or y >= H:
            return
        i = (int(y) * W + int(x)) * 3
        buf[i]   = min(255, int(buf[i]   + c[0] * f))
        buf[i+1] = min(255, int(buf[i+1] + c[1] * f))
        buf[i+2] = min(255, int(buf[i+2] + c[2] * f))

    # background: vertical gradient + radial amber glow + corner vignette
    maxd = math.hypot(W, H) / 2
    glow_rad = R * 1.45
    for y in range(H):
        base = lerp(BG_TOP, BG_BOT, y / (H - 1))
        dy2 = (y - cy) ** 2
        row = y * W * 3
        for x in range(W):
            d = math.sqrt((x - cx) ** 2 + dy2)
            g = max(0.0, 1 - d / glow_rad); g = g * g * 0.9
            vig = 1 - 0.35 * (d / maxd) ** 2
            i = row + x * 3
            buf[i]   = min(255, int(base[0] * vig + GLOW[0] * g))
            buf[i+1] = min(255, int(base[1] * vig + GLOW[1] * g))
            buf[i+2] = min(255, int(base[2] * vig + GLOW[2] * g))

    # faint starfield (deterministic LCG), kept clear of the central disc
    if stars:
        seed = 1337
        for _ in range(70):
            seed = (1103515245 * seed + 12345) & 0x7fffffff; sx = seed % W
            seed = (1103515245 * seed + 12345) & 0x7fffffff; sy = seed % H
            seed = (1103515245 * seed + 12345) & 0x7fffffff
            br = 0.12 + (seed % 100) / 100 * 0.33
            if math.hypot(sx - cx, sy - cy) < R * 1.05:
                continue
            addlight(sx, sy, STAR, br)
            addlight(sx + 1, sy, STAR, br * 0.4)
            addlight(sx, sy + 1, STAR, br * 0.4)

    # heptagram {7/3}
    pts = []
    for k in range(7):
        ang = math.radians(-90 + k * (360 / 7))
        pts.append((cx + R * math.cos(ang), cy + R * math.sin(ang)))
    order = [0, 3, 6, 2, 5, 1, 4]
    segs = [(pts[order[i]], pts[order[(i + 1) % 7]]) for i in range(7)]

    def line(p0, p1, half, col, glowf=0.0):
        x0, y0 = p0; x1, y1 = p1
        ext = half + (glow_r if glowf else 1) + 1
        minx = max(0, int(min(x0, x1) - ext)); maxx = min(W - 1, int(max(x0, x1) + ext))
        miny = max(0, int(min(y0, y1) - ext)); maxy = min(H - 1, int(max(y0, y1) + ext))
        dx = x1 - x0; dy = y1 - y0; L2 = dx*dx + dy*dy or 1
        for y in range(miny, maxy + 1):
            for x in range(minx, maxx + 1):
                t = ((x - x0) * dx + (y - y0) * dy) / L2
                t = 0.0 if t < 0 else (1.0 if t > 1 else t)
                d = math.hypot(x - (x0 + t*dx), y - (y0 + t*dy))
                if glowf:
                    gg = max(0.0, 1 - d / glow_r)
                    if gg > 0:
                        addlight(x, y, col, gg * gg * glowf)
                a = half - d + 0.5
                if a > 0:
                    blend(x, y, col, min(1.0, a))

    for p0, p1 in segs:
        line(p0, p1, line_half, GOLD, glowf=0.5)
    for p0, p1 in segs:
        line(p0, p1, line_half, GOLD)

    # talisman rings
    def ring(r, hw, col, a_mul=1.0):
        ext = hw + 2
        for y in range(max(0, int(cy - r - ext)), min(H, int(cy + r + ext))):
            dy2 = (y - cy) ** 2
            for x in range(max(0, int(cx - r - ext)), min(W, int(cx + r + ext))):
                a = hw - abs(math.sqrt((x - cx) ** 2 + dy2) - r) + 0.5
                if a > 0:
                    blend(x, y, col, min(1.0, a) * a_mul)

    def dashed_ring(r, hw, col, dashes, duty, a_mul):
        for j in range(dashes):
            a0 = (j / dashes) * 2 * math.pi
            a1 = a0 + (duty / dashes) * 2 * math.pi
            for s in range(9):
                ang = a0 + (a1 - a0) * s / 8
                px = cx + r * math.cos(ang); py = cy + r * math.sin(ang)
                for oy in (-1, 0, 1):
                    for ox in (-1, 0, 1):
                        aa = hw - math.hypot(ox, oy) + 0.4
                        if aa > 0:
                            blend(int(px) + ox, int(py) + oy, col, min(1.0, aa) * a_mul)

    ring(ring1, 1.4, GOLD, 0.6)
    dashed_ring(ring2, 1.0, GOLD_LO, 84, 0.5, 0.6)

    # seven points
    def disc(px, py, r, col, glowf=0.0, gr=14.0):
        ext = r + (gr if glowf else 1) + 1
        for y in range(max(0, int(py - ext)), min(H, int(py + ext + 1))):
            for x in range(max(0, int(px - ext)), min(W, int(px + ext + 1))):
                d = math.hypot(x - px, y - py)
                if glowf:
                    gg = max(0.0, 1 - d / gr)
                    if gg > 0:
                        addlight(x, y, col, gg * gg * glowf)
                a = r - d + 0.5
                if a > 0:
                    blend(x, y, col, min(1.0, a))

    for (px, py) in pts:
        disc(px, py, point_r, GOLD_HI, glowf=0.6, gr=16)

    # the eye: parabolic-lid lens
    def lid(x):
        u = (x - cx) / eye_w
        return -1 if abs(u) >= 1 else eye_h * (1 - u * u)

    for y in range(max(0, int(cy - eye_h - 2)), min(H, int(cy + eye_h + 3))):
        for x in range(max(0, int(cx - eye_w - 2)), min(W, int(cx + eye_w + 3))):
            l = lid(x)
            if l < 0:
                continue
            cover = l - abs(y - cy) + 0.5
            if cover <= 0:
                continue
            rr = math.hypot((x - cx) / eye_w, (y - cy) / eye_h)
            col = lerp(EYE_PALE, EYE_MID, rr / 0.5) if rr < 0.5 else lerp(EYE_MID, EYE_EDGE, min(1.0, (rr - 0.5) / 0.5))
            blend(x, y, col, min(1.0, cover))

    # gold rim along both lids
    rim_steps = int(eye_w * 2)
    for s in range(rim_steps + 1):
        x = cx - eye_w + (2 * eye_w) * s / rim_steps
        l = lid(x)
        if l < 0:
            continue
        for sign in (1, -1):
            yy = cy + sign * l
            for oy in (-1, 0, 1):
                a = 1.3 - abs(oy)
                if a > 0:
                    blend(int(x), int(yy) + oy, GOLD, min(1.0, a) * 0.9)

    # iris
    for y in range(int(cy - iris_r - 1), int(cy + iris_r + 2)):
        for x in range(int(cx - iris_r - 1), int(cx + iris_r + 2)):
            d = math.hypot(x - cx, y - cy)
            a = iris_r - d + 0.5
            if a <= 0:
                continue
            t = d / iris_r
            col = lerp(IRIS_HI, IRIS_MID, t / 0.6) if t < 0.6 else lerp(IRIS_MID, IRIS_LO, (t - 0.6) / 0.4)
            blend(x, y, col, min(1.0, a))

    # serpent slit pupil
    for y in range(int(cy - slit_ry - 1), int(cy + slit_ry + 2)):
        for x in range(int(cx - slit_rx - 1), int(cx + slit_rx + 2)):
            rr = math.hypot((x - cx) / slit_rx, (y - cy) / slit_ry)
            a = (1 - rr) * slit_rx + 0.5
            if a > 0:
                blend(x, y, PUPIL, min(1.0, a))

    disc(cx - iris_r * 0.35, cy - iris_r * 0.35, max(2, iris_r * 0.14), HILITE)

    # encode PNG (RGB, no filtering)
    def chunk(typ, data):
        return (struct.pack(">I", len(data)) + typ + data +
                struct.pack(">I", zlib.crc32(typ + data) & 0xffffffff))
    raw = bytearray(); stride = W * 3
    for y in range(H):
        raw.append(0); raw += buf[y * stride:(y + 1) * stride]
    return (b"\x89PNG\r\n\x1a\n" +
            chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(bytes(raw), 9)) +
            chunk(b"IEND", b""))


def og_image_png():
    with _BRAND_LOCK:
        if "og" not in _BRAND_CACHE:
            _BRAND_CACHE["og"] = _sigil_png(1200, 630, 250, 150, 60, 44, 10, 34, 6, 282, 262, 3.4, 15.0, True)
    return _BRAND_CACHE["og"]


def apple_icon_png():
    with _BRAND_LOCK:
        if "apple" not in _BRAND_CACHE:
            _BRAND_CACHE["apple"] = _sigil_png(180, 180, 74, 46, 19, 13.5, 3.2, 10.5, 2.2, 85, 79, 1.3, 5.0, False)
    return _BRAND_CACHE["apple"]


LOGIN_PAGE = (r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>oracle · sign in</title>""" + META_TAGS + r"""
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Newsreader:opsz,wght@6..72,400;6..72,500&display=swap" rel="stylesheet">
<style>""" + AUTH_CSS + r"""</style></head><body>
<form class="card" id="f">
  <div class="brand">ORACLE</div>
  <h1>Sign in</h1>
  <p class="lede">This instance is private. Enter your credentials to continue.</p>
  <label>Username</label><input id="u" autocomplete="username" autofocus>
  <label>Password</label><input id="p" type="password" autocomplete="current-password">
  <div id="totp_row" style="display:none;"><label>Authentication code</label>
    <input id="t" inputmode="numeric" autocomplete="one-time-code" placeholder="6-digit code"></div>
  <button type="submit">Enter</button>
  <div class="err" id="e"></div>
</form>
<script>
document.getElementById("f").onsubmit=async(ev)=>{ev.preventDefault();const e=document.getElementById("e");e.textContent="";
 try{const body={username:document.getElementById("u").value,password:document.getElementById("p").value};
  const t=document.getElementById("t").value.trim();if(t)body.totp=t;
  const r=await fetch("/api/login",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
  const j=await r.json().catch(()=>({}));
  if(j.totp_required&&!j.ok){document.getElementById("totp_row").style.display="block";
    document.getElementById("t").focus();if(j.error)e.textContent=j.error;return;}
  if(!r.ok)throw new Error(j.error||"sign in failed");location.href="/";
 }catch(err){e.textContent=err.message;}};
</script></body></html>""")

SETUP_PAGE = (r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>oracle · setup</title>""" + META_TAGS + r"""
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

INVITE_PAGE = (r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>oracle · register</title>""" + META_TAGS + r"""
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Newsreader:opsz,wght@6..72,400;6..72,500&display=swap" rel="stylesheet">
<style>""" + AUTH_CSS + r"""</style></head><body>
<form class="card" id="f">
  <div class="brand">ORACLE <span class="sub">// invite</span></div>
  <h1>Create your account</h1>
  <p class="lede" id="lede">You have been invited to this private instance. Choose a username and password.</p>
  <label>Username</label><input id="u" autocomplete="username" autofocus>
  <label>Password <span style="text-transform:none;letter-spacing:0;color:var(--faint)">(min 8)</span></label><input id="p" type="password" autocomplete="new-password">
  <label>Confirm password</label><input id="p2" type="password" autocomplete="new-password">
  <button type="submit" id="go">Create &amp; enter</button>
  <div class="err" id="e"></div>
</form>
<script>
var TOKEN=decodeURIComponent((location.pathname.split("/").filter(Boolean).pop())||"");
(async function(){try{const r=await fetch("/api/invite/"+encodeURIComponent(TOKEN));const j=await r.json().catch(()=>({}));
  if(!j.valid){document.getElementById("f").innerHTML='<div class="brand">ORACLE</div><h1>Invite unavailable</h1><p class="lede">'+(j.error||"This invite link is not valid.")+'</p><a href="/login" style="color:var(--accent)">Go to sign in &rarr;</a>';return;}
  if(j.role==="admin")document.getElementById("lede").textContent="You have been invited as an administrator. Choose a username and password.";
 }catch(_){}})();
document.getElementById("f").onsubmit=async(ev)=>{ev.preventDefault();const e=document.getElementById("e");e.textContent="";
 const p=document.getElementById("p").value,p2=document.getElementById("p2").value;
 if(p!==p2){e.textContent="passwords do not match";return;}
 try{const r=await fetch("/api/invite/register",{method:"POST",headers:{"Content-Type":"application/json"},
  body:JSON.stringify({token:TOKEN,username:document.getElementById("u").value,password:p})});
  const j=await r.json().catch(()=>({}));if(!r.ok)throw new Error(j.error||"registration failed");location.href="/";
 }catch(err){e.textContent=err.message;}};
</script></body></html>""")

PAGE_HEAD = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>oracle</title>""" + META_TAGS + r"""
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&display=swap" rel="stylesheet">
<script>
(function(){try{var d=document.documentElement,L=localStorage;
 d.setAttribute('data-theme',L.getItem('oracle_theme')||'dark');
 var pa=L.getItem('oracle_palette');
 if(pa==='custom'){try{var cv=JSON.parse(L.getItem('oracle_custom_vars')||'{}');for(var ck in cv)d.style.setProperty(ck,cv[ck]);}catch(e){}}
 else if(pa&&pa!=='sepia')d.setAttribute('data-palette',pa);
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
    /* composer control height = input-shell padding (24px) + one line of text; tracks the text-size slider so attach/send always match the field */
    --ctrl-h:calc(24px + 24.75px * var(--rs));
    --bg:#100c08; --panel:#15100a; --surface:#1b150d; --surface2:#231b10; --surface3:#2a2114;
    --line:rgba(150,124,84,.10);
    --text:#e8ddc5; --muted:#9a8c72; --faint:#81765b; --dim:#6b5f49;
    --accent:#cf8a3c; --accent2:#cda261; --accent-weak:rgba(207,138,60,.14); --on-accent:#1a1206;
    --user:#b49f78; --bot:#d29a4b; --danger:#c8604c; --danger-weak:rgba(200,90,70,.14); --ok:#94a05c;
    --code-bg:#0c0905; --shadow:0 24px 70px rgba(0,0,0,.55);
    --mono:'IBM Plex Mono',ui-monospace,SFMono-Regular,Consolas,monospace;
    --serif:'Newsreader','Iowan Old Style',Georgia,'Times New Roman',serif;
    color-scheme:dark;
  }
  [data-theme="light"]{
    --bg:#f0e8d6; --panel:#e9e0cc; --surface:#e3d8c1; --surface2:#dacdb0; --surface3:#d2c3a4;
    --line:rgba(90,68,34,.14);
    --text:#2c2316; --muted:#6a5c43; --faint:#80714f; --dim:#94855f;
    --accent:#b26a1d; --accent2:#7d5320; --accent-weak:rgba(178,106,29,.15);
    --user:#6d5a32; --bot:#955414; --danger:#a23a2a; --danger-weak:rgba(162,58,42,.12); --ok:#5f6b2f;
    --code-bg:#e6dcc6; --shadow:0 24px 60px rgba(60,45,20,.25);
    color-scheme:light;
  }
  /* ---- color schemes: a palette re-tints the whole variable set; sepia (no attribute) is the
     original brown/amber. Each palette defines dark + light; anything not overridden falls back
     to the base theme blocks above. Stored per browser (oracle_palette). */
  [data-theme="dark"][data-palette="slate"]{
    --bg:#0b0e13; --panel:#10141b; --surface:#151b24; --surface2:#1c2430; --surface3:#232d3c;
    --line:rgba(110,140,180,.10);
    --text:#d9e1ec; --muted:#8496ac; --faint:#6b7c92; --dim:#57667a;
    --accent:#6aa1d8; --accent2:#9db8d4; --accent-weak:rgba(106,161,216,.14); --on-accent:#0b0e13;
    --user:#93a5ba; --bot:#74a8d8; --ok:#7d9d68; --code-bg:#080a0e;
  }
  [data-theme="light"][data-palette="slate"]{
    --bg:#e9edf2; --panel:#e0e6ee; --surface:#d7dfe8; --surface2:#cad4e0; --surface3:#bdc9d8;
    --line:rgba(60,90,130,.14);
    --text:#232b35; --muted:#52627a; --faint:#68788e; --dim:#8090a4;
    --accent:#2f6cab; --accent2:#24507e; --accent-weak:rgba(47,108,171,.15); --on-accent:#f2f5f9;
    --user:#4b5b70; --bot:#2a6096; --ok:#4f7a4f; --code-bg:#dde4ee; --shadow:0 24px 60px rgba(20,30,45,.22);
  }
  [data-theme="dark"][data-palette="moss"]{
    --bg:#0c100b; --panel:#111710; --surface:#161e15; --surface2:#1d271b; --surface3:#253122;
    --line:rgba(130,160,110,.10);
    --text:#dfe8d8; --muted:#90a385; --faint:#748767; --dim:#5e6f53;
    --accent:#8fb573; --accent2:#b0c898; --accent-weak:rgba(143,181,115,.14); --on-accent:#0c100b;
    --user:#a2b092; --bot:#97bd78; --ok:#94a05c; --code-bg:#080b07;
  }
  [data-theme="light"][data-palette="moss"]{
    --bg:#eaeee1; --panel:#e0e6d3; --surface:#d6dec7; --surface2:#c9d4b6; --surface3:#bbc9a6;
    --line:rgba(80,100,50,.15);
    --text:#262d1d; --muted:#5a684a; --faint:#70805c; --dim:#879673;
    --accent:#52803a; --accent2:#3c5f28; --accent-weak:rgba(82,128,58,.15); --on-accent:#f3f6ec;
    --user:#57654a; --bot:#47732c; --ok:#5f6b2f; --code-bg:#e0e6d2; --shadow:0 24px 60px rgba(30,40,20,.22);
  }
  [data-theme="dark"][data-palette="iris"]{
    --bg:#0e0c14; --panel:#131019; --surface:#191521; --surface2:#211c2c; --surface3:#2a2438;
    --line:rgba(150,130,190,.11);
    --text:#e2dcee; --muted:#948aab; --faint:#786f8f; --dim:#615a76;
    --accent:#a488e8; --accent2:#bfadde; --accent-weak:rgba(164,136,232,.14); --on-accent:#0e0c14;
    --user:#a49ac0; --bot:#ac90e8; --ok:#7d9d68; --code-bg:#0a0810;
  }
  [data-theme="light"][data-palette="iris"]{
    --bg:#ece9f2; --panel:#e3deec; --surface:#d9d3e5; --surface2:#ccc4dc; --surface3:#beb4d1;
    --line:rgba(90,70,130,.14);
    --text:#292234; --muted:#5c5175; --faint:#71678b; --dim:#877ea0;
    --accent:#6a4cbe; --accent2:#4e3790; --accent-weak:rgba(106,76,190,.14); --on-accent:#f4f2f9;
    --user:#574d70; --bot:#5f41b2; --ok:#5f7a3f; --code-bg:#e2ddec; --shadow:0 24px 60px rgba(35,25,55,.22);
  }
  [data-theme="dark"][data-palette="ash"]{
    --bg:#0e0f11; --panel:#131417; --surface:#191a1e; --surface2:#212226; --surface3:#292a30;
    --line:rgba(150,155,170,.10);
    --text:#e3e5e8; --muted:#92959d; --faint:#767a82; --dim:#60636b;
    --accent:#aab6c4; --accent2:#c4ccd6; --accent-weak:rgba(170,182,196,.14); --on-accent:#0e0f11;
    --user:#a0a5ad; --bot:#b4bfcb; --ok:#8a9a70; --code-bg:#0a0b0c;
  }
  [data-theme="light"][data-palette="ash"]{
    --bg:#ebebec; --panel:#e2e2e4; --surface:#d9d9dc; --surface2:#cdcdd1; --surface3:#c1c1c6;
    --line:rgba(70,72,80,.14);
    --text:#26272b; --muted:#595c63; --faint:#6f7279; --dim:#85888f;
    --accent:#4c5a68; --accent2:#363f4a; --accent-weak:rgba(76,90,104,.15); --on-accent:#f4f4f6;
    --user:#4f545c; --bot:#3e4f5e; --ok:#56713f; --code-bg:#e1e1e4; --shadow:0 24px 60px rgba(25,27,32,.22);
  }
  *{box-sizing:border-box;}
  html,body{height:100%;}
  body{margin:0;background:var(--bg);color:var(--text);font-family:var(--serif);font-size:16px;overflow:hidden;-webkit-font-smoothing:antialiased;}
  #app{display:flex;height:100vh;height:100dvh;}
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
  #newbtn:hover{background:var(--accent);color:var(--on-accent);}
  .side-act .row{display:flex;gap:8px;}
  .searchwrap{position:relative;}
  #searchbox{width:100%;background:var(--surface);color:var(--text);border:none;border-radius:9px;padding:9px 30px 9px 11px;font-family:var(--mono);font-size:12px;}
  #searchbox:focus{outline:2px solid var(--accent-weak);outline-offset:-1px;}
  #searchclear{position:absolute;right:5px;top:50%;transform:translateY(-50%);background:none;border:none;color:var(--faint);
               cursor:pointer;font-size:17px;line-height:1;padding:2px 7px;border-radius:6px;display:none;}
  .searchwrap.has #searchclear{display:block;} #searchclear:hover{color:var(--accent);}
  .side-act .row button{flex:1;background:var(--surface);border:none;color:var(--muted);border-radius:9px;padding:9px;font-family:var(--mono);font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;cursor:pointer;}
  .side-act .row button:hover,#selbtn.on{background:var(--surface2);color:var(--text);}
  #tree{flex:1;min-height:0;overflow-y:auto;padding:2px 8px 16px;}
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
  .convo:focus-visible,.folder-head:focus-visible{outline:2px solid var(--accent-weak);outline-offset:-1px;background:var(--surface);}
  .convo.active{background:var(--surface2);}
  .convo .ct{font-family:var(--serif);font-size:calc(15px*var(--rs));line-height:1.3;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
  .convo .cm{font-family:var(--mono);font-size:10px;color:var(--faint);margin-top:3px;letter-spacing:.03em;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
  .convo .cs{font-family:var(--read-font);font-size:11.5px;color:var(--muted);margin-top:5px;line-height:1.4;
             display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;}
  .convo .cs mark{background:var(--accent-weak);color:var(--accent);border-radius:3px;padding:0 1px;}
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
  #log{flex:1;min-height:0;overflow-y:auto;padding:74px 0 10px;}
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
  .bubble pre{position:relative;background:var(--code-bg);border-radius:10px;padding:13px 15px;overflow-x:auto;margin:.7em 0;}
  .bubble pre code{background:none;padding:0;font-size:.84em;line-height:1.55;}
  .copy-code{position:absolute;top:7px;right:7px;background:var(--surface);color:var(--faint);border:1px solid var(--line);
             border-radius:6px;padding:3px 8px;font-family:var(--mono);font-size:10px;letter-spacing:.06em;text-transform:uppercase;
             cursor:pointer;opacity:0;transition:opacity .14s,color .14s;}
  .bubble pre:hover .copy-code,.copy-code:focus{opacity:1;} .copy-code:hover{color:var(--accent);}
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
  .meta .pill.k.has-tip{cursor:help;text-decoration:underline dotted var(--faint);text-underline-offset:2px;}
  .actions{margin-top:11px;display:flex;gap:3px;flex-wrap:wrap;opacity:0;transition:opacity .12s;}
  .msg:hover .actions,.msg:focus-within .actions{opacity:1;}
  .actions button{background:none;border:none;color:var(--faint);font-size:10px;letter-spacing:.07em;text-transform:uppercase;cursor:pointer;padding:4px 8px;border-radius:7px;}
  .actions button:hover{background:var(--surface);color:var(--text);}
  .actions button.danger:hover{color:var(--danger);}
  .edit-area{width:100%;background:var(--code-bg);color:var(--text);border:none;border-radius:10px;padding:12px 13px;font-size:calc(16px*var(--rs));font-family:var(--read-font);line-height:1.6;resize:vertical;min-height:96px;}
  .edit-area:focus{outline:2px solid var(--accent-weak);}
  .edit-row{display:flex;gap:8px;margin-top:8px;}
  .btn-primary{background:var(--accent);color:var(--on-accent);border:none;border-radius:8px;padding:8px 15px;font-family:var(--mono);font-size:11px;letter-spacing:.08em;text-transform:uppercase;cursor:pointer;}
  .btn-primary:hover{filter:brightness(1.07);}
  .btn-ghost{background:var(--surface2);color:var(--muted);border:none;border-radius:8px;padding:8px 15px;font-family:var(--mono);font-size:11px;letter-spacing:.08em;text-transform:uppercase;cursor:pointer;}
  .btn-ghost:hover{background:var(--surface3);color:var(--text);}
  .btn-danger{background:var(--danger-weak);color:var(--danger);border:none;border-radius:8px;padding:8px 15px;font-family:var(--mono);font-size:11px;letter-spacing:.08em;text-transform:uppercase;cursor:pointer;}
  .btn-danger:hover{background:var(--danger);color:#fff;}
  .typing{display:inline-flex;gap:5px;align-items:center;padding:3px 0;}
  .typing i{width:5px;height:5px;border-radius:50%;background:var(--accent);opacity:.4;animation:blink 1.2s infinite;}
  .typing i:nth-child(2){animation-delay:.2s;} .typing i:nth-child(3){animation-delay:.4s;}
  @keyframes blink{0%,80%,100%{opacity:.25;}40%{opacity:1;}}
  /* live reply: settled markdown blocks (.mds) + the block still being written (.mdt). The pair is
     one bubble, so restore the inter-block margin the :first/:last-child rules zero at the seam. */
  .tmpnode{overflow-anchor:none;}   /* scroll anchoring must not fight the autoscroll */
  .bubble .mds:not(:empty)>p:last-child{margin-bottom:.7em;}
  .bubble .mds:not(:empty)+.mdt>p:first-child{margin-top:.7em;}
  .caret{display:inline-block;width:.42em;height:1.02em;margin-left:1px;vertical-align:-.16em;
    background:var(--accent);opacity:.6;border-radius:1px;animation:caret 1.05s steps(1) infinite;}
  @keyframes caret{0%,55%{opacity:.6;}56%,100%{opacity:0;}}
  @media (prefers-reduced-motion:reduce){.caret{animation:none;}}
  .empty{text-align:center;margin-top:16vh;color:var(--faint);}
  .empty .glyph{font-family:var(--mono);font-size:30px;color:var(--accent);opacity:.7;}
  .empty h2{font-family:var(--serif);font-weight:500;color:var(--muted);font-size:22px;margin:14px 0 4px;}
  .empty p{font-family:var(--mono);font-size:11px;letter-spacing:.06em;color:var(--faint);}

  /* ---------------- composer */
  #composer{padding:12px 0 18px;background:var(--bg);padding-bottom:max(18px,env(safe-area-inset-bottom));}
  #composer .wrap{display:flex;gap:10px;align-items:flex-end;}
  .input-shell{flex:1;display:flex;align-items:flex-end;background:var(--surface);border-radius:14px;padding:12px 14px;min-height:var(--ctrl-h);}
  .input-shell:focus-within{outline:2px solid var(--accent-weak);outline-offset:-1px;}
  #input{flex:1;background:none;color:var(--text);border:none;font-size:calc(16.5px*var(--rs));font-family:var(--read-font);resize:none;max-height:230px;line-height:1.5;padding:0;}
  #input:focus{outline:none;}
  #send{flex:0 0 auto;background:var(--accent);color:#fff;border:none;border-radius:14px;width:var(--ctrl-h);height:var(--ctrl-h);cursor:pointer;display:flex;align-items:center;justify-content:center;}
  #send:hover{filter:brightness(1.08);} #send.stop{background:var(--danger);color:#fff;}
  #send .ico{width:20px;height:20px;stroke-width:2.2;}
  .chint{max-width:var(--cw);margin:7px auto 0;padding:0 28px;font-family:var(--mono);font-size:10px;color:var(--dim);letter-spacing:.03em;}

  /* ---------------- context meter */
  #ctxmeter{display:flex;align-items:center;gap:7px;height:32px;padding:0 9px;border-radius:8px;background:var(--surface);}
  #ctxmeter .ctxbar{width:52px;height:5px;border-radius:3px;background:var(--surface3);overflow:hidden;}
  #ctxmeter .ctxbar i{display:block;height:100%;width:0;background:var(--ok);transition:width .3s,background .3s;}
  #ctxmeter.warn .ctxbar i{background:var(--bot);} #ctxmeter.hot .ctxbar i{background:var(--danger);}
  #ctxmeter .ctxtxt{font-family:var(--mono);font-size:10px;letter-spacing:.02em;color:var(--muted);white-space:nowrap;}
  @media (max-width:620px){ #ctxmeter .ctxtxt{display:none;} }

  /* ---------------- composer: attach + pending files */
  #attachbtn{flex:0 0 auto;background:var(--surface);color:var(--muted);border:none;border-radius:14px;width:var(--ctrl-h);height:var(--ctrl-h);cursor:pointer;display:flex;align-items:center;justify-content:center;}
  #attachbtn:hover{background:var(--surface2);color:var(--text);}
  #attachbtn .ico{width:19px;height:19px;stroke-width:2;}
  #pending{max-width:var(--cw);margin:0 auto 8px;padding:0 28px;display:flex;flex-wrap:wrap;gap:7px;}
  #pending:empty{display:none;}
  .achip{display:inline-flex;align-items:center;gap:7px;background:var(--surface2);border:1px solid var(--line);border-radius:9px;padding:5px 9px;font-family:var(--mono);font-size:10.5px;color:var(--text);max-width:280px;}
  .achip .nm{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
  .achip .tk{color:var(--faint);} .achip.busy{opacity:.6;} .achip.err{border-color:var(--danger);color:var(--danger);}
  .achip .rm{background:none;border:none;color:var(--faint);cursor:pointer;font-size:14px;line-height:1;padding:0;} .achip .rm:hover{color:var(--danger);}
  #composer.dragover .input-shell{outline:2px dashed var(--accent);outline-offset:3px;}

  /* ---------------- floating scroll buttons */
  #scrolltop,#scrollbottom{position:absolute;right:16px;bottom:92px;width:42px;height:42px;border-radius:50%;background:var(--surface2);color:var(--text);border:1px solid var(--line);box-shadow:var(--shadow);display:none;align-items:center;justify-content:center;cursor:pointer;z-index:30;opacity:.94;}
  #scrollbottom{bottom:144px;color:var(--accent);}
  #scrolltop.show,#scrollbottom.show{display:flex;}
  #scrolltop:hover,#scrollbottom:hover{background:var(--surface3);opacity:1;}
  #scrolltop .ico,#scrollbottom .ico{width:18px;height:18px;stroke-width:2.3;}
  .msg .atts{display:flex;flex-wrap:wrap;gap:6px;margin:0 0 9px;}
  .msg .atts details{background:var(--surface);border:1px solid var(--line);border-radius:9px;max-width:100%;}
  .msg .atts summary{cursor:pointer;list-style:none;padding:5px 9px;font-family:var(--mono);font-size:10.5px;color:var(--muted);display:flex;gap:7px;align-items:center;}
  .msg .atts summary::-webkit-details-marker{display:none;}
  .msg .atts .atxt{max-height:280px;overflow:auto;margin:0;padding:10px 12px;border-top:1px solid var(--line);font-family:var(--mono);font-size:11px;white-space:pre-wrap;color:var(--muted);}

  /* ---------------- tool steps */
  .toolstep{margin:7px 0;border:1px solid var(--line);border-radius:10px;background:var(--surface);overflow:hidden;}
  .toolstep summary{cursor:pointer;list-style:none;padding:8px 11px;font-family:var(--mono);font-size:11px;color:var(--accent2);display:flex;gap:8px;align-items:center;}
  .toolstep summary::-webkit-details-marker{display:none;}
  .toolstep .tlabel{color:var(--muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;} .toolstep.err summary{color:var(--danger);}
  .toolstep .tbody{max-height:300px;overflow:auto;margin:0;padding:10px 12px;border-top:1px solid var(--line);font-family:var(--mono);font-size:11px;white-space:pre-wrap;color:var(--muted);}
  .toolstep .turl{color:var(--accent);text-decoration:none;word-break:break-all;} .toolstep .turl:hover{text-decoration:underline;}
  .toolspin{display:inline-block;width:10px;height:10px;border:2px solid var(--surface3);border-top-color:var(--accent);border-radius:50%;animation:tsp .7s linear infinite;flex:0 0 auto;}
  @keyframes tsp{to{transform:rotate(360deg);}}
  .thinking .thinkrow{display:flex;align-items:center;gap:9px;color:var(--muted);font-family:var(--read-font);font-style:italic;font-size:calc(15px*var(--rs));padding:3px 0;}
  .thinkdot{width:9px;height:9px;border-radius:50%;background:var(--accent);animation:thinkpulse 1.15s ease-in-out infinite;flex:0 0 auto;}
  @keyframes thinkpulse{0%,100%{opacity:.3;transform:scale(.82);}50%{opacity:1;transform:scale(1.12);}}
  .ctx-notice{margin:4px 0 16px;padding:10px 13px;border-radius:9px;background:var(--danger-weak);
              border:1px solid var(--danger);color:var(--danger);font-family:var(--mono);
              font-size:11.5px;line-height:1.55;letter-spacing:.01em;}
  .chk .chkrow{display:flex;align-items:center;gap:8px;font-size:13px;color:var(--muted);cursor:pointer;}
  .chk .chkrow code{font-family:var(--mono);font-size:11px;background:var(--code-bg);padding:.05em .3em;border-radius:4px;}
  input[type=checkbox]{appearance:none;-webkit-appearance:none;flex:0 0 auto;width:17px;height:17px;margin:0;border:1px solid var(--line);border-radius:5px;background:var(--bg);cursor:pointer;display:inline-grid;place-content:center;transition:background .12s,border-color .12s;}
  input[type=checkbox]:hover{border-color:var(--accent);}
  input[type=checkbox]:focus-visible{outline:2px solid var(--accent-weak);outline-offset:1px;}
  input[type=checkbox]:checked{background:var(--accent);border-color:var(--accent);}
  input[type=checkbox]:checked::after{content:"";width:9px;height:9px;clip-path:polygon(14% 46%,0 60%,40% 100%,100% 18%,86% 6%,38% 70%);background:var(--on-accent);}

  /* ---------------- overlays */
  #backdrop{position:fixed;inset:0;background:rgba(8,6,3,.5);opacity:0;pointer-events:none;transition:opacity .2s;z-index:50;}
  #backdrop.show{opacity:1;pointer-events:auto;}
  .panel{position:fixed;top:0;right:0;height:100vh;height:100dvh;width:430px;max-width:93vw;background:var(--panel);box-shadow:var(--shadow);transform:translateX(103%);transition:transform .22s ease;z-index:70;display:flex;flex-direction:column;}
  .panel.show{transform:translateX(0);}
  .phead{padding:16px 18px;display:flex;align-items:center;gap:10px;}
  .phead h3{margin:0;font-family:var(--mono);font-size:12px;letter-spacing:.16em;text-transform:uppercase;color:var(--accent);}
  .phead .x{margin-left:auto;background:none;border:none;color:var(--muted);font-size:22px;cursor:pointer;width:32px;height:32px;border-radius:8px;line-height:1;}
  .phead .x:hover{background:var(--surface);color:var(--text);}
  .pbody{flex:1;min-height:0;overflow-y:auto;padding:8px 18px 16px;}
  .pfoot{padding:14px 18px;display:flex;gap:8px;align-items:center;}
  .pfoot button{flex:1;border-radius:9px;padding:11px;font-size:12px;letter-spacing:.1em;text-transform:uppercase;cursor:pointer;}
  .pfoot .dhint{flex:1;font-family:var(--mono);font-size:10px;color:var(--faint);letter-spacing:.04em;text-transform:uppercase;}
  .pfoot .dhint+button{flex:0 0 auto;padding:11px 22px;}
  #modal{position:fixed;inset:0;display:none;align-items:center;justify-content:center;z-index:70;padding:20px;}
  #modal.show{display:flex;}
  .modal-card{background:var(--panel);border-radius:16px;box-shadow:var(--shadow);width:800px;max-width:100%;max-height:90vh;max-height:90dvh;display:flex;flex-direction:column;overflow:hidden;}
  .mhead{padding:16px 20px;display:flex;align-items:center;gap:14px;}
  .mhead h3{margin:0;font-family:var(--mono);font-size:12px;letter-spacing:.18em;text-transform:uppercase;color:var(--accent);}
  .mhead .x{margin-left:auto;background:none;border:none;color:var(--muted);font-size:22px;cursor:pointer;}
  .tabs{display:flex;gap:2px;padding:0 16px;flex-wrap:wrap;}
  .tabs button{background:none;border:none;color:var(--faint);padding:9px 13px;font-family:var(--mono);font-size:11px;letter-spacing:.09em;text-transform:uppercase;cursor:pointer;border-radius:8px 8px 0 0;}
  .tabs button.active{color:var(--text);background:var(--surface);}
  .tab-body{flex:1;overflow-y:auto;padding:18px 20px;background:var(--surface);}
  #peekmodal{position:fixed;inset:0;display:none;align-items:center;justify-content:center;z-index:72;padding:20px;}
  #peekmodal.show{display:flex;}
  .peek-card{width:1040px;height:88vh;height:88dvh;}
  #peek-tabs{margin-left:6px;}
  .peek-body{flex:1;min-height:0;background:var(--surface);display:flex;overflow:hidden;}
  .peek-pane{flex:1;min-height:0;display:flex;}
  #peek-chars{display:none;overflow-y:auto;padding:16px 18px;flex-direction:column;gap:10px;}
  .peek-list{flex:0 0 300px;width:300px;overflow-y:auto;border-right:1px solid var(--line);padding:10px;}
  .peek-view{flex:1;min-width:0;overflow-y:auto;padding:16px 22px;}
  .peek-view .wrap{max-width:none;}
  .peek-item{padding:9px 11px;border-radius:9px;cursor:pointer;margin-bottom:3px;}
  .peek-item:hover,.peek-item.on{background:var(--panel);}
  .peek-item .pt{font-family:var(--serif);font-size:14px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
  .peek-item .ps{font-family:var(--mono);font-size:10px;color:var(--faint);letter-spacing:.02em;margin-top:2px;}
  .peek-hint{color:var(--faint);font-family:var(--mono);font-size:12px;padding:24px;text-align:center;}
  .peek-char .cn{font-family:var(--serif);font-size:15px;margin-bottom:6px;}
  .peek-sys{font-family:var(--mono);font-size:11.5px;color:var(--muted);white-space:pre-wrap;line-height:1.55;max-height:240px;overflow:auto;background:var(--bg);border-radius:8px;padding:10px 12px;}
  @media (max-width:680px){.peek-card{height:90vh;height:90dvh;} .peek-list{flex-basis:42%;width:42%;}}
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
  .pg .pgname{display:flex;align-items:center;font-family:var(--mono);font-size:10.5px;color:var(--muted);letter-spacing:.06em;margin-bottom:6px;}
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
  .tip-ic{display:inline-flex;align-items:center;justify-content:center;width:14px;height:14px;border-radius:50%;border:1px solid var(--faint);color:var(--faint);font-size:9px;font-style:normal;font-weight:600;font-family:var(--mono);line-height:1;cursor:help;margin-left:6px;flex:0 0 auto;user-select:none;-webkit-user-select:none;}
  .tip-ic:hover,.tip-ic.on{border-color:var(--accent);color:var(--accent);}
  .tipbox{position:fixed;z-index:120;max-width:262px;white-space:pre-line;background:var(--surface3);color:var(--text);border:1px solid var(--line);border-radius:9px;padding:9px 11px;font-family:var(--mono);font-size:11px;line-height:1.55;letter-spacing:.01em;box-shadow:var(--shadow);opacity:0;pointer-events:none;transition:opacity .12s ease;}
  .tipbox.show{opacity:1;}
  .preset-row{display:flex;gap:8px;align-items:center;margin-top:10px;}
  .preset-row select{flex:1;min-width:0;background:var(--bg);color:var(--text);border:none;border-radius:8px;padding:9px 10px;font-family:var(--mono);font-size:12px;cursor:pointer;}
  .preset-row select:focus{outline:2px solid var(--accent-weak);outline-offset:-1px;}
  .dlg-chk{display:flex;gap:8px;align-items:center;font-family:var(--mono);font-size:12px;color:var(--muted);margin:8px 0 2px;cursor:pointer;}
  .dlg-models{display:flex;flex-wrap:wrap;gap:7px;max-height:170px;overflow:auto;margin:2px 0 4px;padding:2px;}
  .dlg-models label{display:inline-flex;gap:7px;align-items:center;background:var(--surface);border-radius:20px;padding:6px 12px;font-family:var(--mono);font-size:11px;color:var(--muted);cursor:pointer;}
  .dlg-models label:hover{color:var(--text);}
  .dlg-hint{font-family:var(--mono);font-size:10.5px;color:var(--faint);margin:0 0 6px;line-height:1.5;}
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
  .invite-sec{margin-top:20px;border-top:1px solid var(--line);padding-top:16px;}
  .invite-url{width:100%;margin-top:8px;background:var(--surface2);border:none;border-radius:7px;padding:7px 9px;font-family:var(--mono);font-size:11px;color:var(--accent2);cursor:text;}
  .invite-url:focus{outline:2px solid var(--accent-weak);outline-offset:-1px;}
  .row-card .cs{font-family:var(--mono);font-size:11px;color:var(--faint);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-top:2px;}
  .row-card .cbtns{display:flex;gap:6px;}
  .toggle-row{display:flex;align-items:center;gap:8px;font-family:var(--mono);font-size:11px;color:var(--muted);margin-top:14px;letter-spacing:.03em;}
  .seg{display:inline-flex;background:var(--bg);border-radius:9px;padding:3px;gap:3px;}
  .seg button{background:none;border:none;color:var(--muted);padding:7px 14px;border-radius:7px;font-family:var(--mono);font-size:11px;letter-spacing:.06em;text-transform:uppercase;cursor:pointer;}
  .seg button.on{background:var(--surface2);color:var(--text);}
  .seg .pdot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;vertical-align:0;}
  #palseg{flex-wrap:wrap;}
  .cpick{display:flex;align-items:center;gap:7px;font-family:var(--mono);font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);cursor:pointer;}
  .cpick input[type=color]{width:32px;height:24px;border:1px solid var(--line);border-radius:6px;background:none;padding:1px;cursor:pointer;}
  .stats-table{width:100%;border-collapse:collapse;font-size:12.5px;}
  .stats-table th{font-family:var(--mono);font-size:10px;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);}
  .stats-table td{padding:6px 8px;border-bottom:1px solid var(--line);color:var(--text);}
  .stats-table td.mono{font-family:var(--mono);font-size:11px;}
  .krow{display:flex;gap:10px;align-items:center;font-family:var(--mono);font-size:11.5px;color:var(--muted);padding:5px 0;border-bottom:1px solid var(--line);}
  .krow .kn{color:var(--text);flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
  .krow .kc{color:var(--faint);flex:0 0 auto;}
  .atts .attimg{max-width:min(340px,100%);max-height:340px;border-radius:10px;display:block;margin:6px 0;}
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
  .dlg input:not([type=checkbox]),.dlg select{width:100%;background:var(--surface);color:var(--text);border:none;border-radius:9px;padding:11px;font-size:14px;font-family:var(--mono);margin-bottom:6px;}
  .dlg input:focus,.dlg select:focus{outline:2px solid var(--accent-weak);}
  .dlg .dlg-btns{display:flex;gap:8px;justify-content:flex-end;margin-top:14px;}
  .dlg .sharelink{display:flex;gap:8px;align-items:stretch;margin-top:4px;}
  .dlg .sharelink input{flex:1;margin-bottom:0;font-size:12px;}
  .dlg .sharelink .btn-primary{flex:0 0 auto;}
  .dlg .share-meta{font-family:var(--mono);font-size:11px;color:var(--faint);margin:9px 0 0;letter-spacing:.02em;}
  .dlg .share-actions{display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin-top:13px;}
  .dlg .linkbtn{background:none;border:none;padding:0;cursor:pointer;text-decoration:none;
                font-family:var(--mono);font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);}
  .dlg .linkbtn:hover{color:var(--accent);}
  .dlg .linkbtn.danger{color:var(--faint);} .dlg .linkbtn.danger:hover{color:var(--danger);}
  .dlg .share-actions .grow{flex:1;}
  #toast{position:fixed;bottom:96px;left:50%;transform:translateX(-50%) translateY(8px);background:var(--surface3);padding:10px 16px;border-radius:10px;font-family:var(--mono);font-size:12px;opacity:0;transition:opacity .25s,transform .25s;pointer-events:none;z-index:99;box-shadow:var(--shadow);letter-spacing:.03em;}
  #toast.show{opacity:1;transform:translateX(-50%) translateY(0);}
  #toast.err{background:var(--danger-weak);color:var(--danger);box-shadow:var(--shadow),inset 0 0 0 1px var(--danger);}

  /* responsive */
  @media (max-width:860px){
    #sidebar{position:fixed;left:0;top:0;height:100vh;height:100dvh;transform:translateX(-103%);transition:transform .22s ease;box-shadow:var(--shadow);z-index:80;}
    #sidebar.show{transform:translateX(0);}
    #app.sbcollapsed #sidebar{display:flex;}
    #resizer{display:none;}
    .menubtn{display:flex;} #revealbtn{display:none!important;} #collapsebtn{display:none;}
    #charchip{display:none;} select.msel{max-width:31vw;}
    .wrap,.chint,#pending{padding:0 16px;}
    .bubble{font-size:calc(17px*var(--rs));}
    /* tap a message to reveal its actions (instead of always-on hover) */
    .actions{display:none;}
    .msg.revealed .actions{display:flex;opacity:1;}
    .msg.revealed>.bubble{box-shadow:-3px 0 0 var(--accent-weak);}
    .convo .cmenu,.folder-head .fmenu{opacity:1;}
  }
  /* compose mode: one continuable document, edited raw in place (a loom, not a chat) */
  .compose{max-width:var(--cw);margin:0 auto;padding:0 28px 20px;display:flex;flex-direction:column;height:100%;gap:10px;}
  .cbar{display:flex;align-items:center;gap:12px;color:var(--faint);font-family:var(--mono);font-size:11px;flex:0 0 auto;}
  .cbar .grow{flex:1;}
  #ctext{flex:1;min-height:240px;width:100%;resize:none;background:var(--surface);color:var(--text);border:none;border-radius:12px;
         padding:18px 20px;font-family:var(--mono);font-size:13.5px;line-height:1.75;tab-size:2;}
  #ctext:focus{outline:2px solid var(--accent-weak);outline-offset:-1px;}
  .crow{display:flex;align-items:center;gap:8px;flex:0 0 auto;}
  .crow button{background:var(--surface);border:none;color:var(--muted);border-radius:9px;padding:9px 16px;
               font-family:var(--mono);font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;cursor:pointer;}
  .crow button:hover{background:var(--surface2);color:var(--text);}
  .crow button.pri{background:var(--accent-weak);color:var(--accent);font-weight:600;}
  .crow button.pri:hover{background:var(--accent);color:var(--on-accent);}
  .crow .chint{color:var(--faint);font-family:var(--mono);font-size:10.5px;margin-left:auto;}
  @media (max-width:560px){
    .compose{padding:0 12px 12px;} .crow .chint{display:none;}
    .params-grid{grid-template-columns:1fr;} .barbtn .t{display:none;}
    #composer .wrap{gap:6px;}
    #scrolltop{bottom:86px;right:12px;}
    #scrollbottom{bottom:136px;right:12px;}
  }
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
        <div class="searchwrap"><input id="searchbox" placeholder="search…"><button id="searchclear" type="button" title="clear search" aria-label="clear search">&times;</button></div>
        <div class="row"><button id="composebtn" title="a single continuable text — paste something and the model keeps writing it">+ composition</button><button id="foldernew" title="new folder">+ folder</button><button id="selbtn" title="select multiple">select</button></div>
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
          <div id="ctxmeter" title="context usage" style="display:none"><div class="ctxbar"><i></i></div><span class="ctxtxt"></span></div>
          <button class="barbtn" id="sharebtn" title="share this chat publicly"><svg class="ico" viewBox="0 0 24 24" style="width:15px;height:15px"><path d="M4 12v7a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-7M16 6l-4-4-4 4M12 2v13"/></svg><span class="t">share</span></button>
          <button class="barbtn" id="exportbtn" title="export this chat">export</button>
          <button class="barbtn" id="charchip"><span class="t">character</span></button>
          <select class="msel" id="modelsel" title="model"></select>
          <button class="barbtn" id="tunebtn">tune</button>
        </div>
      </header>
      <div id="log"></div>
      <footer id="composer">
        <div id="pending"></div>
        <div class="wrap">
          <button id="attachbtn" title="attach a file (PDF, text, markdown)"><svg class="ico" viewBox="0 0 24 24"><path d="M21.44 11.05l-9.19 9.19a5 5 0 0 1-7.07-7.07l9.19-9.19a3 3 0 0 1 4.24 4.24l-9.19 9.19a1 1 0 0 1-1.41-1.41l8.49-8.49"/></svg></button>
          <input type="file" id="fileinput" multiple accept=".pdf,.txt,.md,.markdown,.text,.csv,.tsv,.json,.log,.rst,.yaml,.yml,image/png,image/jpeg,image/webp,image/gif" style="display:none">
          <div class="input-shell">
            <textarea id="input" rows="1" placeholder="say something…   (enter to send, shift+enter for newline)"></textarea>
          </div>
          <button id="send" title="send"><svg class="ico" viewBox="0 0 24 24"><path d="M5 12h14M13 6l6 6-6 6"/></svg></button>
        </div>
        <div class="chint" id="chint"></div>
      </footer>
      <button id="scrolltop" title="scroll to top" aria-label="scroll to top"><svg class="ico" viewBox="0 0 24 24"><path d="M12 19V5M5 12l7-7 7 7"/></svg></button>
      <button id="scrollbottom" title="jump to latest" aria-label="jump to latest"><svg class="ico" viewBox="0 0 24 24"><path d="M12 5v14M5 12l7 7 7-7"/></svg></button>
    </main>
  </div>

  <div id="drawer" class="panel" role="dialog" aria-modal="true" aria-label="chat settings">
    <div class="phead"><h3>chat settings</h3><button class="x" data-close-drawer aria-label="close">&times;</button></div>
    <div class="pbody">
      <label class="fld"><span class="lab">character</span><select id="d_char"></select></label>
      <label class="fld" id="d_endpoint_wrap"><span class="lab">endpoint <span class="sub">(admin)</span></span><select id="d_endpoint"></select></label>
      <label class="fld"><span class="lab">model</span><select id="d_model"></select></label>
      <label class="fld"><span class="lab">system prompt <span class="sub">this chat only</span></span><textarea id="d_system" placeholder="empty = no system prompt"></textarea></label>
      <label class="fld chk"><span class="lab">web tools <span class="sub">let the model fetch web pages</span></span><label class="chkrow"><input type="checkbox" id="d_tools"> <span>enable <code>fetch_url</code> for this chat</span></label></label>
      <label class="fld" id="d_think_wrap" style="display:none;"><span class="lab">thinking <span class="sub">let the model reason before answering</span></span><select id="d_think"><option value="">model default</option><option value="1">on</option><option value="0">off</option></select></label>
      <div class="fld" id="d_summary_wrap" style="display:none;"><span class="lab">context summary <span class="sub">what the model sees in place of trimmed history — edit or clear it</span></span>
        <textarea id="d_summary" rows="5"></textarea>
        <div class="params-foot"><button class="mini" id="d_summary_save">save summary</button><button class="mini danger" id="d_summary_clear">clear (forget)</button></div>
      </div>
      <div class="fld params-section"><span class="lab">sampler parameters <span class="sub">blank = server default</span></span>
        <div class="preset-row"><select id="d_preset"></select><button class="mini" id="d_preset_save">save preset</button><button class="mini" id="d_preset_del" style="display:none">delete</button></div>
        <div class="params-grid" id="d_params"></div>
        <div class="params-foot"><button class="mini" id="d_defaults">server defaults</button><button class="mini" id="d_clear">clear all</button></div>
      </div>
    </div>
    <div class="pfoot"><span class="dhint">changes apply instantly</span><button class="btn-primary" data-close-drawer>done</button></div>
  </div>

  <div id="modal">
    <div class="modal-card" role="dialog" aria-modal="true" aria-label="settings">
      <div class="mhead"><h3>settings</h3><button class="x" data-close-modal aria-label="close">&times;</button></div>
      <div class="tabs" id="tabs"></div>
      <div class="tab-body">
        <div class="tab-pane" id="tab-account">
          <div class="lbl" style="margin-bottom:10px;">your name</div>
          <label class="fld"><span class="lab">display name <span class="sub">fills <code>{{user}}</code> in prompts · blank = your username</span></span><input type="text" id="ac_persona" maxlength="60" placeholder="e.g. Alex"></label>
          <div class="edit-row" style="margin-top:10px;"><button class="btn-primary" id="ac_persona_save">save name</button></div>
          <div class="lbl" style="margin:24px 0 10px;">security</div>
          <div id="totp_off">
            <button class="mini" id="totp_enable">enable two-factor (TOTP)</button>
            <div class="hintbox" style="margin-top:8px;">Adds a 6-digit authenticator code to sign-in. If you lose the authenticator, an admin must clear it in the database.</div>
          </div>
          <div id="totp_setup" style="display:none;">
            <div class="hintbox">Add this secret to your authenticator app (manual entry), then confirm with a code:</div>
            <div class="mono" id="totp_secret" style="margin:8px 0;user-select:all;word-break:break-all;"></div>
            <div style="display:flex;gap:8px;align-items:center;"><input type="text" id="totp_code" inputmode="numeric" placeholder="123456" style="max-width:120px;"><button class="mini" id="totp_confirm">confirm &amp; enable</button><button class="mini" id="totp_cancel">cancel</button></div>
          </div>
          <div id="totp_on" style="display:none;">
            <div class="hintbox">Two-factor is <b>on</b> for this account.</div>
            <div style="display:flex;gap:8px;align-items:center;margin-top:8px;"><input type="password" id="totp_pw" placeholder="password" style="max-width:180px;"><button class="mini danger" id="totp_disable">disable 2FA</button></div>
          </div>
          <div style="margin-top:14px;" id="push_wrap">
            <label class="chkrow" style="cursor:pointer;"><input type="checkbox" id="push_toggle"> <span>push notifications on this device (even with the tab closed)</span></label>
          </div>
          <div class="lbl" style="margin:24px 0 10px;">change password</div>
          <label class="fld"><span class="lab">current password</span><input type="password" id="ac_old"></label>
          <label class="fld"><span class="lab">new password <span class="sub">(min 8)</span></span><input type="password" id="ac_new"></label>
          <label class="fld"><span class="lab">confirm new</span><input type="password" id="ac_new2"></label>
          <div class="edit-row" style="margin-top:14px;"><button class="btn-primary" id="ac_save">change password</button></div>
          <div class="lbl" style="margin:24px 0 8px;">your data</div>
          <button class="mini" id="exportall">export all my chats (json)</button>
          <div style="margin-top:10px;"><button class="mini" id="imp_btn" type="button">import chats from file&#8230;</button>
            <input type="file" id="imp_file" style="display:none" accept=".json,.jsonl" multiple></div>
          <div class="hintbox" style="margin-top:8px;">Accepts a ChatGPT export (<code>conversations.json</code>), a SillyTavern chat (<code>.jsonl</code>), or any JSON with a <code>messages</code> list of <code>{role, content}</code>.</div>
          <div class="danger-zone">
            <div class="dz-t">danger zone</div>
            <p>Delete your account and every conversation, folder, and private character you own. This cannot be undone.</p>
            <button class="btn-danger" id="acctdel">delete my account</button>
          </div>
        </div>
        <div class="tab-pane" id="tab-appearance">
          <label class="fld"><span class="lab">theme</span><div class="seg" id="themeseg"><button data-th="dark" class="on">dark</button><button data-th="light">light</button></div></label>
          <label class="fld"><span class="lab">color</span><div class="seg" id="palseg"><button data-pal="sepia" class="on"><span class="pdot" style="background:#cf8a3c"></span>sepia</button><button data-pal="slate"><span class="pdot" style="background:#6aa1d8"></span>slate</button><button data-pal="moss"><span class="pdot" style="background:#8fb573"></span>moss</button><button data-pal="iris"><span class="pdot" style="background:#a488e8"></span>iris</button><button data-pal="ash"><span class="pdot" style="background:#aab6c4"></span>ash</button><button data-pal="custom"><span class="pdot" style="background:conic-gradient(#c85a46,#cf8a3c,#94a05c,#6aa1d8,#a488e8,#c85a46)"></span>custom</button></div></label>
          <div id="custwrap" style="display:none;margin:2px 0 6px;">
            <div style="display:flex;gap:16px;flex-wrap:wrap;align-items:center;">
              <label class="cpick"><input type="color" id="cp_bg"><span>background</span></label>
              <label class="cpick"><input type="color" id="cp_text"><span>text</span></label>
              <label class="cpick"><input type="color" id="cp_accent"><span>accent</span></label>
              <button class="mini" id="cp_reset">reset</button>
            </div>
            <div class="hintbox" style="margin-top:8px;">Pick three colors — surfaces, borders, and bubbles are derived from them. Custom overrides the light/dark toggle.</div>
          </div>
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
            <div class="hintbox" style="margin-top:8px;">Macros (resolved when sending): <code>{{user}}</code> <code>{{char}}</code> <code>{{model}}</code> <code>{{date}}</code> <code>{{weekday}}</code> <code>{{random:a,b,c}}</code> <code>{{roll:2d6}}</code> <code>{{newline}}</code></div>
            <div class="fld" style="margin-top:12px;"><span class="lab">knowledge <span class="sub">documents searched each turn; best matches are shown to the model</span></span>
              <div id="ch_klist"></div>
              <div style="margin-top:6px;"><button class="mini" id="ch_kadd" type="button">+ add document</button>
                <input type="file" id="ch_kfile" style="display:none" multiple accept=".txt,.md,.markdown,.text,.csv,.tsv,.json,.log,.rst,.yaml,.yml,.pdf"></div>
            </div>
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
          <label class="fld"><span class="lab">web search endpoint <span class="sub">SearXNG base URL (JSON api enabled) · blank disables the web_search tool</span></span>
            <input type="text" id="def_search" placeholder="https://searx.example.com"></label>
          <div class="fld"><span class="lab">thinking-capable models <span class="sub">show a per-chat thinking toggle for these</span></span>
            <div class="hintbox" style="margin-top:6px;">Only enable for models whose chat template supports it (e.g. Qwen3, DeepSeek-R1). Sends <code>enable_thinking</code> to the endpoint.</div>
            <div class="model-pick" id="tm_pick"></div>
          </div>
          <div class="fld"><span class="lab">vision-capable models <span class="sub">image attachments are sent as pixels to these; others get a text note</span></span>
            <div class="model-pick" id="vm_pick"></div>
          </div>
          <label class="fld"><span class="lab">utility model <span class="sub">for titles + context summaries · blank = the chat's own model</span></span>
            <input type="text" id="def_utility" placeholder="e.g. a small always-loaded model"></label>
          <label class="fld"><span class="lab">embedding model <span class="sub">enables hybrid (semantic) knowledge retrieval via /v1/embeddings · blank = keyword only</span></span>
            <input type="text" id="def_embed" placeholder="e.g. nomic-embed-text"></label>
        </div>
        <div class="tab-pane" id="tab-stats">
          <div id="stats-body"></div>
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
          <div class="invite-sec">
            <div class="lab" style="margin-bottom:8px;">invite links <span class="sub">let people self-register</span></div>
            <div id="invite-list"></div>
            <button class="mini" id="invite-add">+ create invite link</button>
            <div id="invite-edit" style="display:none;margin-top:12px;">
              <label class="fld"><span class="lab">role</span><select id="iv_role"><option value="user">user</option><option value="admin">admin</option></select></label>
              <div class="fld"><span class="lab">allowed models <span class="sub">none checked = use global list</span></span><div class="model-pick" id="iv_models"></div></div>
              <div class="edit-row">
                <label class="fld" style="flex:1;"><span class="lab">expires in days <span class="sub">blank = never</span></span><input type="number" id="iv_days" min="0" step="1" placeholder="never"></label>
                <label class="fld" style="flex:1;"><span class="lab">max uses <span class="sub">blank = unlimited</span></span><input type="number" id="iv_uses" min="1" step="1" placeholder="unlimited"></label>
              </div>
              <label class="fld"><span class="lab">note <span class="sub">optional, only you see it</span></span><input type="text" id="iv_note" placeholder="e.g. for the book club"></label>
              <div class="edit-row" style="margin-top:12px;"><button class="btn-primary" id="iv_save">create link</button><button class="btn-ghost" id="iv_cancel">cancel</button></div>
            </div>
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

  <div id="menu" role="menu"></div>
  <div id="dlgwrap"><div class="dlg" id="dlg" role="dialog" aria-modal="true"></div></div>
  <div id="peekmodal">
    <div class="modal-card peek-card">
      <div class="mhead"><h3 id="peek-title">user</h3><div class="seg" id="peek-tabs"><button data-pt="chats" class="on">chats</button><button data-pt="chars">characters</button></div><button class="x" data-close-peek>&times;</button></div>
      <div class="peek-body">
        <div id="peek-chats" class="peek-pane">
          <div class="peek-list" id="peek-convos"></div>
          <div class="peek-view" id="peek-view"></div>
        </div>
        <div id="peek-chars" class="peek-pane" style="display:none;"></div>
      </div>
    </div>
  </div>
  <div id="backdrop"></div>
  <div id="tipbox" class="tipbox" role="tooltip"></div>
  <div id="toast" role="status" aria-live="polite"></div>
"""

# HTML-escape + markdown helpers shared verbatim by the app and the public share viewer; composed
# into both pages at startup so the two copies can't drift apart. (Function declarations hoist, so
# placement within each script doesn't matter.)
MD_ESC_JS = r"""
function esc(s){return (s==null?"":String(s)).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function md(src){
  if(!src)return "";
  let s=src.replace(/\r\n/g,"\n").replace(/\r/g,"\n");
  const code=[];s=s.replace(/```[ \t]*([a-zA-Z0-9_+\-]*)\n?([\s\S]*?)```/g,(m,l,b)=>{code.push(b);return "@@C"+(code.length-1)+"@@";});
  const ic=[];s=s.replace(/`([^`\n]+)`/g,(m,c)=>{ic.push(c);return "@@I"+(ic.length-1)+"@@";});
  const inline=t=>{t=esc(t);
    t=t.replace(/\[([^\]]+)\]\((https?:[^\s)]+)\)/g,'<a href="$2" target="_blank" rel="noopener nofollow">$1</a>');
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
"""

PAGE_JS1 = r"""<script>
"use strict";
const $=s=>document.querySelector(s), $$=s=>Array.from(document.querySelectorAll(s));
let CFG=null, current=null, busy=false, activeController=null;
let convoCache=[], folderCache=[], selMode=false, selected=new Set();
let collapsed={};try{collapsed=JSON.parse(localStorage.getItem("oracle_collapsed"))||{};}catch(_){}
function toggleFolder(id){collapsed[id]=!collapsed[id];if(!collapsed[id])delete collapsed[id];
  try{localStorage.setItem("oracle_collapsed",JSON.stringify(collapsed));}catch(_){}
  renderTree();}
let pendingAtt=[];
let autoScroll=true;   // pinned to bottom; user scrolling up unpins until they return to the bottom

const ICON_SEND='<svg class="ico" viewBox="0 0 24 24"><path d="M5 12h14M13 6l6 6-6 6"/></svg>';
const ICON_STOP='<svg class="ico" viewBox="0 0 24 24"><rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor" stroke="none"/></svg>';
const ICON_THUMB='<svg class="ico" viewBox="0 0 24 24"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"/></svg>';
const ICON_MOON='<path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/>';
const ICON_SUN='<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>';

function fmtNum(n){return n==null?"?":Number(n).toLocaleString();}
function isAdmin(){return CFG&&CFG.is_admin;}
function toast(m,ms,cls){const t=$("#toast");t.textContent=m;t.classList.remove("err");if(cls)t.classList.add(cls);
  t.classList.add("show");clearTimeout(toast._t);toast._t=setTimeout(()=>t.classList.remove("show"),ms||2600);}
function copyText(t,okMsg){
  const ok=()=>toast(okMsg||"copied");
  const fail=()=>{ // fallback for non-secure (http://) contexts where the clipboard API is absent
    try{const ta=document.createElement("textarea");ta.value=t;ta.style.position="fixed";ta.style.opacity="0";
      document.body.appendChild(ta);ta.select();const r=document.execCommand("copy");ta.remove();
      r?ok():toast("copy failed — select the text manually","",'err');}
    catch(_){toast("copy failed — select the text manually","",'err');}
  };
  if(navigator.clipboard&&navigator.clipboard.writeText)navigator.clipboard.writeText(t).then(ok,fail);else fail();
}
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
// ---- custom palette: three picked colors; everything else derived so the scheme stays coherent
const CUSTOM_DEFAULT={bg:"#100c08",text:"#e8ddc5",accent:"#cf8a3c"};
function hex2rgb(h){h=(h||"").replace("#","");if(h.length===3)h=h.split("").map(c=>c+c).join("");const n=parseInt(h,16)||0;return [n>>16&255,n>>8&255,n&255];}
function rgb2hex(r,g,b){return "#"+[r,g,b].map(v=>Math.round(Math.max(0,Math.min(255,v))).toString(16).padStart(2,"0")).join("");}
function cmix(a,b,t){const A=hex2rgb(a),B=hex2rgb(b);return rgb2hex(A[0]+(B[0]-A[0])*t,A[1]+(B[1]-A[1])*t,A[2]+(B[2]-A[2])*t);}
function clum(h){const c=hex2rgb(h);return (0.2126*c[0]+0.7152*c[1]+0.0722*c[2])/255;}
function crgba(h,a){const c=hex2rgb(h);return "rgba("+c[0]+","+c[1]+","+c[2]+","+a+")";}
function customVars(c){
  const dark=clum(c.bg)<0.5;
  return {"--bg":c.bg,"--text":c.text,"--accent":c.accent,
    "--panel":cmix(c.bg,c.text,0.03),"--surface":cmix(c.bg,c.text,0.06),
    "--surface2":cmix(c.bg,c.text,0.10),"--surface3":cmix(c.bg,c.text,0.14),
    "--line":crgba(c.text,0.12),
    "--muted":cmix(c.text,c.bg,0.35),"--faint":cmix(c.text,c.bg,0.48),"--dim":cmix(c.text,c.bg,0.58),
    "--accent2":cmix(c.accent,c.text,0.35),"--accent-weak":crgba(c.accent,0.14),
    "--on-accent":clum(c.accent)>0.45?"#141210":"#f6f4ef",
    "--user":cmix(c.text,c.bg,0.2),"--bot":cmix(c.accent,c.text,0.15),
    "--code-bg":dark?cmix(c.bg,"#000000",0.35):cmix(c.bg,"#000000",0.05),
    "color-scheme":dark?"dark":"light"};
}
function getCustom(){try{return Object.assign({},CUSTOM_DEFAULT,JSON.parse(localStorage.getItem("oracle_custom")||"{}"));}catch(_){return Object.assign({},CUSTOM_DEFAULT);}}
function applyCustom(){
  const v=customVars(getCustom()),s=document.documentElement.style;
  for(const k in v)s.setProperty(k,v[k]);
  // pre-derived map for the pre-paint boot script, so a reload can apply it without any color math
  try{localStorage.setItem("oracle_custom_vars",JSON.stringify(v));}catch(_){}
}
function clearCustom(){
  const s=document.documentElement.style;
  Object.keys(customVars(CUSTOM_DEFAULT)).forEach(k=>s.removeProperty(k));
}
function applyPalette(p){
  if(!p)p="sepia";
  const d=document.documentElement;
  if(p==="custom"){d.removeAttribute("data-palette");applyCustom();}
  else{clearCustom();if(p==="sepia")d.removeAttribute("data-palette");else d.setAttribute("data-palette",p);}
  localStorage.setItem("oracle_palette",p);
  $$("#palseg button").forEach(b=>b.classList.toggle("on",b.dataset.pal===p));
  const w=$("#custwrap");if(w)w.style.display=(p==="custom")?"block":"none";
  if(p==="custom"){const c=getCustom();if($("#cp_bg")){$("#cp_bg").value=c.bg;$("#cp_text").value=c.text;$("#cp_accent").value=c.accent;}}
}
function curPalette(){return localStorage.getItem("oracle_palette")||"sepia";}
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
  const prev=document.activeElement;   // restore keyboard focus to the opener when the dialog closes
  const done=v=>{closeDlg();if(prev&&prev.focus){try{prev.focus();}catch(_){}}res(v);};
  d._resolve=done;
  const esckey=e=>{if(e.key==="Escape"){document.removeEventListener("keydown",esckey);done(null);}};
  document.addEventListener("keydown",esckey);
  w.onclick=e=>{if(e.target===w){document.removeEventListener("keydown",esckey);done(null);}};
  const f=d.querySelector("input,select,textarea")||d.querySelector("[data-ok]")||d.querySelector("button");
  if(f)f.focus();
  if(onmount)onmount(d,done);
});}
function uiConfirm(message,{title="Confirm",danger=false,ok="Confirm"}={}){
  return dialog('<h4>'+esc(title)+'</h4><p>'+esc(message)+'</p><div class="dlg-btns"><button class="btn-ghost" data-x>cancel</button><button class="'+(danger?"btn-danger":"btn-primary")+'" data-ok>'+esc(ok)+'</button></div>',
    (d,done)=>{d.querySelector("[data-x]").onclick=()=>done(false);d.querySelector("[data-ok]").onclick=()=>done(true);
      d.onkeydown=e=>{if(e.key==="Enter"){e.preventDefault();done(true);}};});
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

// ---------------- markdown: see MD_ESC_JS (shared with the public share viewer)
""" + MD_ESC_JS + r"""
// ---------------- config
function charById(id){return (CFG.characters||[]).find(c=>c.id===id)||null;}
function endpointName(id){return ((CFG.settings&&CFG.settings.endpoints)||[]).find(e=>e.id===id);}
async function loadConfig(){
  CFG=await api("GET","/api/config");
  CFG.characters=CFG.characters||[];CFG.models=CFG.models||[];CFG.presets=CFG.presets||[];folderCache=CFG.folders||[];
  $("#who-nm").textContent=CFG.me.username;$("#who-rl").textContent=CFG.me.role;
  $("#ac_persona").value=CFG.me.persona||"";
  if(!isAdmin())$("#d_endpoint_wrap").style.display="none";
  buildTabs();
  rebuildModelSelect($("#modelsel"),CFG.models);
}
function rebuildModelSelect(sel,models,value){
  const cur=value!==undefined?value:sel.value;sel.innerHTML="";const seen=new Set();
  (models||[]).forEach(m=>{if(seen.has(m))return;seen.add(m);const o=document.createElement("option");o.value=m;o.textContent=m;sel.appendChild(o);});
  if(cur&&!seen.has(cur)){const o=document.createElement("option");o.value=cur;o.textContent=cur;sel.insertBefore(o,sel.firstChild);}
  if(cur)sel.value=cur;
  if(!sel.options.length){const o=document.createElement("option");o.value="";o.textContent="(endpoint unreachable)";o.disabled=true;sel.appendChild(o);}
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
    head.onclick=e=>{if(e.target.classList.contains("fmenu"))return;toggleFolder(f.id);};
    head.tabIndex=0;
    head.addEventListener("keydown",e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();toggleFolder(f.id);}});
    head.querySelector(".fmenu").onclick=e=>{e.stopPropagation();folderMenu(e,f);};
    items.forEach(c=>body.appendChild(convoRow(c)));
    setupDrop(fol,f.id);tree.appendChild(fol);
  });
  const uf=document.createElement("div");uf.className="folder";uf.dataset.folder="";
  if(folderCache.length)uf.innerHTML='<div class="lbl" style="padding:10px 7px 4px;">unfiled</div>';
  unfiled.forEach(c=>uf.appendChild(convoRow(c)));
  setupDrop(uf,"");tree.appendChild(uf);
  // full-text results: conversations matching message bodies but not the title (title matches show above)
  let contentShown=0;
  if(q&&q.length>=2&&searchContent.length){
    const titleIds=new Set(convoCache.filter(match).map(c=>c.id));
    const extra=searchContent.filter(c=>!titleIds.has(c.id));
    if(extra.length){
      const grp=document.createElement("div");grp.className="folder";
      grp.innerHTML='<div class="lbl" style="padding:12px 7px 4px;">matches in messages</div>';
      extra.forEach(c=>grp.appendChild(searchRow(c,q)));
      tree.appendChild(grp);contentShown=extra.length;
    }
  }
  const titleCount=convoCache.filter(match).length;
  if(!convoCache.length)tree.insertAdjacentHTML("beforeend",'<div class="empty-list">no conversations yet</div>');
  else if(q&&!titleCount&&!contentShown&&!searchPending)tree.insertAdjacentHTML("beforeend",'<div class="empty-list">no matches</div>');
}
function convoRow(c){
  const d=document.createElement("div");
  d.className="convo"+(current&&c.id===current.id?" active":"")+(selected.has(c.id)?" checked":"");
  d.dataset.id=c.id;d.draggable=!selMode;
  const ch=charById(c.character_id);
  d.innerHTML='<span class="sel"></span><div class="ct">'+(c.mode==="compose"?'¶ ':'')+esc(c.title||"untitled")+'</div><div class="cm">'+
    (ch?esc((ch.avatar?ch.avatar+" ":"")+ch.name)+' · ':'')+esc(c.model||"")+' · '+relTime(c.updated)+'</div><button class="cmenu">&#8943;</button>';
  d.onclick=e=>{if(e.target.classList.contains("cmenu"))return;
    if(selMode){toggleSel(c.id,d);return;}openConvo(c.id);closeSidebar();};
  d.tabIndex=0;d.setAttribute("role","button");
  d.addEventListener("keydown",e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();
    if(selMode){toggleSel(c.id,d);}else{openConvo(c.id);closeSidebar();}}});
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
    '<button data-a="share">share &hellip;</button>'+
    '<button data-a="exmd">export markdown</button><button data-a="exjson">export json</button>'+
    '<div class="sep"></div><div class="mhint">move to folder</div>'+(fopts||'<div class="mhint" style="color:var(--dim)">no folders</div>')+
    '<div class="sep"></div><button class="danger" data-a="del">delete</button>',
    m=>{
      m.querySelectorAll("[data-mv]").forEach(b=>b.onclick=async()=>{hideMenu();const c2=await api("POST","/api/conversations/"+c.id+"/settings",{folder_id:b.dataset.mv||null});if(current&&current.id===c.id)current=c2;refreshList();toast("moved");});
      m.querySelector('[data-a="open"]').onclick=()=>{hideMenu();openConvo(c.id);closeSidebar();};
      m.querySelector('[data-a="rename"]').onclick=async()=>{hideMenu();const t=await uiPrompt("Rename conversation",c.title||"",{title:"Rename"});if(t!=null&&t.trim()){const c2=await api("POST","/api/conversations/"+c.id+"/settings",{title:t.trim()});if(current&&current.id===c.id){current=c2;syncBar();}refreshList();}};
      m.querySelector('[data-a="share"]').onclick=()=>{hideMenu();shareDialog(c);};
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

async function shareDialog(c){
  let st;
  try{st=await api("GET","/api/conversations/"+c.id+"/share");}catch(e){toast(e.message);return;}
  dialog('<h4>Share conversation</h4><div id="shareBody"></div>',(d,done)=>{
    const body=d.querySelector("#shareBody");
    function render(){
      if(st&&st.shared){
        const meta=(st.views?(st.views+' view'+(st.views===1?'':'s')+' · '):'')+'snapshot saved '+esc(relTime(st.updated));
        body.innerHTML=
          '<p>Anyone with this link reads a frozen, stripped-down copy of this conversation. Your system prompt and any later messages stay private.</p>'+
          '<div class="sharelink"><input id="shareUrl" readonly value="'+esc(st.url)+'"><button class="btn-primary" data-s="copy">copy</button></div>'+
          '<div class="share-meta">'+meta+'</div>'+
          '<div class="share-actions">'+
            '<a class="linkbtn" href="'+esc(st.url)+'" target="_blank" rel="noopener">open ↗</a>'+
            '<button class="linkbtn" data-s="update">update snapshot</button>'+
            '<button class="linkbtn danger" data-s="unshare">unshare</button>'+
            '<span class="grow"></span>'+
            '<button class="btn-primary" data-x>done</button>'+
          '</div>';
        const inp=body.querySelector("#shareUrl");
        body.querySelector('[data-s="copy"]').onclick=()=>{inp.focus();inp.select();copyText(st.url,"link copied");};
        body.querySelector('[data-s="update"]').onclick=async(e)=>{e.target.disabled=true;try{st=await api("POST","/api/conversations/"+c.id+"/share");render();toast("snapshot updated to the current thread");}catch(err){toast(err.message);render();}};
        body.querySelector('[data-s="unshare"]').onclick=async(e)=>{e.target.disabled=true;try{st=await api("DELETE","/api/conversations/"+c.id+"/share");render();toast("link disabled");}catch(err){toast(err.message);render();}};
        body.querySelector('[data-x]').onclick=()=>done(null);
      }else{
        body.innerHTML=
          '<p>Publish a public link to a frozen snapshot of this conversation’s current thread — a clean, gorgeous, read-only page anyone can open, no sign-in needed. Your system prompt is never shared, and new messages won’t appear unless you update the snapshot.</p>'+
          '<div class="dlg-btns"><button class="btn-ghost" data-x>cancel</button><button class="btn-primary" data-s="create">create link</button></div>';
        body.querySelector('[data-s="create"]').onclick=async(e)=>{e.target.disabled=true;try{st=await api("POST","/api/conversations/"+c.id+"/share");render();toast("share link ready");}catch(err){toast(err.message);e.target.disabled=false;}};
        body.querySelector('[data-x]').onclick=()=>done(null);
      }
    }
    render();
  });
}

async function exportChat(id,fmt){
  if(fmt==="json"){
    // full tree: every branch, plus ratings / reasoning / tool steps (same shape as export-all)
    const full=await api("GET","/api/conversations/"+id+"/export");
    const safe=(full.title||"chat").replace(/[^a-z0-9]+/gi,"-").slice(0,50).replace(/^-|-$/g,"")||"chat";
    download(safe+".json",JSON.stringify(full,null,2),"application/json");return;
  }
  const c=await api("GET","/api/conversations/"+id);   // markdown: the active thread as prose
  const safe=(c.title||"chat").replace(/[^a-z0-9]+/gi,"-").slice(0,50).replace(/^-|-$/g,"")||"chat";
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
  if(current.tools)bits.push("◇ tools");
  if(current.mode==="compose")bits.push("¶ composition");
  $("#submeta").textContent=bits.join("  ·  ");
  updateCtx();
}
function nearBottom(){const l=$("#log");return l.scrollHeight-l.scrollTop-l.clientHeight<40;}
function scrollDown(){const l=$("#log");l.scrollTop=l.scrollHeight;}
function renderConvo(opts){
  opts=opts||{};
  if(current&&current.mode==="compose"){renderCompose();return;}
  $("#composer").style.display="";
  const stick=(opts.stick!==undefined)?opts.stick:autoScroll;
  const prev=$("#log").scrollTop;syncBar();
  const log=$("#log");log.innerHTML="";
  const wrap=document.createElement("div");wrap.className="wrap";
  const msgs=(current.messages||[]).filter(m=>m.role==="user"||m.role==="assistant");
  if(!msgs.length)wrap.innerHTML='<div class="empty"><div class="glyph">&rsaquo;_</div><h2>new conversation</h2><p>say something to begin</p></div>';
  else current.messages.forEach((m,i)=>{if(m.role==="user"||m.role==="assistant"||m.role==="tool")wrap.appendChild(msgEl(m,i));});
  addCodeCopy(wrap);
  log.appendChild(wrap);
  if(stick){autoScroll=true;scrollDown();}else log.scrollTop=prev;
}
function addCodeCopy(root){
  root.querySelectorAll(".bubble pre").forEach(pre=>{
    if(pre.querySelector(".copy-code"))return;
    const b=document.createElement("button");b.className="copy-code";b.type="button";b.textContent="copy";
    b.onclick=e=>{e.stopPropagation();const code=pre.querySelector("code");copyText(code?code.textContent:pre.textContent,"code copied");};
    pre.appendChild(b);
  });
}
function paramsTipText(m){
  const x=m&&m.meta;if(!x)return "";
  const p=x.params,lines=[];
  if(x.preset)lines.push("preset · "+x.preset);
  if(p&&typeof p==="object"){
    const order=(CFG.param_specs||[]).map(s=>s.key),keys=[];
    order.forEach(k=>{if(p[k]!=null)keys.push(k);});
    Object.keys(p).forEach(k=>{if(keys.indexOf(k)<0&&p[k]!=null)keys.push(k);});
    if(keys.length){if(!x.preset)lines.push("sampler parameters");keys.forEach(k=>lines.push(k+": "+p[k]));}
  }
  return lines.join("\n");
}
function metaLine(m){
  if(!m.meta)return "";const x=m.meta,b=[];
  if(m.model)b.push('<span class="pill k">'+esc(m.model)+'</span>');
  if(x.elapsed_ms!=null)b.push('<span class="pill">'+(x.elapsed_ms/1000).toFixed(1)+'s</span>');
  if(x.completion_tokens!=null)b.push('<span class="pill" title="'+(x.tokens_est?'estimated':'reported')+'">'+(x.tokens_est?"~":"")+fmtNum(x.completion_tokens)+' tok</span>');
  if(x.tps)b.push('<span class="pill">'+x.tps+' tok/s</span>');
  if(x.cached_tokens!=null&&x.prompt_tokens)b.push('<span class="pill" title="prompt tokens served from the model server’s prefix cache">'+Math.round(100*x.cached_tokens/x.prompt_tokens)+'% cache</span>');
  let tip=[];if(x.ttft_ms!=null)tip.push("ttft "+(x.ttft_ms/1000).toFixed(2)+"s");if(x.prompt_tokens!=null)tip.push("prompt "+fmtNum(x.prompt_tokens)+" tok");
  return '<div class="meta" title="'+esc(tip.join(" · "))+'">'+b.join("")+'</div>';
}
function sibNav(m){
  if(!m.sib_count||m.sib_count<2)return "";
  const i=m.sib_index;
  return '<span class="sib" title="branch '+(i+1)+' of '+m.sib_count+'"><button data-sib="prev"'+(i<=0?' disabled':'')+'>&lsaquo;</button><span class="n">'+(i+1)+' / '+m.sib_count+'</span><button data-sib="next"'+(i>=m.sib_count-1?' disabled':'')+'>&rsaquo;</button></span>';
}
function msgEl(m,i){
  if(m.role==="tool")return toolResultEl(m);
  // intermediate assistant turn that only requested a tool (no prose): hide; the tool step carries it
  if(m.role==="assistant"&&m.tool&&m.tool.tool_calls&&!(m.content||"").trim()){const s=document.createElement("div");s.style.display="none";return s;}
  const d=document.createElement("div");d.className="msg "+m.role;d.dataset.id=m.id;
  const ch=charById(current.character_id);
  const role=m.role==="user"?(CFG.me.username||"you"):(ch?ch.name:"oracle");
  let reason=m.reasoning?'<details class="reason"><summary>reasoning</summary><div class="rbody">'+esc(m.reasoning)+'</div></details>':"";
  const edited=m.edited?' · edited':'';
  const mark=m.rating?'<span class="ratemark '+(m.rating>0?'up':'down')+'" title="'+(m.rating>0?'rated good':'rated bad')+'">'+(m.rating>0?'&#9650;':'&#9660;')+'</span>':'';
  let disp=m.content||"";
  if(m.role==="assistant"&&disp.indexOf("<tool_call>")>=0)disp=stripTC(disp)||"_(tried to use a web tool — enable **web tools** in tune to allow it)_";
  const body=m.role==="assistant"?'<div class="bubble">'+md(disp)+'</div>':'<div class="bubble raw">'+esc(m.content)+'</div>';
  const rateBtns=m.role==="assistant"?'<button data-act="up" class="rate up'+(m.rating>0?' on':'')+'" title="good response (saved for RLHF)">'+ICON_THUMB+'</button><button data-act="down" class="rate down'+(m.rating<0?' on':'')+'" title="bad response (saved for RLHF)">'+ICON_THUMB+'</button>':'';
  d.innerHTML='<div class="head"><span class="role">'+esc(role)+'</span>'+sibNav(m)+'<span class="tm">'+esc(fmtTime(m.ts)+edited)+'</span>'+mark+'</div>'+
    reason+attsHtml(m)+body+metaLine(m)+
    '<div class="actions"><button data-act="copy">copy</button>'+
    (m.role==="assistant"?'<button data-act="raw">raw</button><button data-act="continue" title="keep generating from the end of this message (prefill)">continue</button><button data-act="regen">regenerate</button><button data-act="regenwith" title="regenerate this reply with a different model (for side-by-side comparison via the branch switcher)">model&#8230;</button>'+rateBtns:'')+
    '<button data-act="edit">edit</button><button data-act="del" class="danger">delete</button></div>';
  d.querySelectorAll(".actions button").forEach(b=>b.onclick=()=>handleAction(b.dataset.act,m,d,b));
  d.querySelectorAll("[data-sib]").forEach(b=>b.onclick=()=>{if(b.disabled)return;switchSibling(m.siblings[m.sib_index+(b.dataset.sib==="next"?1:-1)]);});
  const mk=d.querySelector(".meta .pill.k"),mtip=paramsTipText(m);
  if(mk&&mtip){mk.classList.add("has-tip");bindTip(mk,mtip);}
  d.addEventListener("click",e=>{   // tap a message (mobile) to reveal its actions
    if(e.target.closest("button,a,input,textarea,summary,.edit-wrap"))return;
    if(String(window.getSelection?window.getSelection():"").trim())return;
    d.classList.toggle("revealed");
  });
  return d;
}
function handleAction(act,m,d,btn){
  if(act==="copy"){copyText(m.content);return;}
  if(act==="raw"){const b=d.querySelector(".bubble");if(b.classList.contains("raw")){b.classList.remove("raw");b.innerHTML=md(m.content);addCodeCopy(d);btn.textContent="raw";}else{b.classList.add("raw");b.textContent=m.content;btn.textContent="markdown";}return;}
  if(act==="edit"){startEdit(m,d);return;}
  if(act==="regen"){regenerate(m);return;}
  if(act==="regenwith"){regenWithModel(m);return;}
  if(act==="continue"){continueMsg(m);return;}
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

// ---------------- attachments + context meter
function fmtK(n){n=Math.round(n||0);if(n<1000)return ""+n;if(n<10000)return (n/1000).toFixed(1).replace(/\.0$/,'')+"k";return Math.round(n/1000)+"k";}
function estTok(s){return Math.max(1,Math.round((s||"").length/4));}
function stripTC(s){return (s||"").replace(/<tool_call>[\s\S]*?<\/tool_call>/g,"").replace(/<tool_call>[\s\S]*$/,"").trim();}
function fileToB64(file){return new Promise((res,rej)=>{const r=new FileReader();r.onload=()=>res(String(r.result).split(",")[1]||"");r.onerror=()=>rej(new Error("read failed"));r.readAsDataURL(file);});}
function pendingReady(){return pendingAtt.filter(a=>!a.busy&&!a.error&&(a.text||a.image));}
function clearPending(){pendingAtt=[];renderPending();}
function renderPending(){
  const box=$("#pending");if(!box)return;box.innerHTML="";
  pendingAtt.forEach((a,i)=>{
    const c=document.createElement("div");c.className="achip"+(a.busy?" busy":"")+(a.error?" err":"");
    const tail=a.busy?'<span class="toolspin"></span>':(a.error?'<span class="tk">'+esc(a.error)+'</span>':'<span class="tk">~'+fmtK(a.tokens_est)+' tok</span>');
    c.innerHTML='<span class="nm">'+esc(a.name)+'</span>'+tail+'<button class="rm" title="remove">&times;</button>';
    c.querySelector(".rm").onclick=()=>{pendingAtt.splice(i,1);renderPending();};
    box.appendChild(c);
  });
  updateCtx();
}
async function uploadFiles(files){
  for(const f of files){
    const item={name:f.name,busy:true};pendingAtt.push(item);renderPending();
    if(/^image\//.test(f.type||"")){
      try{
        if(f.size>6*1024*1024)throw new Error("image over 6 MB");
        const url=await new Promise((res,rej)=>{const r=new FileReader();r.onload=()=>res(String(r.result));r.onerror=()=>rej(new Error("read failed"));r.readAsDataURL(f);});
        Object.assign(item,{busy:false,image:url,tokens_est:1024});
        const vm=CFG.vision_models||[];
        if(current&&vm.indexOf(current.model)<0)toast("note: "+(current.model||"this model")+" isn\u2019t marked vision-capable \u2014 the image will only be mentioned by name",6000);
      }catch(e){item.busy=false;item.error=(e.message||"failed").slice(0,40);toast(f.name+": "+(e.message||"failed"));}
      renderPending();continue;
    }
    try{
      const data=await fileToB64(f);
      const r=await api("POST","/api/extract",{name:f.name,data});
      Object.assign(item,{busy:false,text:r.text,tokens_est:r.tokens_est,chars:r.chars,truncated:r.truncated});
      if(r.truncated)toast(f.name+": truncated to fit");
    }catch(e){item.busy=false;item.error=(e.message||"failed").slice(0,40);toast(f.name+": "+(e.message||"failed"));}
    renderPending();
  }
}
function ctxLimit(){const m=current&&current.model;return (CFG.model_contexts&&CFG.model_contexts[m])||CFG.default_context||8192;}
function ctxUsed(){
  if(!current||!current.messages)return 0;
  for(let i=current.messages.length-1;i>=0;i--){const m=current.messages[i];
    if(m.role==="assistant"&&m.meta&&m.meta.prompt_tokens!=null)return m.meta.prompt_tokens+(m.meta.completion_tokens||0);}
  let t=current.system?estTok(current.system):0;
  current.messages.forEach(m=>{t+=estTok(m.content);(m.attachments||[]).forEach(a=>t+=estTok(a.text));});
  return t;
}
function updateCtx(){
  const el=$("#ctxmeter");if(!el)return;
  if(!current){el.style.display="none";return;}
  const lim=ctxLimit();let used=ctxUsed();
  const draft=($("#input")&&$("#input").value)||"";used+=estTok(draft);
  pendingReady().forEach(a=>used+=(a.tokens_est||estTok(a.text)));
  const pct=Math.min(100,Math.round(used/lim*100));
  el.style.display="flex";
  el.classList.toggle("warn",pct>=70&&pct<90);el.classList.toggle("hot",pct>=90);
  el.querySelector("i").style.width=pct+"%";
  el.querySelector(".ctxtxt").textContent=fmtK(used)+" / "+fmtK(lim);
  el.title="context: "+used.toLocaleString()+" / "+lim.toLocaleString()+" tokens ("+pct+"%)";
}
function attsHtml(m){
  if(!m.attachments||!m.attachments.length)return "";
  return '<div class="atts">'+m.attachments.map(a=>{
    if(a.image)return '<img class="attimg" src="'+esc(a.image)+'" alt="'+esc(a.name||"image")+'" title="'+esc(a.name||"")+'">';
    const tx=a.text||"";return '<details><summary>&#9636; '+esc(a.name)+' <span style="color:var(--faint)">~'+fmtK(estTok(tx))+' tok</span></summary><pre class="atxt">'+esc(tx.slice(0,20000))+(tx.length>20000?"\n…":"")+'</pre></details>';}).join("")+'</div>';
}
function toolResultEl(m){
  const t=m.tool||{},ui=t.ui||{},ok=ui.ok!==false;
  const d=document.createElement("div");d.className="msg assistant";d.dataset.id=m.id;
  const label=esc((t.name||"tool")+(ui.summary?" — "+ui.summary:""));
  const urlLine=ui.url?'<a class="turl" href="'+esc(ui.url)+'" target="_blank" rel="noopener">'+esc(ui.url)+'</a>\n\n':'';
  d.innerHTML='<details class="toolstep'+(ok?"":" err")+'"><summary>&#9671; <span class="tlabel">'+label+'</span></summary><div class="tbody">'+urlLine+esc((m.content||"").slice(0,20000))+'</div></details>';
  return d;
}
// ---------------- streaming
function liveWrap(){let w=$("#log").querySelector(".wrap");if(!w){const l=$("#log");l.innerHTML="";w=document.createElement("div");w.className="wrap";l.appendChild(w);}const e=w.querySelector(".empty");if(e)e.remove();return w;}
function appendLive(){
  const stick=autoScroll;const wrap=liveWrap();
  const ch=charById(current.character_id);const role=ch?ch.name:"oracle";
  const d=document.createElement("div");d.className="msg assistant tmpnode";
  // reasoning streams into a real <details>, open, in its finished italic-quote form; it auto-collapses
  // when the answer begins (still re-expandable).
  d.innerHTML='<div class="head"><span class="role">'+esc(role)+'</span><span class="tm"></span></div>'+
    '<details class="reason" open style="display:none"><summary>reasoning</summary><div class="rbody"></div></details>'+
    '<div class="bubble"><span class="typing"><i></i><i></i><i></i></span></div>';
  wrap.appendChild(d);if(stick)scrollDown();
  return {bubble:d.querySelector(".bubble"),reasonBox:d.querySelector(".reason"),reason:d.querySelector(".rbody"),
          started:false,reasonCollapsed:false,
          acc:"",text:"",shown:0,hasTC:false,racc:"",rshown:0,stableLen:0,ended:false};
}
function appendThinking(){
  const w=liveWrap();const d=document.createElement("div");d.className="msg assistant thinking tmpnode";
  d.innerHTML='<div class="thinkrow"><span class="thinkdot"></span><span class="tlabel">thinking…</span></div>';
  w.appendChild(d);if(autoScroll)scrollDown();return d;
}
function showNotice(msg){
  // non-fatal context-window warning, shown inline just above the incoming reply
  const w=liveWrap();let n=w.querySelector(".ctx-notice");
  if(!n){n=document.createElement("div");n.className="ctx-notice tmpnode";}
  n.textContent="⚠ "+msg;w.appendChild(n);if(autoScroll)scrollDown();
}
function appendToolStep(tc){
  const w=liveWrap();const d=document.createElement("div");d.className="msg assistant tmpnode";
  const url=(tc.args&&tc.args.url)?tc.args.url:"";
  d.innerHTML='<div class="toolstep"><div class="summary" style="padding:8px 11px;display:flex;gap:8px;align-items:center;font-family:var(--mono);font-size:11px;color:var(--accent2)"><span class="toolspin"></span> <span class="tlabel">'+esc(tc.name||"tool")+(url?" "+esc(url):"")+'…</span></div></div>';
  w.appendChild(d);return d;
}
function updateToolStep(el,tr){
  if(!el)return;const ok=tr.ok!==false;
  const label=esc((tr.name||"tool")+(tr.summary?" — "+tr.summary:""));
  const urlLine=tr.url?'<a class="turl" href="'+esc(tr.url)+'" target="_blank" rel="noopener">'+esc(tr.url)+'</a>':esc(tr.summary||"");
  const note=tr.chars?"\n\n["+Number(tr.chars).toLocaleString()+" chars fetched]":"";
  el.innerHTML='<details class="toolstep'+(ok?"":" err")+'"><summary>&#9671; <span class="tlabel">'+label+'</span></summary><div class="tbody">'+urlLine+note+'</div></details>';
}
async function streamRequest(path,body,handlers,signal){
  const r=await fetch(path,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body),signal});
  if(r.status===401){location.href="/login";throw new Error("session expired");}
  if(!r.ok){const j=await r.json().catch(()=>({}));throw new Error(j.error||("HTTP "+r.status));}
  const reader=r.body.getReader(),dec=new TextDecoder();let buf="",result=null,errMsg=null;
  while(true){const {value,done}=await reader.read();if(done)break;buf+=dec.decode(value,{stream:true});
    let nl;while((nl=buf.indexOf("\n"))>=0){const line=buf.slice(0,nl);buf=buf.slice(nl+1);if(!line.trim())continue;
      let o;try{o=JSON.parse(line);}catch(_){continue;}
      if(o.error)errMsg=o.error;   // keep reading: a `done` with the persisted partial may follow
      else if(o.delta!==undefined)handlers.onDelta(o.delta);
      else if(o.reasoning!==undefined)handlers.onReason(o.reasoning);
      else if(o.status!==undefined)handlers.onStatus&&handlers.onStatus(o.status);
      else if(o.tool_call)handlers.onToolCall&&handlers.onToolCall(o.tool_call);
      else if(o.tool_result)handlers.onToolResult&&handlers.onToolResult(o.tool_result);
      else if(o.notice!==undefined)handlers.onNotice&&handlers.onNotice(o.notice);
      else if(o.done)result=o;}}
  if(errMsg){const ex=new Error(errMsg);ex.convo=result&&result.convo;throw ex;}
  return result;
}
// ---------------- paced live rendering
// Tokens land in bursts (sampler jitter, then network chunking), so painting each delta the moment
// it arrives looks jittery. Everything received goes into an accumulator and a single rAF loop
// reveals it at a smoothed rate — the buffer drains exponentially, so the text keeps flowing at a
// near-constant pace through a stall and speeds up (never falls behind) when a burst lands.
// The markdown is rendered in two pieces: blocks that can no longer change are rendered once into
// .mds and left alone, and only the block still being written is re-parsed each frame (.mdt). Long
// replies therefore stay at a flat cost per frame, and settled text never blinks or reflows.
function liveMd(s){   // close an unterminated fence so a half-written code block reads as code
  if(((s.match(/```/g)||[]).length)%2)s+="```";   // no newline: it would push the caret to a new line
  try{return md(s);}catch(_){return "<p>"+esc(s)+"</p>";}
}
function liveCaret(L){
  const host=L.tail.lastElementChild;
  if(!host){L.tail.appendChild(L.caret);return;}
  if(host.tagName==="UL"||host.tagName==="OL")(host.lastElementChild||host).appendChild(L.caret);
  else if(host.tagName==="PRE")(host.querySelector("code")||host).appendChild(L.caret);
  else host.appendChild(L.caret);
}
function liveSettle(L,txt){
  // Settle everything up to the last blank line — but never inside a fence, and never where the
  // next block would continue a list or quote (splitting one would restart the <ul> and shift the
  // layout). A very long unsettleable tail is settled anyway: one small shift beats dropped frames.
  // Only the newly-settled segment is parsed and appended, so already-settled DOM is never touched.
  const force=txt.length-L.stableLen>8000;
  let idx=txt.lastIndexOf("\n\n");
  while(idx>=L.stableLen){
    const seg=txt.slice(L.stableLen,idx+2),rest=txt.slice(idx+2);
    if(rest.trim()&&((seg.match(/```/g)||[]).length%2===0)&&
       (force||!/^\s*(?:[-*+]\s|\d+[.)]\s|>)/.test(rest))){
      L.stableLen=idx+2;L.stable.insertAdjacentHTML("beforeend",liveMd(seg));return;
    }
    idx=txt.lastIndexOf("\n\n",idx-1);
  }
}
function liveRender(L){
  if(!L.stable){   // first painted character: swap the typing dots for the two render slots
    L.bubble.innerHTML='<div class="mds"></div><div class="mdt"></div>';
    L.stable=L.bubble.querySelector(".mds");L.tail=L.bubble.querySelector(".mdt");
    L.caret=document.createElement("span");L.caret.className="caret";
  }
  const txt=L.text.slice(0,L.shown);
  liveSettle(L,txt);
  L.tail.innerHTML=liveMd(txt.slice(L.stableLen));
  if(!L.ended)liveCaret(L);
}
let _pacer=null,_pacerRaf=0,_pacerT=0;
function pacerStop(){if(_pacerRaf)cancelAnimationFrame(_pacerRaf);_pacerRaf=0;_pacer=null;_pacerT=0;}
function pacerStart(L){if(_pacer===L&&_pacerRaf)return;_pacer=L;_pacerT=0;
  if(!_pacerRaf)_pacerRaf=requestAnimationFrame(pacerTick);}
function pacerTick(ts){
  _pacerRaf=0;const L=_pacer;if(!L)return;
  let dt=_pacerT?ts-_pacerT:16;_pacerT=ts;
  if(dt>250)dt=250;   // hidden tab / long task: catch up in one step rather than crawling for ages
  let drew=false;
  const take=(pending)=>L.ended?pending:Math.min(pending,Math.max(1,Math.ceil(pending*dt/110)));
  const rp=L.racc.length-L.rshown;
  if(rp>0){const n=take(rp);
    if(L.reason)L.reason.appendChild(document.createTextNode(L.racc.substr(L.rshown,n)));
    L.rshown+=n;drew=true;}
  const p=L.text.length-L.shown;
  if(p>0){L.shown+=take(p);liveRender(L);drew=true;}
  else if(L.shown>L.text.length){L.shown=L.text.length;liveRender(L);drew=true;}   // stripTC shrank it
  if(drew&&autoScroll)scrollDown();   // inline: scheduling another frame would always lag one behind
  if(L.ended&&L.shown>=L.text.length&&L.rshown>=L.racc.length){pacerStop();return;}
  _pacerRaf=requestAnimationFrame(pacerTick);
}
function liveFinish(L){   // stream over: paint the remainder at once and drop the caret
  if(!L){pacerStop();return;}
  L.ended=true;
  L.rshown=L.racc.length;L.shown=L.text.length;
  if(L.reason&&L.racc)L.reason.textContent=L.racc;
  if(L.started)liveRender(L);
  pacerStop();
  if(autoScroll)scrollDown();
}
function maybeAskNotif(){
  if(!("Notification" in window))return;
  if(Notification.permission==="default"&&!localStorage.getItem("oracle_notif_asked")){
    localStorage.setItem("oracle_notif_asked","1");
    try{Notification.requestPermission();}catch(_){}
  }
}
function notifyDone(ok){
  if(!("Notification" in window)||Notification.permission!=="granted"||!document.hidden)return;
  try{
    const n=new Notification("ORACLE",{body:ok?((current&&current.title)||"reply finished"):"generation failed",tag:"oracle-done"});
    n.onclick=()=>{try{window.focus();}catch(_){}n.close();};
  }catch(_){}
}
async function streamTurn(body){
  let live=null,think=null;const toolEls={};
  function clearThink(){if(think){think.remove();think=null;}}
  function showThink(t){if(live)return;if(!think)think=appendThinking();const l=think.querySelector(".tlabel");if(l&&t)l.textContent=t;if(autoScroll)scrollDown();}
  function bubble(){clearThink();if(!live)live=appendLive();return live;}
  // continue/prefill: seed the live bubble with the existing text so generation visibly resumes from it
  if(body.continue_id){const L=bubble();L.started=true;
    // acc is re-seeded from the stripped text: deltas extend it directly, so a trimmed prefix must
    // not leave `shown` pointing past the end and briefly un-paint the tail
    L.acc=L.text=stripTC(body.prefix||"");L.shown=L.text.length;liveRender(L);if(autoScroll)scrollDown();}
  else showThink("thinking…");
  const c=new AbortController();activeController=c;
  try{const res=await streamRequest("/api/conversations/"+current.id+"/stream",body,{
      onStatus:s=>{if(!s){clearThink();return;}showThink(s);},
      onDelta:d=>{const L=bubble();if(!L.started){L.started=true;
          if(L.reasonBox&&L.racc&&!L.reasonCollapsed){L.reasonBox.open=false;L.reasonCollapsed=true;}}  // answer began -> fold the reasoning
        L.acc+=d;
        // a tool call the server didn't intercept must not show as markup; scanning only the new
        // tail keeps this O(delta) instead of O(reply) per token
        if(!L.hasTC&&L.acc.slice(-(d.length+11)).indexOf("<tool_call>")>=0)L.hasTC=true;
        L.text=L.hasTC?stripTC(L.acc):L.acc;
        pacerStart(L);},
      onReason:r=>{const L=bubble();L.racc+=r;if(L.reasonBox){L.reasonBox.style.display="";}
        pacerStart(L);},
      onToolCall:tc=>{liveFinish(live);clearThink();const el=appendToolStep(tc);if(tc.id)toolEls[tc.id]=el;live=null;if(autoScroll)scrollDown();},
      onToolResult:tr=>{updateToolStep(toolEls[tr.id],tr);if(autoScroll)scrollDown();},
      onNotice:msg=>{showNotice(msg);toast("context window reached",6000);}
    },c.signal);
    clearThink();liveFinish(live);
    return {res,stopped:false};
  }catch(e){clearThink();liveFinish(live);if(e.name==="AbortError")return {res:null,stopped:true};throw e;}
  finally{activeController=null;clearTimeout(_stopFallback);pacerStop();}
}
function setBusy(b){busy=b;const s=$("#send");s.classList.toggle("stop",b);s.innerHTML=b?ICON_STOP:ICON_SEND;s.title=b?"stop":"send";}
let _stopFallback=null;
function stopStream(){
  if(!busy||!current)return;
  // Cooperative stop: the server finalizes the partial, sets it as the active branch, and sends
  // `done` over the open stream — so a stopped regenerate keeps the new branch instead of snapping
  // back to the previous version. Abort is only a fallback if a stalled stream never winds down.
  api("POST","/api/conversations/"+current.id+"/stop").catch(()=>{});
  if(activeController){const c=activeController;clearTimeout(_stopFallback);_stopFallback=setTimeout(()=>{try{c.abort();}catch(_){}},4000);}
}
function showStreamError(msg){
  const w=liveWrap();const n=document.createElement("div");n.className="ctx-notice";
  n.textContent="✕ "+msg;w.appendChild(n);if(autoScroll)scrollDown();
}
function appendFinal(){
  const wrap=$("#log").querySelector(".wrap");
  if(!wrap){renderConvo();return;}
  wrap.querySelectorAll('.tmpnode,[data-id="tmp"]').forEach(n=>n.remove());
  const have=new Set(Array.from(wrap.querySelectorAll("[data-id]")).map(n=>n.dataset.id));
  let added=false;
  (current.messages||[]).forEach((m,i)=>{
    if(!["user","assistant","tool"].includes(m.role)||have.has(m.id))return;
    wrap.appendChild(msgEl(m,i));added=true;
  });
  if(added)addCodeCopy(wrap);
  syncBar();
  if(autoScroll)scrollDown();
}
async function runStream(body,optimistic,restore){
  if(busy||!current)return;setBusy(true);
  maybeAskNotif();
  if(optimistic)optimistic();
  try{const {res,stopped}=await streamTurn(body);
    if(res&&res.convo){
      current=res.convo;
      // plain sends append in place instead of rebuilding the whole DOM (matters on long chats);
      // branch-structure changes (regen/edit/continue) still re-render fully
      const plain=!body.regenerate_id&&!body.edit_user_id&&!body.continue_id&&!body.regenerate;
      if(plain)appendFinal();else renderConvo();
    }else await openConvo(current.id);
    if(!stopped)notifyDone(true);
    refreshList();
    // the server generates the real title in the background after `done` on a first exchange —
    // poll it in shortly (twice, in case the title model is cold) instead of blocking the stream
    if(res&&res.convo&&(res.convo.messages||[]).filter(m=>m.role==="assistant").length===1){
      const tcid=res.convo.id;
      const pickup=async()=>{try{const c2=await api("GET","/api/conversations/"+tcid);
        if(current&&current.id===tcid&&c2.title!==current.title){current.title=c2.title;syncBar();}refreshList();}catch(_){}};
      setTimeout(pickup,4000);setTimeout(pickup,15000);
    }
  }catch(e){
    const kept=!!e.convo;   // server persisted the turn (incl. any partial reply) before failing
    if(kept){current=e.convo;renderConvo({stick:false});}
    else{try{await openConvo(current.id);}catch(_){}}
    showStreamError(e.message||"stream failed");
    notifyDone(false);
    if(!kept&&restore)restore();   // nothing saved server-side -> put the draft back
    toast(e.message||"stream failed",4000,'err');
    refreshList();
  }
  finally{setBusy(false);$("#input").focus();}
}
async function send(){
  const text=$("#input").value.trim();const atts=pendingReady();
  if((!text&&!atts.length)||busy||send._creating)return;
  if(pendingAtt.some(a=>a.busy)){toast("still reading a file…");return;}
  if(!current){
    send._creating=true;   // a double-click during the await must not create two conversations
    try{current=await api("POST","/api/conversations",{system:CFG.default_system,model:$("#modelsel").value||CFG.default_model});}
    catch(e){toast(e.message,4000,'err');return;}
    finally{send._creating=false;}
  }
  $("#input").value="";$("#input").style.height="auto";saveDraft();
  const sendAtts=atts.map(a=>a.image?{name:a.name,image:a.image}:{name:a.name,text:a.text});
  const restorable=pendingAtt.slice();
  clearPending();
  runStream({content:text,attachments:sendAtts},
    ()=>{current.messages.push({id:"tmp",role:"user",content:text,attachments:sendAtts,ts:new Date().toISOString()});renderConvo({stick:true});},
    ()=>{$("#input").value=text;pendingAtt=restorable;renderPending();$("#input").dispatchEvent(new Event("input"));});
}
function regenerate(m){
  const idx=current.messages.findIndex(x=>x.id===m.id);
  runStream({regenerate_id:m.id},()=>{if(idx>=0)current.messages=current.messages.slice(0,idx);renderConvo({stick:true});});
}
async function regenWithModel(m){
  // one-off model override: the reply forks as a sibling branch, so the switcher becomes an A/B view
  const models=(CFG.models||[]).filter(Boolean);
  if(!models.length){toast("no models available");return;}
  const pick=await uiSelect("Regenerate this reply with:",models.map(v=>({value:v,label:v+(v===current.model?"  (current)":"")})),{title:"Regenerate with model"});
  if(!pick)return;
  const idx=current.messages.findIndex(x=>x.id===m.id);
  runStream({regenerate_id:m.id,model:pick},()=>{if(idx>=0)current.messages=current.messages.slice(0,idx);renderConvo({stick:true});});
}
function continueMsg(m){
  // resume generation from the end of this assistant message; the server appends to it in place
  const idx=current.messages.findIndex(x=>x.id===m.id);
  const prefix=m.content||"";
  runStream({continue_id:m.id,prefix},()=>{if(idx>=0)current.messages=current.messages.slice(0,idx);renderConvo({stick:true});});
}
function resendEdited(mid,newContent){
  const idx=current.messages.findIndex(x=>x.id===mid);
  const atts=(idx>=0?current.messages[idx].attachments:null)||null;   // the server carries these over; show them too
  runStream({edit_user_id:mid,content:newContent},()=>{if(idx>=0){current.messages=current.messages.slice(0,idx);current.messages.push({id:"tmp",role:"user",content:newContent,attachments:atts,ts:new Date().toISOString()});}renderConvo({stick:true});});
}

// per-conversation composer drafts (this browser only, capped at the 20 most recent)
let draftCid=null,_draftT=null;
function saveDraft(){
  if(!draftCid)return;
  try{const d=JSON.parse(localStorage.getItem("oracle_drafts")||"{}");const t=$("#input").value;
    if(t&&t.trim())d[draftCid]=t;else delete d[draftCid];
    const ks=Object.keys(d);while(ks.length>20)delete d[ks.shift()];
    localStorage.setItem("oracle_drafts",JSON.stringify(d));}catch(_){}
}
function loadDraft(cid){
  draftCid=cid;
  try{const d=JSON.parse(localStorage.getItem("oracle_drafts")||"{}");
    const inp=$("#input");if(inp.value!==(d[cid]||"")){inp.value=d[cid]||"";inp.style.height="auto";inp.dispatchEvent(new Event("input"));}}catch(_){}
}
async function openConvo(id){saveDraft();composeNew=false;current=await api("GET","/api/conversations/"+id);loadDraft(id);renderConvo({stick:true});renderTree();}
// The setup a new chat inherits from the one you are in: dialling in a model, character, prompt or
// sampler values and then hitting "new chat" used to throw all of it away and snap back to the site
// defaults. With no chat open (fresh load) there is nothing to carry, so the defaults still apply.
function carrySettings(){
  if(!current)return {system:CFG.default_system,model:CFG.default_model,character_id:null,params:{},tools:false,think:null};
  const c={system:current.system||"",model:current.model,character_id:current.character_id||null,
           params:current.params||{},tools:!!current.tools,
           think:current.think===undefined?null:current.think};
  if(isAdmin())c.endpoint_id=current.endpoint_id||null;
  return c;
}
async function newChat(){
  const carry=carrySettings();
  // Reuse an existing empty conversation instead of stacking blank rows in the sidebar, but only one
  // already on the character being carried — repurposing a blank chat that belongs to a different
  // character (or to none) would quietly rewrite it.
  const empty=convoCache.find(c=>!c.turns&&(c.character_id||null)===carry.character_id);
  if(empty){
    await openConvo(empty.id);
    // That shell kept whatever settings it was made with, so bring it in line with the carried setup.
    // Skipped when it already matches, so reusing a blank chat doesn't bump it up the sidebar.
    const stale=Object.keys(carry).some(k=>JSON.stringify(carry[k]??null)!==JSON.stringify(current[k]??null));
    if(stale){try{current=await api("POST","/api/conversations/"+current.id+"/settings",carry);renderConvo();refreshList();}catch(e){toast(e.message);}}
    $("#input").focus();closeSidebar();return;
  }
  current=await api("POST","/api/conversations",carry);
  renderConvo();refreshList();$("#input").focus();closeSidebar();
}

// ---------------- compose mode (a loom, not a chat)
// A composition is a conversation whose whole body is ONE assistant message. You paste or write
// text; the model resumes it from the exact character it ends on (the same prefill the "continue"
// button uses, which only works at all because the continuation hints ride in chat_template_kwargs
// — see _stream). "branch" writes the continuation to a *sibling* of that message instead of
// extending it, so one prefix carries several endings and the ‹ 1/2 › switcher walks them.
// The optional steering instruction is just the conversation's system prompt — no new field.
let composeNew=false;
// Set on the pointerdown that starts a "branch". Branching a loom means trimming the text back to
// the point you want to diverge at and forking from *there* — so the blur that click causes must
// not first write the trimmed text over the branch you are forking away from, which would destroy
// the very continuation you were keeping.
let composeForking=false;
function composeMsg(){return current?(current.messages||[]).filter(m=>m.role==="assistant").pop():null;}
function newComposition(){
  composeNew=true;composeGen++;current=null;draftCid=null;
  renderCompose();renderTree();closeSidebar();
  const t=$("#ctext");if(t)t.focus();
}
function composeCount(){
  const ta=$("#ctext"),n=$(".cbar .cn");if(!ta||!n)return;
  const v=ta.value;
  n.textContent=v?(fmtNum(v.length)+" chars  ·  ~"+fmtK(estTok(v))+" tok"):"";
}
function composeBusy(b){
  const go=document.querySelector('.crow [data-c="continue"]');if(!go)return;
  go.textContent=b?"stop":"continue";
  const br=document.querySelector('.crow [data-c="branch"]');if(br)br.disabled=b||!composeMsg();
}
// A composition is an ordinary conversation, so it reopens from the sidebar like a chat — but its
// row is only worth creating once there is text to put in it. Both the blur-save and a run come
// through here and share one in-flight create, so blurring the textarea and immediately hitting
// continue cannot turn one document into two conversations. composeGen guards the other direction:
// if the writer starts a *different* composition while a create is still in flight, the result is
// dropped rather than swapped in under them.
let composeCreating=null,composeGen=0;
function composeEnsure(text){
  if(current)return Promise.resolve(current);
  const gen=composeGen;
  if(!composeCreating)composeCreating=api("POST","/api/conversations",Object.assign(carrySettings(),{mode:"compose",seed:text}))
    .then(c=>{if(gen===composeGen){current=c;composeNew=false;draftCid=c.id;}refreshList();renderTree();return c;})
    .finally(()=>{composeCreating=null;});
  return composeCreating;
}
async function composeSave(){   // text left behind without ever running must still be there later
  const ta=$("#ctext");
  if(!ta||busy)return;
  if(composeForking){composeForking=false;return;}
  const m=composeMsg();
  if(!m){   // never run: without this, pasting and then clicking away would drop the whole document
    if(!ta.value.trim())return;
    try{await composeEnsure(ta.value);toast("composition saved — reopen it from the sidebar");}
    catch(e){toast(e.message,4000,'err');}
    return;
  }
  if(ta.value===(m.content||""))return;
  // blur fires on the mousedown that starts a run, i.e. *before* busy is set, so this patch can
  // still be in flight while the stream is going. It therefore updates the message in place and
  // never reassigns `current` — swapping in a pre-stream convo here would wipe the continuation.
  m.content=ta.value;
  try{await api("POST","/api/conversations/"+current.id+"/messages/"+m.id,{content:m.content});refreshList();}
  catch(e){toast(e.message,4000,'err');}
}
function renderCompose(){
  $("#composer").style.display="none";
  const m=composeMsg();
  if(current)syncBar();
  else{$("#title").textContent="new composition";$("#submeta").textContent="¶ composition";$("#ctxmeter").style.display="none";}
  const log=$("#log");log.innerHTML="";
  const el=document.createElement("div");el.className="compose";
  el.innerHTML='<div class="cbar">'+(m?sibNav(m):"")+'<span class="grow"></span><span class="cn"></span></div>'+
    '<textarea id="ctext" spellcheck="false" placeholder="paste or write the text to continue…"></textarea>'+
    '<div class="crow"><button class="pri" data-c="continue" title="keep writing from the end of this text">continue</button>'+
    '<button data-c="branch"'+(m?"":" disabled")+' title="write this continuation to a new branch, leaving the current one intact">branch</button>'+
    '<span class="chint">steer it with the system prompt in <b>tune</b></span></div>';
  log.appendChild(el);
  const ta=el.querySelector("#ctext");
  ta.value=m?(m.content||""):"";
  ta.scrollTop=ta.scrollHeight;
  el.querySelectorAll("[data-sib]").forEach(b=>b.onclick=()=>{if(b.disabled)return;switchSibling(m.siblings[m.sib_index+(b.dataset.sib==="next"?1:-1)]);});
  el.querySelector('[data-c="continue"]').onclick=()=>{if(busy)stopStream();else composeRun(false);};
  const br=el.querySelector('[data-c="branch"]');
  br.addEventListener("pointerdown",()=>{composeForking=true;});
  br.onclick=()=>{if(!busy)composeRun(true);};
  ta.addEventListener("input",composeCount);
  ta.addEventListener("blur",composeSave);
  composeCount();composeBusy(busy);
}
async function composeRun(branch){
  const ta=$("#ctext");if(busy||!ta)return;
  const text=ta.value;
  if(!text.trim()){toast("paste or write something to continue from");return;}
  setBusy(true);composeBusy(true);maybeAskNotif();
  // follow the tail only while the caret is already at the end, so streaming can't yank the view
  // away from a spot the writer scrolled back to
  const follow=ta.scrollTop+ta.clientHeight>=ta.scrollHeight-24;
  const c=new AbortController();activeController=c;
  try{
    await composeEnsure(text);
    const m=composeMsg();
    if(!m)throw new Error("nothing to continue");
    const res=await streamRequest("/api/conversations/"+current.id+"/stream",
      {continue_id:m.id,content:text,branch:!!branch},
      {onDelta:d=>{ta.value+=d;if(follow)ta.scrollTop=ta.scrollHeight;composeCount();},
       onReason:()=>{},                       // prefill resumes past the thinking block; nothing to show
       onNotice:msg=>toast(msg,6000)},c.signal);
    if(res&&res.convo)current=res.convo;else current=await api("GET","/api/conversations/"+current.id);
    notifyDone(true);
  }catch(e){
    // a stop (abort) still leaves the partial persisted server-side, so re-read either way
    if(e.name!=="AbortError"){
      // a prefill that ends on a finished sentence makes the model emit nothing at all; in a chat
      // that is a glitch, here it is just "this text is done" and the fix is to trim and retry
      const done=/empty response/.test(e.message||"");
      toast(done?"the model ended the text there — trim the last sentence and try again":(e.message||"stream failed"),5000,'err');
      notifyDone(false);
    }
    if(current&&current.id){try{current=await api("GET","/api/conversations/"+current.id);}catch(_){}}
  }
  finally{
    activeController=null;clearTimeout(_stopFallback);composeForking=false;setBusy(false);
    renderCompose();refreshList();
  }
}

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
// ---------------- tooltips (hover on desktop, tap on mobile)
let _tipEl=null;
function positionTip(box,el){
  box.classList.add("show");
  const r=el.getBoundingClientRect(),bw=box.offsetWidth,bh=box.offsetHeight,pad=8;
  let left=r.left+r.width/2-bw/2;left=Math.max(pad,Math.min(left,innerWidth-bw-pad));
  let top=r.top-bh-8;if(top<pad)top=r.bottom+8;
  box.style.left=left+"px";box.style.top=top+"px";
}
function showTip(el){const box=$("#tipbox"),t=el.dataset.tip||"";if(!t)return;box.textContent=t;_tipEl=el;el.classList.add("on");positionTip(box,el);}
function hideTip(){const box=$("#tipbox");box.classList.remove("show");if(_tipEl){_tipEl.classList.remove("on");_tipEl=null;}}
function bindTip(el,text){
  el.dataset.tip=text||"";
  el.addEventListener("pointerenter",e=>{if(e.pointerType==="mouse")showTip(el);});
  el.addEventListener("pointerleave",e=>{if(e.pointerType==="mouse")hideTip();});
  el.addEventListener("click",e=>{e.preventDefault();e.stopPropagation();if(_tipEl===el)hideTip();else{hideTip();showTip(el);}});
}
document.addEventListener("click",e=>{if(_tipEl&&e.target!==_tipEl)hideTip();});
document.addEventListener("scroll",()=>{if(_tipEl)hideTip();},true);
window.addEventListener("resize",()=>{if(_tipEl)hideTip();});

function buildParamsGrid(target,values){
  target.innerHTML="";
  CFG.param_specs.forEach(p=>{
    const v=(values&&values[p.key]!=null)?values[p.key]:"";
    const pg=document.createElement("div");pg.className="pg"+(v===""?" off":"");
    let html='<div class="pgname">'+esc(p.label)+(p.tip?'<span class="tip-ic" aria-label="'+esc(p.label)+' help">i</span>':'')+'</div>';
    html+='<div class="pgrow"><input type="number" data-k="'+p.key+'" step="'+(p.step||"any")+'"'+(p.min!=null?' min="'+p.min+'"':'')+(p.max!=null?' max="'+p.max+'"':'')+' placeholder="'+esc(p.ph||"")+'" value="'+(v===""?"":v)+'"><button class="reset" title="reset to server default">reset</button></div>';
    if(p.slider)html+='<input type="range" data-r="'+p.key+'" min="'+p.min+'" max="'+p.max+'" step="'+(p.step||"any")+'" value="'+(v===""?p.default:v)+'">';
    pg.innerHTML=html;
    const num=pg.querySelector("[data-k]"),rng=pg.querySelector("[data-r]"),rst=pg.querySelector(".reset"),tic=pg.querySelector(".tip-ic");
    if(tic&&p.tip)bindTip(tic,p.tip);
    const sync=on=>pg.classList.toggle("off",!on);
    num.addEventListener("input",()=>{if(rng&&num.value!=="")rng.value=num.value;sync(num.value!=="");});
    if(rng)rng.addEventListener("input",()=>{num.value=rng.value;sync(true);});
    rst.onclick=()=>{num.value="";if(rng)rng.value=p.default;sync(false);num.dispatchEvent(new Event("input",{bubbles:true}));};
    target.appendChild(pg);
  });
}
function readParamsGrid(target){const out={};target.querySelectorAll("input[data-k]").forEach(inp=>{const raw=inp.value.trim();if(raw==="")return;const sp=CFG.param_specs.find(p=>p.key===inp.dataset.k);out[inp.dataset.k]=sp&&sp.type==="int"?parseInt(raw,10):parseFloat(raw);});return out;}
function fillDefaults(target){const o={};CFG.param_specs.forEach(p=>{o[p.key]=p.default;});buildParamsGrid(target,o);}

// ---------------- per-model sampler presets
function presetsForModel(model){return (CFG.presets||[]).filter(p=>!p.models||!p.models.length||p.models.indexOf(model)>=0);}
function presetById(id){return (CFG.presets||[]).find(p=>p.id===id)||null;}
function paramsEqual(a,b){a=a||{};b=b||{};const ak=Object.keys(a),bk=Object.keys(b);if(ak.length!==bk.length)return false;return ak.every(k=>b[k]!=null&&String(a[k])===String(b[k]));}
function buildPresetSelect(selectId){
  const sel=$("#d_preset");if(!sel)return;
  const model=$("#d_model").value;
  sel.innerHTML="";
  const def=document.createElement("option");def.value="__default__";def.textContent="server default";sel.appendChild(def);
  const list=presetsForModel(model);
  const grp=(label,items)=>{if(!items.length)return;const og=document.createElement("optgroup");og.label=label;
    items.forEach(p=>{const o=document.createElement("option");o.value="p:"+p.id;o.textContent=p.name+((!p.models||!p.models.length)?" · all models":"");og.appendChild(o);});sel.appendChild(og);};
  grp("site presets",list.filter(p=>p.scope==="site"));
  grp("my presets",list.filter(p=>p.scope!=="site"));
  if(selectId){applyPresetById(selectId);sel.value="p:"+selectId;if(sel.value!=="p:"+selectId)syncPresetSelect();}
  else syncPresetSelect();
  updatePresetDel();
}
function syncPresetSelect(){
  const sel=$("#d_preset");if(!sel)return;
  const old=sel.querySelector('option[value="__custom__"]');if(old)old.remove();
  const cur=readParamsGrid($("#d_params"));
  if(Object.keys(cur).length===0){sel.value="__default__";updatePresetDel();return;}
  const match=presetsForModel($("#d_model").value).find(p=>paramsEqual(p.params,cur));
  if(match){sel.value="p:"+match.id;}
  else{const o=document.createElement("option");o.value="__custom__";o.textContent="(custom)";sel.insertBefore(o,sel.firstChild);sel.value="__custom__";}
  updatePresetDel();
}
function updatePresetDel(){
  const sel=$("#d_preset"),btn=$("#d_preset_del");if(!sel||!btn)return;
  const v=sel.value;let ok=false;
  if(v&&v.indexOf("p:")===0){const p=presetById(v.slice(2));ok=!!(p&&p.editable);}
  btn.style.display=ok?"":"none";
}
function applyPresetById(id){const p=presetById(id);if(p)buildParamsGrid($("#d_params"),p.params||{});}
function onPresetChange(){
  const sel=$("#d_preset"),v=sel.value;
  if(v==="__default__")buildParamsGrid($("#d_params"),{});
  else if(v.indexOf("p:")===0)applyPresetById(v.slice(2));
  if(v!=="__custom__"){const c=sel.querySelector('option[value="__custom__"]');if(c)c.remove();sel.value=v;}
  updatePresetDel();
  if(v!=="__custom__")applyParams();
}
async function savePresetDialog(){
  const model=$("#d_model").value;
  const params=readParamsGrid($("#d_params"));
  if(Object.keys(params).length===0){toast("set at least one parameter first");return;}
  const sel=$("#d_preset"),selected=sel.value.indexOf("p:")===0?presetById(sel.value.slice(2)):null;
  const editing=selected&&selected.editable?selected:null;
  const adminBox=isAdmin()?'<label class="dlg-chk"><input type="checkbox" id="ps_site"'+(editing&&editing.scope==="site"?" checked":"")+'> site-wide (shared with all users)</label>':'';
  const allModels=(CFG.all_models||CFG.models||[]).slice();
  if(model&&allModels.indexOf(model)<0)allModels.unshift(model);
  const pre=new Set(editing?(editing.models||[]):(model?[model]:[]));
  const pills=allModels.map(m=>'<label><input type="checkbox" value="'+esc(m)+'"'+(pre.has(m)?" checked":"")+'> '+esc(m)+'</label>').join("")||'<span class="dlg-hint">no models available</span>';
  const html='<h4>'+(editing?"update preset":"save preset")+'</h4>'
    +'<p>save the current sampler parameters as a reusable preset.</p>'
    +'<input id="ps_name" placeholder="preset name" value="'+esc(editing?editing.name:"")+'">'
    +'<div class="dlg-hint">applies to these models — leave all unchecked for every model</div>'
    +'<div class="dlg-models" id="ps_models">'+pills+'</div>'
    +adminBox
    +(editing?'<label class="dlg-chk"><input type="checkbox" id="ps_new"> save as a new preset instead of updating</label>':'')
    +'<div class="dlg-btns"><button class="btn-ghost" data-x>cancel</button><button class="btn-primary" data-ok>save</button></div>';
  const data=await dialog(html,(d,done)=>{
    const nm=d.querySelector("#ps_name");nm.focus();nm.select();
    const collect=()=>({name:nm.value.trim(),
      models:Array.from(d.querySelectorAll("#ps_models input:checked")).map(i=>i.value),
      site:!!(d.querySelector("#ps_site")&&d.querySelector("#ps_site").checked),
      asNew:!!(d.querySelector("#ps_new")&&d.querySelector("#ps_new").checked)});
    nm.onkeydown=e=>{if(e.key==="Enter")done(collect());};
    d.querySelector("[data-x]").onclick=()=>done(null);
    d.querySelector("[data-ok]").onclick=()=>done(collect());
  });
  if(!data)return;
  if(!data.name){toast("a preset name is required");return;}
  const preset={name:data.name,models:data.models,params,scope:data.site?"site":"private"};
  if(editing&&!data.asNew)preset.id=editing.id;
  try{const r=await api("POST","/api/presets",{preset});CFG.presets=r.presets;buildPresetSelect(r.id);toast("preset saved");}catch(e){toast(e.message);}
}
async function deletePreset(){
  const v=$("#d_preset").value;if(v.indexOf("p:")!==0)return;
  const p=presetById(v.slice(2));if(!p)return;
  if(!await uiConfirm("Delete preset '"+p.name+"'?",{danger:true,ok:"Delete"}))return;
  try{const r=await api("DELETE","/api/presets/"+p.id);CFG.presets=r.presets;buildPresetSelect();toast("deleted");}catch(e){toast(e.message);}
}

// ---------------- chat-settings drawer
async function openDrawer(){
  if(!current){toast("open or start a chat first");return;}
  const cs=$("#d_char");cs.innerHTML='<option value="">none</option>';
  CFG.characters.forEach(c=>{const o=document.createElement("option");o.value=c.id;o.textContent=(c.avatar?c.avatar+" ":"")+c.name+(c.scope==="site"?"  ·site":"");cs.appendChild(o);});
  cs.value=current.character_id||"";
  if(isAdmin()){const es=$("#d_endpoint");es.innerHTML='<option value="">default ('+esc((endpointName(CFG.settings.active_endpoint)||{}).name||"-")+')</option>';
    CFG.settings.endpoints.forEach(e=>{const o=document.createElement("option");o.value=e.id;o.textContent=e.name;es.appendChild(o);});es.value=current.endpoint_id||"";
    es.onchange=async()=>{await refreshDrawerModels(es.value,$("#d_model").value);buildPresetSelect();applyDrawer({endpoint_id:es.value||null,model:$("#d_model").value});};}
  await refreshDrawerModels(current.endpoint_id,current.model);
  cs.onchange=()=>{const c=charById(cs.value),patch={character_id:cs.value||null};
    // picking a character applies its system prompt (+ model); picking "none" just detaches it and
    // leaves the current prompt intact (no cancel button now, so don't wipe it out from under them).
    if(c){$("#d_system").value=c.system||"";patch.system=c.system||"";if(c.model){rebuildModelSelect($("#d_model"),Array.from($("#d_model").options).map(o=>o.value),c.model);patch.model=c.model;}}
    buildPresetSelect();applyDrawer(patch,{rerender:true,relist:true});};
  $("#d_model").onchange=()=>{buildPresetSelect();syncThinkRow();applyDrawer({model:$("#d_model").value},{relist:true});};
  $("#d_system").value=current.system||"";
  $("#d_system").oninput=applySystem;
  $("#d_tools").checked=!!current.tools;
  $("#d_tools").onchange=()=>applyDrawer({tools:$("#d_tools").checked});
  $("#d_think").onchange=()=>{const v=$("#d_think").value;applyDrawer({think:v===""?null:(v==="1"?1:0)});};
  syncThinkRow();
  // rolling context summary: only shown once compression has produced one for this chat
  const sw=$("#d_summary_wrap");
  if(current.ctx_summary){sw.style.display="block";$("#d_summary").value=current.ctx_summary;}
  else sw.style.display="none";
  $("#d_summary_save").onclick=async()=>{try{current=await api("POST","/api/conversations/"+current.id+"/settings",{ctx_summary:$("#d_summary").value});toast("summary saved");}catch(e){toast(e.message);}};
  $("#d_summary_clear").onclick=async()=>{if(!await uiConfirm("Clear the rolling summary? The model will re-summarize from scratch next time the window overflows.",{ok:"Clear"}))return;
    try{current=await api("POST","/api/conversations/"+current.id+"/settings",{ctx_summary:""});$("#d_summary").value="";sw.style.display="none";toast("summary cleared");}catch(e){toast(e.message);}};
  buildParamsGrid($("#d_params"),current.params||{});
  buildPresetSelect();
  showOverlay($("#drawer"));
}
function syncThinkRow(){
  // only show the thinking control for models an admin has marked thinking-capable site-wide
  const model=($("#d_model").value)||(current&&current.model)||"";
  const ok=(CFG.thinking_models||[]).includes(model);
  $("#d_think_wrap").style.display=ok?"":"none";
  if(ok)$("#d_think").value=(current&&current.think===1)?"1":(current&&current.think===0)?"0":"";   // tri-state
}
async function refreshDrawerModels(endpointId,value){
  let models=CFG.models;
  if(isAdmin()&&endpointId){try{const r=await api("GET","/api/models?endpoint="+encodeURIComponent(endpointId));models=r.models;}catch(_){}}
  rebuildModelSelect($("#d_model"),models,value||current.model);
}
function debounce(fn,ms){let t;return function(){clearTimeout(t);t=setTimeout(fn,ms);};}
// ---------------- search (title instantly client-side; message bodies via a debounced server query)
let searchContent=[],searchPending=false;
const _searchFetch=debounce(async()=>{
  const q=($("#searchbox").value||"").trim();
  if(q.length<2){searchContent=[];searchPending=false;renderTree();return;}
  try{const r=await api("GET","/api/search?q="+encodeURIComponent(q));searchContent=r.results||[];}
  catch(_){searchContent=[];}
  searchPending=false;renderTree();
},220);
function onSearchInput(){const v=$("#searchbox").value||"";$(".searchwrap").classList.toggle("has",v.length>0);searchPending=v.trim().length>=2;renderTree();_searchFetch();}
function clearSearch(){$("#searchbox").value="";onSearchInput();$("#searchbox").focus();}
function hlSnippet(text,q){
  const i=(text||"").toLowerCase().indexOf(q.toLowerCase());
  if(i<0)return esc(text||"");
  return esc(text.slice(0,i))+'<mark>'+esc(text.slice(i,i+q.length))+'</mark>'+esc(text.slice(i+q.length));
}
function searchRow(c,q){
  const d=convoRow(c);   // reuse the normal row (click to open, context menu, drag)
  if(c.snippet){const s=document.createElement("div");s.className="cs";s.innerHTML=hlSnippet(c.snippet,q);d.appendChild(s);}
  return d;
}
async function applyDrawer(patch,opts){
  if(!current)return;opts=opts||{};
  try{current=await api("POST","/api/conversations/"+current.id+"/settings",patch);
    syncBar();if(opts.rerender)renderConvo();if(opts.relist)refreshList();
  }catch(e){toast(e.message);}
}
const applyParams=debounce(()=>applyDrawer({params:readParamsGrid($("#d_params"))}),350);
const applySystem=debounce(()=>applyDrawer({system:$("#d_system").value}),450);

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
  if(isAdmin())tabs.push({id:"models",t:"user models"},{id:"endpoints",t:"endpoints"},{id:"defaults",t:"defaults"},{id:"users",t:"users"},{id:"stats",t:"stats"});
  $("#tabs").innerHTML=tabs.map((x,i)=>'<button data-tab="'+x.id+'"'+(i===0?' class="active"':'')+'>'+x.t+'</button>').join("");
  $$("#tabs button").forEach(b=>b.onclick=()=>switchTab(b.dataset.tab));
  $("#ch_sitewrap").style.display=isAdmin()?"flex":"none";
}
function switchTab(name){
  $$("#tabs button").forEach(b=>b.classList.toggle("active",b.dataset.tab===name));
  $$(".tab-pane").forEach(p=>p.classList.toggle("active",p.id==="tab-"+name));
  $("#settings-save").style.display=["models","endpoints","defaults"].includes(name)?"block":"none";
  $("#settings-note").textContent="";
  if(name==="users"){loadUsers();loadInvites();}
  if(name==="stats")loadStats();
}
async function loadStats(){
  const el=$("#stats-body");el.innerHTML='<div class="hintbox">loading…</div>';
  try{
    const s=await api("GET","/api/stats");
    if(!s.models.length){el.innerHTML='<div class="hintbox">no replies recorded yet</div>';return;}
    let h='<table class="stats-table"><tr><th>model</th><th>replies</th><th>tokens</th><th>30d replies</th><th>tok/s</th><th>ttft</th><th>cache</th></tr>';
    s.models.forEach(m=>{h+='<tr><td class="mono">'+esc(m.model)+'</td><td>'+fmtNum(m.replies)+'</td><td>'+fmtNum(m.tokens)+'</td><td>'+fmtNum(m.replies_30d)+'</td>'+
      '<td>'+(m.avg_tps!=null?m.avg_tps:"–")+'</td><td>'+(m.avg_ttft_ms!=null?(m.avg_ttft_ms/1000).toFixed(1)+"s":"–")+'</td>'+
      '<td>'+(m.avg_cache_pct!=null?m.avg_cache_pct+"%":"–")+'</td></tr>';});
    h+='</table><div class="hintbox" style="margin-top:10px;">'+fmtNum(s.totals.replies)+' replies · '+fmtNum(s.totals.tokens)+' completion tokens all-time. Averages cover replies that reported the metric.</div>';
    el.innerHTML=h;
  }catch(e){el.innerHTML='<div class="hintbox">'+esc(e.message)+'</div>';}
}
function openSettings(tab){
  renderCharacters();
  applyTheme(curTheme());applyPalette(curPalette());applyFont(curFont());syncTotpUI();syncPushUI();if($("#fs_range"))$("#fs_range").value=curFS();if($("#cw_range"))$("#cw_range").value=curCW();
  if(isAdmin()){renderEndpoints();$("#def_model").value=CFG.settings.default_model||"";$("#def_system").value=CFG.settings.default_system||"";$("#def_search").value=CFG.settings.search_url||"";$("#def_utility").value=CFG.settings.utility_model||"";$("#def_embed").value=CFG.settings.embed_model||"";buildParamsGrid($("#def_params"),CFG.settings.default_params||{});renderUserModels();renderThinkingModels();renderVisionModels();}
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
  renderKnowledge(c);
  $("#char-edit").style.display="block";$("#ch_name").focus();}
let editingCharId=null;
function renderKnowledge(c){
  editingCharId=(c&&c.id)||null;
  const wrap=$("#ch_klist");if(!wrap)return;wrap.innerHTML="";
  if(!editingCharId){wrap.innerHTML='<div class="hintbox">save the character first, then add documents</div>';$("#ch_kadd").disabled=true;return;}
  $("#ch_kadd").disabled=false;
  const items=c.knowledge||[];
  if(!items.length){wrap.innerHTML='<div class="hintbox">no documents yet — txt, md, pdf, csv…</div>';return;}
  items.forEach(k=>{
    const row=document.createElement("div");row.className="krow";
    row.innerHTML='<span class="kn">'+esc(k.name)+'</span><span class="kc">'+fmtNum(k.chars)+' chars</span><button class="mini danger" type="button">remove</button>';
    row.querySelector("button").onclick=async()=>{
      try{const r=await api("DELETE","/api/characters/"+c.id+"/knowledge/"+k.id);c.knowledge=r.knowledge;renderKnowledge(c);}catch(e){toast(e.message);}};
    wrap.appendChild(row);});
}
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
function renderThinkingModels(){const wrap=$("#tm_pick");if(!wrap)return;wrap.innerHTML="";const wl=new Set(CFG.settings.thinking_models||[]);
  (CFG.all_models||CFG.models||[]).forEach(m=>wrap.insertAdjacentHTML("beforeend",'<label><input type="checkbox" value="'+esc(m)+'" '+(wl.has(m)?"checked":"")+'>'+esc(m)+'</label>'));}
function renderVisionModels(){const wrap=$("#vm_pick");if(!wrap)return;wrap.innerHTML="";const wl=new Set(CFG.settings.vision_models||[]);
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
      // tests the endpoint as currently edited in the form — nothing is saved as a side effect
      try{const r=await api("POST","/api/settings/test",{endpoint:CFG.settings.endpoints[+card.dataset.idx]});
        if(r.ok){st.textContent="ok · "+r.models.length+" models";st.style.color="var(--ok)";}
        else{st.textContent="x "+(r.error||"failed");st.style.color="var(--danger)";}
      }catch(e){st.textContent="x "+e.message;st.style.color="var(--danger)";}};
    list.appendChild(card);
  });
}
function collectEndpoints(){$$("#ep-list .ep-card").forEach(card=>{const ep=CFG.settings.endpoints[+card.dataset.idx];if(!ep)return;card.querySelectorAll("[data-f]").forEach(i=>ep[i.dataset.f]=i.value.trim());});}
function addEndpoint(){CFG.settings.endpoints.push({id:"ep-"+Math.random().toString(36).slice(2,8),name:"New endpoint",url:"http://localhost:8000/v1/chat/completions",models_url:"",key:""});renderEndpoints();}
async function saveSettings(){
  collectEndpoints();
  const body={endpoints:CFG.settings.endpoints,active_endpoint:CFG.settings.active_endpoint,
    default_model:$("#def_model").value.trim(),default_system:$("#def_system").value,
    default_params:readParamsGrid($("#def_params")),user_models:readModelPick($("#um_pick")),
    thinking_models:readModelPick($("#tm_pick")),vision_models:readModelPick($("#vm_pick")),
    search_url:$("#def_search").value.trim(),utility_model:$("#def_utility").value.trim(),
    embed_model:$("#def_embed").value.trim()};
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
    row.innerHTML='<div class="cav">'+esc(u.username.slice(0,2).toUpperCase())+'</div><div class="cmain"><div class="cn">'+esc(u.username)+'<span class="badge">'+esc(u.role)+'</span>'+(u.disabled?'<span class="badge" style="color:var(--danger)">disabled</span>':'')+'</div><div class="cs">models: '+esc(am)+'</div></div><div class="cbtns"><button class="mini" data-peek title="read-only view of this user\'s chats and characters">view</button><button class="mini" data-edit>edit</button><button class="mini danger" data-del>del</button></div>';
    row.querySelector("[data-peek]").onclick=()=>openPeek(u.id,u.username);
    row.querySelector("[data-edit]").onclick=()=>editUser(u);
    row.querySelector("[data-del]").onclick=async()=>{if(!await uiConfirm("Delete user '"+u.username+"' and ALL their conversations?",{danger:true,ok:"Delete"}))return;try{const r=await api("DELETE","/api/users/"+u.id);userCache=r.users;renderUsers();toast("deleted");}catch(e){toast(e.message);}};
    list.appendChild(row);});
}
// ---------------- admin read-only peek at a user's chats + characters
let peekData=null;
function peekMsgEl(m,charName,userName){
  if(m.role==="tool")return toolResultEl(m);
  if(m.role==="assistant"&&m.tool&&m.tool.tool_calls&&!(m.content||"").trim()){const s=document.createElement("div");s.style.display="none";return s;}
  const d=document.createElement("div");d.className="msg "+m.role;
  const role=m.role==="user"?(userName||"user"):(charName||"assistant");
  const reason=m.reasoning?'<details class="reason"><summary>reasoning</summary><div class="rbody">'+esc(m.reasoning)+'</div></details>':"";
  const edited=m.edited?' · edited':'';
  const mark=m.rating?'<span class="ratemark '+(m.rating>0?'up':'down')+'">'+(m.rating>0?'&#9650;':'&#9660;')+'</span>':'';
  let disp=m.content||"";
  if(m.role==="assistant"&&disp.indexOf("<tool_call>")>=0)disp=stripTC(disp)||"";
  const body=m.role==="assistant"?'<div class="bubble">'+md(disp)+'</div>':'<div class="bubble raw">'+esc(m.content)+'</div>';
  d.innerHTML='<div class="head"><span class="role">'+esc(role)+'</span><span class="tm">'+esc(fmtTime(m.ts)+edited)+'</span>'+mark+'</div>'+reason+attsHtml(m)+body+metaLine(m);
  const mk=d.querySelector(".meta .pill.k"),mtip=paramsTipText(m);
  if(mk&&mtip){mk.classList.add("has-tip");bindTip(mk,mtip);}
  return d;
}
function peekCharName(cid){const c=(peekData&&peekData.characters||[]).find(x=>x.id===cid);return c?c.name:"assistant";}
async function openPeek(uid,username){
  let d;try{d=await api("GET","/api/admin/users/"+uid);}catch(e){toast(e.message);return;}
  peekData=d;
  $("#peek-title").textContent="\u{1F441} "+username+" · read-only";
  $$("#peek-tabs button").forEach(b=>b.classList.toggle("on",b.dataset.pt==="chats"));
  $("#peek-chats").style.display="flex";$("#peek-chars").style.display="none";
  renderPeekConvos();renderPeekChars();
  $("#peek-view").innerHTML='<div class="peek-hint">select a conversation to read</div>';
  closeModal();
  $("#backdrop").classList.add("show");$("#peekmodal").classList.add("show");
}
function renderPeekConvos(){
  const L=$("#peek-convos");L.innerHTML="";
  if(!peekData.conversations.length){L.innerHTML='<div class="peek-hint">no conversations</div>';return;}
  peekData.conversations.forEach(c=>{const it=document.createElement("div");it.className="peek-item";
    it.innerHTML='<div class="pt">'+esc(c.title||"untitled")+'</div><div class="ps">'+esc((c.model||"?")+" · "+relTime(c.updated)+" · "+(c.turns||0)+" msgs")+'</div>';
    it.onclick=()=>{$$("#peek-convos .peek-item").forEach(x=>x.classList.remove("on"));it.classList.add("on");openPeekConvo(c.id);};
    L.appendChild(it);});
}
async function openPeekConvo(cid){
  const V=$("#peek-view");V.innerHTML='<div class="peek-hint">loading…</div>';
  let convo;try{convo=await api("GET","/api/admin/conversations/"+cid);}catch(e){V.innerHTML='<div class="peek-hint">'+esc(e.message)+'</div>';return;}
  const charName=peekCharName(convo.character_id),userName=peekData.user.username;
  V.innerHTML="";const wrap=document.createElement("div");wrap.className="wrap";
  (convo.messages||[]).forEach(m=>{if(m.role==="user"||m.role==="assistant"||m.role==="tool")wrap.appendChild(peekMsgEl(m,charName,userName));});
  if(!wrap.children.length)wrap.innerHTML='<div class="peek-hint">empty conversation</div>';
  addCodeCopy(wrap);
  V.appendChild(wrap);V.scrollTop=0;
}
function renderPeekChars(){
  const L=$("#peek-chars");L.innerHTML="";
  if(!peekData.characters.length){L.innerHTML='<div class="peek-hint">no characters</div>';return;}
  peekData.characters.forEach(c=>{const card=document.createElement("div");card.className="ep-card peek-char";
    card.innerHTML='<div class="cn">'+esc((c.avatar?c.avatar+" ":"")+c.name)+(c.scope==="site"?' <span class="badge">site</span>':'')+(c.model?' <span class="badge">'+esc(c.model)+'</span>':'')+'</div><div class="peek-sys">'+esc(c.system||"(no system prompt)")+'</div>';
    L.appendChild(card);});
}
function closePeek(){$("#peekmodal").classList.remove("show");if(!$("#modal").classList.contains("show")&&!$("#drawer").classList.contains("show"))$("#backdrop").classList.remove("show");peekData=null;}
function userModelPick(selected){const wrap=$("#us_models");wrap.innerHTML="";const sel=new Set(selected||[]);
  (CFG.all_models||CFG.models||[]).forEach(m=>wrap.insertAdjacentHTML("beforeend",'<label><input type="checkbox" value="'+esc(m)+'" '+(sel.has(m)?"checked":"")+'>'+esc(m)+'</label>'));}
function editUser(u){$("#user-edit").style.display="block";
  if(u){$("#us_id").value=u.id;$("#us_newonly").style.display="none";$("#us_reset_wrap").style.display="block";$("#us_role").value=u.role;userModelPick(u.allowed_models);$("#us_reset").value="";}
  else{$("#us_id").value="";$("#us_newonly").style.display="block";$("#us_reset_wrap").style.display="none";$("#us_name").value="";$("#us_pw").value="";$("#us_role").value="user";userModelPick([]);}}
async function saveUser(){const id=$("#us_id").value;
  try{if(id){const body={role:$("#us_role").value,allowed_models:readModelPick($("#us_models"))};if($("#us_reset").value)body.password=$("#us_reset").value;const r=await api("POST","/api/users/"+id,body);userCache=r.users;}
    else{const r=await api("POST","/api/users",{username:$("#us_name").value.trim(),password:$("#us_pw").value,role:$("#us_role").value,allowed_models:readModelPick($("#us_models"))});userCache=r.users;}
    $("#user-edit").style.display="none";renderUsers();toast("saved");}catch(e){toast(e.message);}}
// invite links
let inviteCache=[];
async function loadInvites(){try{const j=await api("GET","/api/invites");inviteCache=j.invites||[];renderInvites();}catch(e){}}
function inviteUrl(tok){return location.origin+"/invite/"+tok;}
function inviteExpiryText(iv){
  if(iv.expires==null)return "never expires";
  const ms=iv.expires*1000-Date.now();
  if(ms<=0)return "expired";
  const days=Math.floor(ms/86400000);
  if(days>=1)return "expires in "+days+" day"+(days>1?"s":"");
  const hrs=Math.max(1,Math.floor(ms/3600000));
  return "expires in "+hrs+" hr"+(hrs>1?"s":"");
}
function copyInvite(url){copyText(url,"link copied");}
function renderInvites(){
  const list=$("#invite-list");list.innerHTML="";
  if(!inviteCache.length){list.innerHTML='<div style="color:var(--faint);font-family:var(--mono);font-size:11px;margin-bottom:10px;">no invite links yet.</div>';return;}
  inviteCache.forEach(iv=>{
    const url=inviteUrl(iv.token),dead=iv.status!=="active";
    const uses=iv.uses+"/"+(iv.max_uses==null?"∞":iv.max_uses)+" used";
    const row=document.createElement("div");row.className="row-card";row.style.alignItems="flex-start";
    row.innerHTML='<div class="cmain"><div class="cn">'+esc(iv.role)+'<span class="badge"'+(dead?' style="color:var(--danger);border-color:var(--danger-weak)"':'')+'>'+esc(iv.status)+'</span></div>'
      +'<div class="cs">'+esc(uses)+' · '+esc(inviteExpiryText(iv))+(iv.note?' · '+esc(iv.note):'')+'</div>'
      +'<input class="invite-url" readonly value="'+esc(url)+'"></div>'
      +'<div class="cbtns"><button class="mini" data-copy>copy</button><button class="mini danger" data-revoke>revoke</button></div>';
    const inp=row.querySelector(".invite-url");
    inp.onclick=()=>inp.select();
    row.querySelector("[data-copy]").onclick=()=>copyInvite(url);
    row.querySelector("[data-revoke]").onclick=async()=>{if(!await uiConfirm("Revoke this invite link? Anyone holding it can no longer register.",{danger:true,ok:"Revoke"}))return;try{const r=await api("DELETE","/api/invites/"+iv.token);inviteCache=r.invites;renderInvites();toast("revoked");}catch(e){toast(e.message);}};
    list.appendChild(row);
  });
}
function newInvite(){$("#invite-edit").style.display="block";$("#iv_role").value="user";$("#iv_days").value="";$("#iv_uses").value="";$("#iv_note").value="";
  const wrap=$("#iv_models");wrap.innerHTML="";(CFG.all_models||CFG.models||[]).forEach(m=>wrap.insertAdjacentHTML("beforeend",'<label><input type="checkbox" value="'+esc(m)+'">'+esc(m)+'</label>'));}
async function createInvite(){
  const body={role:$("#iv_role").value,allowed_models:readModelPick($("#iv_models")),
    days:$("#iv_days").value.trim()?parseFloat($("#iv_days").value):null,
    max_uses:$("#iv_uses").value.trim()?parseInt($("#iv_uses").value,10):null,
    note:$("#iv_note").value.trim()};
  try{const r=await api("POST","/api/invites",body);inviteCache=r.invites;$("#invite-edit").style.display="none";renderInvites();
    copyText(inviteUrl(r.token),"invite link created · copied");}catch(e){toast(e.message);}
}
// account
async function changePassword(){const o=$("#ac_old").value,n=$("#ac_new").value,n2=$("#ac_new2").value;
  if(n!==n2){toast("passwords do not match");return;}
  try{await api("POST","/api/account/password",{old:o,new:n});$("#ac_old").value=$("#ac_new").value=$("#ac_new2").value="";toast("password changed");}catch(e){toast(e.message);}}
async function savePersona(){try{const r=await api("POST","/api/account",{persona:$("#ac_persona").value.trim()});CFG.me=r.me;toast("name saved");}catch(e){toast(e.message);}}
async function deleteAccount(){
  if(!await uiConfirm("Delete your account and EVERYTHING you own (chats, folders, private characters)? This cannot be undone.",{danger:true,ok:"Delete account"}))return;
  const c=await uiPrompt("Type your username to confirm:","",{title:"Confirm deletion",ok:"Delete forever"});
  if(c===null)return;if(c!==CFG.me.username){toast("username did not match");return;}
  try{await api("DELETE","/api/account");location.href="/login";}catch(e){toast(e.message);}}

// ---------------- overlays (focus moves in on open and back to the opener on close)
function _focusInto(p){p._prevFocus=document.activeElement;const f=p.querySelector("button,input,select,textarea");if(f)f.focus();}
function _focusBack(p){if(p._prevFocus&&p._prevFocus.focus){try{p._prevFocus.focus();}catch(_){}}p._prevFocus=null;}
function showOverlay(p){$("#backdrop").classList.add("show");p.classList.add("show");_focusInto(p);}
function closeOverlay(p){p.classList.remove("show");if(!$("#modal").classList.contains("show"))$("#backdrop").classList.remove("show");_focusBack(p);}
function showModal(){const m=$("#modal");$("#backdrop").classList.add("show");m.classList.add("show");_focusInto(m);}
function closeModal(){const m=$("#modal");m.classList.remove("show");if(!$("#drawer").classList.contains("show"))$("#backdrop").classList.remove("show");_focusBack(m);}
function openSidebar(){$("#sidebar").classList.add("show");$("#backdrop").classList.add("show");}
function closeSidebar(){$("#sidebar").classList.remove("show");if(!$("#modal").classList.contains("show")&&!$("#drawer").classList.contains("show"))$("#backdrop").classList.remove("show");}
function closeAll(){$("#drawer").classList.remove("show");closeModal();$("#peekmodal").classList.remove("show");peekData=null;$("#sidebar").classList.remove("show");$("#backdrop").classList.remove("show");hideMenu();}

// ---------------- wire up
$("#newbtn").onclick=newChat;
$("#composebtn").onclick=newComposition;
$("#menubtn").onclick=openSidebar;
$("#revealbtn").onclick=()=>toggleCollapse(false);
$("#collapsebtn").onclick=()=>toggleCollapse(true);
$("#searchbox").addEventListener("input",onSearchInput);
$("#searchbox").addEventListener("keydown",e=>{if(e.key==="Escape"&&($("#searchbox").value||"")){e.stopPropagation();clearSearch();}});
$("#searchclear").onclick=clearSearch;
$("#foldernew").onclick=async()=>{const n=await uiPrompt("Name your new folder:","",{title:"New folder",ok:"Create"});if(n!=null){await api("POST","/api/folders",{name:(n.trim()||"New folder")});refreshFolders();toast("folder created");}};
$("#selbtn").onclick=()=>setSelMode(!selMode);
$("#seldone").onclick=()=>setSelMode(false);
$("#selmove").onclick=bulkMove;
$("#seldel").onclick=bulkDelete;
$("#title").onclick=editTitle;
$("#charchip").onclick=()=>openSettings("characters");
$("#tunebtn").onclick=openDrawer;
$("#exportbtn").onclick=e=>{if(!current){toast("open a chat first");return;}showMenu(e,'<button data-f="md">export markdown</button><button data-f="json">export json</button>',m=>{m.querySelectorAll("[data-f]").forEach(b=>b.onclick=()=>{hideMenu();exportChat(current.id,b.dataset.f);});});};
$("#sharebtn").onclick=()=>{if(!current){toast("open a chat first");return;}shareDialog(current);};
$("#settingsbtn").onclick=()=>openSettings(isAdmin()?"endpoints":"account");
$("#whobtn").onclick=()=>openSettings("account");
$("#themebtn").onclick=()=>applyTheme(curTheme()==="dark"?"light":"dark");
$("#logoutbtn").onclick=async()=>{try{await api("POST","/api/logout");}catch(_){}; location.href="/login";};
$("#modelsel").onchange=async()=>{if(current){try{current=await api("POST","/api/conversations/"+current.id+"/settings",{model:$("#modelsel").value});syncBar();toast("model: "+$("#modelsel").value);}catch(e){toast(e.message);syncBar();}}};
$("#send").onclick=()=>busy?stopStream():send();
$("#d_defaults").onclick=()=>{fillDefaults($("#d_params"));syncPresetSelect();applyParams();};
$("#d_clear").onclick=()=>{buildParamsGrid($("#d_params"),{});syncPresetSelect();applyParams();};
$("#d_preset").onchange=onPresetChange;
$("#d_preset_save").onclick=savePresetDialog;
$("#d_preset_del").onclick=deletePreset;
$("#d_params").addEventListener("input",()=>{syncPresetSelect();applyParams();});
$$("[data-close-drawer]").forEach(b=>b.onclick=()=>closeOverlay($("#drawer")));
$$("[data-close-modal]").forEach(b=>b.onclick=closeModal);
$$("[data-close-peek]").forEach(b=>b.onclick=closePeek);
$$("#peek-tabs button").forEach(b=>b.onclick=()=>{$$("#peek-tabs button").forEach(x=>x.classList.toggle("on",x===b));const t=b.dataset.pt;$("#peek-chats").style.display=t==="chats"?"flex":"none";$("#peek-chars").style.display=t==="chars"?"flex":"none";});
$("#backdrop").onclick=closeAll;
$("#ep-add").onclick=addEndpoint;
$("#settings-save").onclick=saveSettings;
$("#char-add").onclick=()=>editCharacter(null);
$("#ch_save").onclick=saveCharacter;
$("#ch_cancel").onclick=()=>$("#char-edit").style.display="none";
// ---------------- TOTP 2FA
function syncTotpUI(){
  const on=!!(CFG.me&&CFG.me.totp);
  $("#totp_off").style.display=on?"none":"block";
  $("#totp_on").style.display=on?"block":"none";
  $("#totp_setup").style.display="none";
}
$("#totp_enable").onclick=async()=>{
  try{const r=await api("POST","/api/account/totp/setup",{});
    $("#totp_secret").textContent=r.secret;$("#totp_setup").dataset.secret=r.secret;
    $("#totp_off").style.display="none";$("#totp_setup").style.display="block";$("#totp_code").focus();
  }catch(e){toast(e.message);}};
$("#totp_cancel").onclick=()=>syncTotpUI();
$("#totp_confirm").onclick=async()=>{
  try{const r=await api("POST","/api/account/totp/confirm",{secret:$("#totp_setup").dataset.secret,code:$("#totp_code").value});
    CFG.me=r.me;$("#totp_code").value="";syncTotpUI();toast("two-factor enabled");
  }catch(e){toast(e.message,5000,'err');}};
$("#totp_disable").onclick=async()=>{
  try{const r=await api("POST","/api/account/totp/disable",{password:$("#totp_pw").value});
    CFG.me=r.me;$("#totp_pw").value="";syncTotpUI();toast("two-factor disabled");
  }catch(e){toast(e.message,5000,'err');}};
// ---------------- web push
function b64uToU8(s){s=s.replace(/-/g,"+").replace(/_/g,"/");s+="=".repeat((4-s.length%4)%4);
  const b=atob(s),a=new Uint8Array(b.length);for(let i=0;i<b.length;i++)a[i]=b.charCodeAt(i);return a;}
async function syncPushUI(){
  const wrap=$("#push_wrap"),t=$("#push_toggle");
  if(!("serviceWorker" in navigator)||!("PushManager" in window)){wrap.style.display="none";return;}
  try{
    const k=await api("GET","/api/push/key");
    if(!k.key){wrap.style.display="none";return;}
    wrap.dataset.key=k.key;
    const reg=await navigator.serviceWorker.ready;
    t.checked=!!(await reg.pushManager.getSubscription());
  }catch(_){wrap.style.display="none";}
}
$("#push_toggle").onchange=async()=>{
  const t=$("#push_toggle");
  try{
    const reg=await navigator.serviceWorker.ready;
    if(t.checked){
      if(Notification.permission==="denied")throw new Error("notifications are blocked for this site in the browser");
      const sub=await reg.pushManager.subscribe({userVisibleOnly:true,applicationServerKey:b64uToU8($("#push_wrap").dataset.key)});
      await api("POST","/api/push/subscribe",{subscription:sub.toJSON()});toast("push enabled on this device");
    }else{
      const sub=await reg.pushManager.getSubscription();
      if(sub){await api("POST","/api/push/unsubscribe",{endpoint:sub.endpoint});await sub.unsubscribe();}
      toast("push disabled");
    }
  }catch(e){t.checked=!t.checked;toast(e.message,5000,'err');}
};
$("#ch_kadd").onclick=()=>$("#ch_kfile").click();
$("#ch_kfile").onchange=async e=>{
  const c=(CFG.characters||[]).find(x=>x.id===editingCharId);
  if(!c){e.target.value="";return;}
  for(const f of Array.from(e.target.files)){
    if(f.size>12*1024*1024){toast(f.name+": over the 12 MB limit",4000,'err');continue;}
    const data=await new Promise(res=>{const r=new FileReader();r.onload=()=>res(String(r.result).split(",")[1]);r.readAsDataURL(f);});
    try{const r=await api("POST","/api/characters/"+c.id+"/knowledge",{name:f.name,data});c.knowledge=r.knowledge;renderKnowledge(c);toast("added "+f.name);}
    catch(err){toast(f.name+": "+err.message,5000,'err');}
  }
  e.target.value="";
};
// ---------------- chat import (ChatGPT export / SillyTavern jsonl / generic {messages})
function parseImportFile(name,txt){
  const convs=[];
  if(name.endsWith(".jsonl")){          // SillyTavern chat log
    const msgs=[];let title=null;
    for(const line of txt.split("\n")){
      if(!line.trim())continue;
      let o;try{o=JSON.parse(line);}catch(_){continue;}
      if(o.user_name!==undefined&&o.character_name!==undefined){title=o.character_name;continue;}
      if(o.mes!==undefined)msgs.push({role:o.is_user?"user":"assistant",content:String(o.mes)});
    }
    if(msgs.length)convs.push({title:title||name.replace(/\.jsonl$/,""),messages:msgs});
    return convs;
  }
  const data=JSON.parse(txt);
  const fromMapping=c=>{                // ChatGPT export: walk current_node -> parent chain
    const msgs=[];let node=c.current_node;
    while(node&&c.mapping&&c.mapping[node]){
      const n=c.mapping[node],m=n.message;
      if(m&&m.author&&["user","assistant","system"].includes(m.author.role)&&m.content&&Array.isArray(m.content.parts)){
        const t=m.content.parts.filter(p=>typeof p==="string").join("\n").trim();
        if(t&&!(m.metadata&&m.metadata.is_visually_hidden_from_conversation))msgs.push({role:m.author.role,content:t});
      }
      node=n.parent;
    }
    msgs.reverse();
    return msgs.length?{title:c.title||"imported",messages:msgs}:null;
  };
  if(Array.isArray(data)){
    for(const c of data){
      if(c&&c.mapping){const v=fromMapping(c);if(v)convs.push(v);}
      else if(c&&Array.isArray(c.messages))convs.push({title:c.title,messages:c.messages.map(m=>({role:m.role,content:String(m.content||"")}))});
    }
  }else if(data&&data.mapping){const v=fromMapping(data);if(v)convs.push(v);}
  else if(data&&Array.isArray(data.conversations)){
    for(const c of data.conversations)if(c&&Array.isArray(c.messages))convs.push({title:c.title,messages:c.messages.map(m=>({role:m.role,content:String(m.content||"")}))});
  }else if(data&&Array.isArray(data.messages)){
    convs.push({title:data.title,messages:data.messages.map(m=>({role:m.role,content:String(m.content||m.mes||"")}))});
  }
  return convs;
}
$("#imp_btn").onclick=()=>$("#imp_file").click();
$("#imp_file").onchange=async e=>{
  let convs=[];
  for(const f of Array.from(e.target.files)){
    try{convs=convs.concat(parseImportFile(f.name,await f.text()));}
    catch(err){toast(f.name+": "+(err.message||"could not parse"),5000,'err');}
  }
  e.target.value="";
  convs=convs.filter(c=>c&&c.messages&&c.messages.some(m=>["user","assistant"].includes(m.role)&&(m.content||"").trim()));
  if(!convs.length){toast("no conversations found in the file",4000,'err');return;}
  if(!await uiConfirm("Import "+convs.length+" conversation"+(convs.length>1?"s":"")+"?",{ok:"Import"}))return;
  let done=0;
  try{
    for(let i=0;i<convs.length;i+=20){   // batches keep each request under the body-size cap
      const r=await api("POST","/api/import",{conversations:convs.slice(i,i+20)});
      done+=r.imported||0;
    }
  }catch(err){toast(err.message,5000,'err');}
  toast("imported "+done+" conversation"+(done===1?"":"s"));
  refreshList();
};
$("#def_defaults").onclick=()=>fillDefaults($("#def_params"));
$("#def_clear").onclick=()=>buildParamsGrid($("#def_params"),{});
$("#user-add").onclick=()=>editUser(null);
$("#us_save").onclick=saveUser;
$("#us_cancel").onclick=()=>$("#user-edit").style.display="none";
$("#invite-add").onclick=newInvite;
$("#iv_save").onclick=createInvite;
$("#iv_cancel").onclick=()=>$("#invite-edit").style.display="none";
$("#ac_save").onclick=changePassword;
$("#ac_persona_save").onclick=savePersona;
$("#acctdel").onclick=deleteAccount;
$("#exportall").onclick=()=>{const a=document.createElement("a");a.href="/api/export";a.click();toast("exporting…");};
$$("#themeseg button").forEach(b=>b.onclick=()=>applyTheme(b.dataset.th));
$$("#palseg button").forEach(b=>b.onclick=()=>applyPalette(b.dataset.pal));
["cp_bg","cp_text","cp_accent"].forEach(id=>{const el=$("#"+id);if(el)el.oninput=()=>{
  const c=getCustom();c[id.slice(3)]=el.value;
  try{localStorage.setItem("oracle_custom",JSON.stringify(c));}catch(_){}
  applyCustom();};});
if($("#cp_reset"))$("#cp_reset").onclick=()=>{localStorage.removeItem("oracle_custom");localStorage.removeItem("oracle_custom_vars");applyPalette("custom");};
$$("#fontseg button").forEach(b=>b.onclick=()=>applyFont(b.dataset.f));
$("#fs_range").addEventListener("input",e=>applyFS(parseFloat(e.target.value)));
$("#fs_minus").onclick=()=>applyFS(curFS()-0.05);
$("#fs_plus").onclick=()=>applyFS(curFS()+0.05);
$("#fs_reset").onclick=()=>applyFS(1);
$("#cw_range").addEventListener("input",e=>applyCW(parseInt(e.target.value)));
$("#cw_reset").onclick=()=>applyCW(840);
const inp=$("#input");
inp.addEventListener("keydown",e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();if(!busy)send();}});
inp.addEventListener("input",()=>{inp.style.height="auto";inp.style.height=Math.min(inp.scrollHeight,230)+"px";updateCtx();});
inp.addEventListener("paste",e=>{   // a big paste (essay, transcript, …) becomes an attachment instead of flooding the box
  const cd=e.clipboardData||window.clipboardData;if(!cd)return;
  const text=cd.getData("text");if(!text)return;
  if(text.length>=1600||(text.match(/\n/g)||[]).length>=18){
    e.preventDefault();
    const n=pendingAtt.filter(a=>a.pasted).length+1;
    pendingAtt.push({name:"pasted text"+(n>1?" "+n:""),text:text,tokens_est:estTok(text),pasted:true});
    renderPending();
    toast("Pasted text added as an attachment (~"+fmtK(estTok(text))+" tok)");
  }
});
$("#attachbtn").onclick=()=>$("#fileinput").click();
$("#fileinput").onchange=e=>{const fs=[...e.target.files];e.target.value="";if(fs.length)uploadFiles(fs);};
(function(){const comp=$("#composer");if(!comp)return;
  let dc=0;
  comp.addEventListener("dragenter",e=>{e.preventDefault();dc++;comp.classList.add("dragover");});
  comp.addEventListener("dragover",e=>{e.preventDefault();});
  comp.addEventListener("dragleave",e=>{e.preventDefault();if(--dc<=0){dc=0;comp.classList.remove("dragover");}});
  comp.addEventListener("drop",e=>{e.preventDefault();dc=0;comp.classList.remove("dragover");const fs=[...((e.dataTransfer&&e.dataTransfer.files)||[])];if(fs.length)uploadFiles(fs);});
})();
(function(){const log=$("#log"),btn=$("#scrolltop"),down=$("#scrollbottom");if(!log||!btn)return;
  let last=0;
  log.addEventListener("scroll",()=>{
    const st=log.scrollTop;
    if(st<last-2)autoScroll=false;       // user scrolled up -> stop following
    if(nearBottom())autoScroll=true;      // back at the bottom -> resume following
    last=st;
    btn.classList.toggle("show",st>500);
    if(down)down.classList.toggle("show",!nearBottom()&&log.scrollHeight-log.clientHeight>200);
  });
  btn.onclick=()=>{autoScroll=false;log.scrollTo({top:0,behavior:"smooth"});};
  if(down)down.onclick=()=>{autoScroll=true;scrollDown();down.classList.remove("show");};
})();
document.addEventListener("keydown",e=>{
  const mod=e.ctrlKey||e.metaKey;
  if(mod&&!e.shiftKey&&e.key.toLowerCase()==="k"){e.preventDefault();if(innerWidth<=860)openSidebar();$("#searchbox").focus();$("#searchbox").select();return;}
  if(mod&&e.shiftKey&&e.key.toLowerCase()==="o"){e.preventDefault();if(!busy)newChat();return;}
  if(e.key==="Escape"){if($("#dlgwrap").classList.contains("show"))return;closeAll();}
});

(async function init(){
  applyTheme(localStorage.getItem("oracle_theme")||"dark");
  applyPalette(curPalette());
  applyFont(curFont());
  if("serviceWorker" in navigator){try{navigator.serviceWorker.register("/sw.js");}catch(_){}}
  $("#input").addEventListener("input",()=>{clearTimeout(_draftT);_draftT=setTimeout(saveDraft,400);});
  applyFS(parseFloat(localStorage.getItem("oracle_fs"))||1);
  applyCW(parseInt(localStorage.getItem("oracle_cw"))||840);
  applySidebar();initResize();
  try{await loadConfig();}catch(e){
    $("#log").innerHTML='<div class="wrap"><div class="empty"><div class="glyph">!</div><h2>could not load</h2>'+
      '<p>'+esc(e.message||"network error")+'</p>'+
      '<p style="margin-top:14px;"><button class="btn-ghost" id="retrybtn">retry</button></p></div></div>';
    const rb=$("#retrybtn");if(rb)rb.onclick=()=>location.reload();
    return;
  }
  if(!CFG.models.length)toast("model endpoint unreachable — check Settings → Endpoints",6000,'err');
  await refreshList();renderEmpty();
  if(convoCache.length){try{await openConvo(convoCache[0].id);}catch(_){}}
})();
</script></body></html>"""

PAGE = PAGE_HEAD + PAGE_BODY + PAGE_JS1 + PAGE_JS2 + PAGE_JS3

# Precompressed once at startup: the ~140 KB page gzips ~4x, and the ETag turns repeat loads into 304s.
PAGE_BYTES = PAGE.encode("utf-8")
PAGE_GZ = gzip.compress(PAGE_BYTES, 9)
PAGE_ETAG = '"' + hashlib.sha1(PAGE_BYTES).hexdigest() + '"'
PAGE_CSP = csp_for(PAGE)   # hash-based script-src, computed once (page is static)


# ---------------------------------------------------------------- public share viewer
# A standalone, dependency-free reader for a frozen conversation snapshot. Deliberately stripped
# down: no app chrome, no auth, no private data — just the prose, set for gorgeous readability.
SHARE_PAGE_TMPL = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__ · ORACLE</title>
<script>try{var _t=localStorage.getItem('oracle_theme');if(_t==='light'||_t==='dark')document.documentElement.setAttribute('data-theme',_t);}catch(e){}</script>
<meta name="robots" content="noindex, nofollow">
<meta name="theme-color" content="#100c08">
<meta name="color-scheme" content="dark light">
<meta property="og:type" content="article">
<meta property="og:site_name" content="ORACLE">
<meta property="og:title" content="__TITLE__">
<meta property="og:description" content="__OGDESC__">
<meta property="og:image" content="__HOME__/og-image.png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="__TITLE__">
<meta name="twitter:description" content="__OGDESC__">
<meta name="twitter:image" content="__HOME__/og-image.png">
<link rel="icon" type="image/svg+xml" href="__HOME__/favicon.svg">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400;1,6..72,500&display=swap" rel="stylesheet">
<style>
  :root{
    --bg:#100c08; --surface:#1b150d; --line:rgba(150,124,84,.13);
    --text:#e8ddc5; --muted:#9a8c72; --faint:#6a6049;
    --accent:#cf8a3c; --accent2:#cda261; --user:#b49f78; --code-bg:#0c0905;
    --mono:'IBM Plex Mono',ui-monospace,SFMono-Regular,Consolas,monospace;
    --serif:'Newsreader','Iowan Old Style',Georgia,'Times New Roman',serif;
    color-scheme:dark;
  }
  /* light palette: follow the OS by default, but let an explicit data-theme override it */
  @media (prefers-color-scheme: light){:root:not([data-theme]){
    --bg:#f4ecda; --surface:#e6dcc6; --line:rgba(90,68,34,.16);
    --text:#2c2316; --muted:#6a5c43; --faint:#9a8b6e;
    --accent:#b26a1d; --accent2:#7d5320; --user:#6d5a32; --code-bg:#e9e0cc;
    color-scheme:light;
  }}
  :root[data-theme="light"]{
    --bg:#f4ecda; --surface:#e6dcc6; --line:rgba(90,68,34,.16);
    --text:#2c2316; --muted:#6a5c43; --faint:#9a8b6e;
    --accent:#b26a1d; --accent2:#7d5320; --user:#6d5a32; --code-bg:#e9e0cc;
    color-scheme:light;
  }
  *{box-sizing:border-box;}
  html{-webkit-text-size-adjust:100%;}
  body{margin:0;background:var(--bg);color:var(--text);font-family:var(--serif);
       font-size:19px;line-height:1.75;-webkit-font-smoothing:antialiased;}
  ::selection{background:var(--accent);color:var(--bg);}
  a{color:var(--accent);text-underline-offset:2px;}
  .page{max-width:46rem;margin:0 auto;padding:clamp(28px,6vw,80px) clamp(20px,5vw,40px) 40px;}
  header.doc{border-bottom:1px solid var(--line);padding-bottom:26px;margin-bottom:14px;}
  .brandrow{display:flex;align-items:center;justify-content:space-between;gap:12px;}
  .brand{display:inline-flex;align-items:center;gap:9px;font-family:var(--mono);font-size:11px;
         letter-spacing:.34em;text-transform:uppercase;color:var(--accent);text-decoration:none;font-weight:500;}
  .brand img{width:20px;height:20px;display:block;opacity:.92;}
  .theme-toggle{background:none;border:none;padding:5px;margin:-5px;cursor:pointer;color:var(--faint);
                border-radius:8px;display:inline-flex;line-height:0;transition:color .15s;}
  .theme-toggle:hover{color:var(--accent);}
  .theme-toggle svg{width:17px;height:17px;stroke:currentColor;stroke-width:1.7;fill:none;
                    stroke-linecap:round;stroke-linejoin:round;display:block;}
  h1{font-family:var(--serif);font-weight:600;font-size:clamp(28px,5vw,40px);line-height:1.18;
     margin:20px 0 10px;letter-spacing:-.01em;}
  .sub{font-family:var(--mono);font-size:11px;letter-spacing:.06em;color:var(--faint);text-transform:uppercase;}
  .turn{padding:22px 0;border-bottom:1px solid var(--line);}
  .turn:last-of-type{border-bottom:none;}
  .who{font-family:var(--mono);font-size:10.5px;letter-spacing:.2em;text-transform:uppercase;
       color:var(--faint);margin-bottom:10px;}
  .turn.assistant .who{color:var(--accent2);}
  .turn.user{}
  .turn.user .who{color:var(--user);}
  .user-body{font-style:italic;color:var(--muted);font-size:.96em;white-space:pre-wrap;overflow-wrap:anywhere;
             border-left:2px solid var(--line);padding-left:18px;}
  .body{overflow-wrap:anywhere;word-break:break-word;}
  .body p{margin:.75em 0;} .body p:first-child{margin-top:0;} .body p:last-child{margin-bottom:0;}
  .body h1,.body h2,.body h3,.body h4{font-weight:600;line-height:1.25;margin:1.1em 0 .45em;letter-spacing:-.005em;}
  .body h1{font-size:1.5em;} .body h2{font-size:1.32em;} .body h3{font-size:1.15em;} .body h4{font-size:1.03em;}
  .body ul,.body ol{margin:.6em 0;padding-left:1.45em;} .body li{margin:.3em 0;}
  .body code{font-family:var(--mono);font-size:.8em;background:var(--code-bg);border-radius:5px;padding:.08em .36em;}
  .body pre{position:relative;background:var(--code-bg);border:1px solid var(--line);border-radius:11px;padding:15px 17px;overflow-x:auto;margin:.85em 0;}
  .copy-code{position:absolute;top:8px;right:8px;background:var(--surface);color:var(--faint);border:1px solid var(--line);
             border-radius:6px;padding:3px 8px;font-family:var(--mono);font-size:10px;letter-spacing:.06em;text-transform:uppercase;
             cursor:pointer;opacity:0;transition:opacity .14s,color .14s;}
  .body pre:hover .copy-code,.copy-code:focus{opacity:1;} .copy-code:hover{color:var(--accent);}
  #totop{position:fixed;right:clamp(14px,4vw,32px);bottom:clamp(14px,4vw,32px);width:40px;height:40px;border-radius:50%;
         background:var(--surface);color:var(--accent);border:1px solid var(--line);cursor:pointer;font-size:18px;line-height:1;
         display:flex;align-items:center;justify-content:center;opacity:0;pointer-events:none;transition:opacity .2s,transform .2s;
         transform:translateY(8px);box-shadow:0 6px 20px rgba(0,0,0,.25);}
  #totop.show{opacity:1;pointer-events:auto;transform:translateY(0);} #totop:hover{color:var(--bg);background:var(--accent);}
  .body pre code{background:none;padding:0;font-size:.82em;line-height:1.6;}
  .body blockquote{margin:.8em 0;padding:.15em 0 .15em 1.15em;border-left:2px solid var(--accent);
                   color:var(--muted);font-style:italic;}
  .body hr{border:none;border-top:1px solid var(--line);margin:1.3em 0;}
  .body strong{font-weight:600;} .body a{color:var(--accent);}
  .files{margin-top:12px;display:flex;flex-wrap:wrap;gap:7px;}
  .files .f{font-family:var(--mono);font-size:11px;color:var(--faint);background:var(--surface);
            border:1px solid var(--line);border-radius:7px;padding:3px 9px;}
  footer.doc{margin-top:40px;padding-top:22px;border-top:1px solid var(--line);
             font-family:var(--mono);font-size:11px;letter-spacing:.05em;color:var(--faint);
             display:flex;justify-content:space-between;align-items:center;gap:14px;flex-wrap:wrap;}
  footer.doc a{color:var(--muted);text-decoration:none;}
  footer.doc a:hover{color:var(--accent);}
</style>
</head><body>
<div class="page">
  <header class="doc">
    <div class="brandrow">
      <a class="brand" href="__HOME__/"><img src="__HOME__/favicon.svg" alt=""> Oracle</a>
      <button class="theme-toggle" id="themebtn" type="button" aria-label="Toggle light or dark theme" title="Toggle theme"></button>
    </div>
    <h1 id="title"></h1>
    <div class="sub" id="sub"></div>
  </header>
  <main id="log"></main>
  <footer class="doc">
    <span>A shared conversation · read-only</span>
    <a href="__HOME__/">__HOMELABEL__ &rsaquo;</a>
  </footer>
</div>
<button id="totop" type="button" aria-label="Scroll to top" title="Back to top">&uarr;</button>
<script id="d" type="application/json">__DATA__</script>
<script>""" + MD_ESC_JS + r"""
(function(){
  let D;try{D=JSON.parse(document.getElementById("d").textContent);}catch(e){D={title:"Conversation",messages:[]};}
  document.title=(D.title||"Conversation")+" · ORACLE";
  document.getElementById("title").textContent=D.title||"Conversation";
  const log=document.getElementById("log");
  (D.messages||[]).forEach(m=>{
    const sec=document.createElement("section");sec.className="turn "+(m.role==="user"?"user":"assistant");
    const who=document.createElement("div");who.className="who";who.textContent=m.role==="user"?"Prompt":"Oracle";
    sec.appendChild(who);
    const body=document.createElement("div");
    if(m.role==="assistant"){body.className="body";body.innerHTML=md(m.content||"");}
    else{body.className="body user-body";body.textContent=m.content||"";}
    sec.appendChild(body);
    if(m.files&&m.files.length){const fb=document.createElement("div");fb.className="files";
      m.files.forEach(n=>{const c=document.createElement("span");c.className="f";c.textContent="⧉ "+n;fb.appendChild(c);});
      sec.appendChild(fb);}
    log.appendChild(sec);
  });
  const sub=document.getElementById("sub");
  const n=(D.messages||[]).length;
  const words=(D.messages||[]).reduce((a,m)=>a+(m.content||"").split(/\s+/).filter(Boolean).length,0);
  const mins=Math.max(1,Math.round(words/220));
  sub.textContent=n+(n===1?" message":" messages")+" · ~"+mins+" min read";
  // copy button on each code block
  log.querySelectorAll(".body pre").forEach(pre=>{
    const b=document.createElement("button");b.className="copy-code";b.type="button";b.textContent="copy";
    b.onclick=()=>{const code=pre.querySelector("code"),txt=code?code.textContent:pre.textContent;
      navigator.clipboard.writeText(txt).then(()=>{b.textContent="copied";setTimeout(()=>b.textContent="copy",1300);},()=>{});};
    pre.appendChild(b);
  });
})();
// scroll-to-top
(function(){
  const t=document.getElementById("totop");if(!t)return;
  const onscroll=()=>t.classList.toggle("show",(window.scrollY||document.documentElement.scrollTop)>600);
  window.addEventListener("scroll",onscroll,{passive:true});onscroll();
  t.onclick=()=>window.scrollTo({top:0,behavior:"smooth"});
})();
// ---- light/dark toggle (defaults to the OS preference; an explicit choice is remembered)
(function(){
  const MOON='<svg viewBox="0 0 24 24"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>';
  const SUN='<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="4.2"/><path d="M12 2v2.2M12 19.8V22M4.2 4.2l1.6 1.6M18.2 18.2l1.6 1.6M2 12h2.2M19.8 12H22M4.2 19.8l1.6-1.6M18.2 5.8l1.6-1.6"/></svg>';
  const btn=document.getElementById("themebtn");if(!btn)return;
  const cur=()=>document.documentElement.getAttribute("data-theme")||(matchMedia("(prefers-color-scheme: light)").matches?"light":"dark");
  const paint=()=>{btn.innerHTML=cur()==="dark"?MOON:SUN;};
  btn.onclick=()=>{const t=cur()==="dark"?"light":"dark";document.documentElement.setAttribute("data-theme",t);
    try{localStorage.setItem("oracle_theme",t);}catch(e){}
    const mc=document.querySelector('meta[name="theme-color"]');if(mc)mc.setAttribute("content",t==="dark"?"#100c08":"#f4ecda");
    paint();};
  paint();
})();
</script>
</body></html>"""

SHARE_404 = (r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>not found · ORACLE</title>
<meta name="robots" content="noindex, nofollow">
<style>html,body{height:100%;margin:0;background:#100c08;color:#9a8c72;
font-family:'IBM Plex Mono',ui-monospace,monospace;display:flex;align-items:center;justify-content:center;}
.b{text-align:center;padding:30px;} .b b{display:block;color:#cf8a3c;letter-spacing:.34em;text-transform:uppercase;
font-size:12px;margin-bottom:14px;} .b p{font-size:14px;line-height:1.7;} .b a{color:#cf8a3c;}
@media (prefers-color-scheme:light){html,body{background:#f4ecda;color:#6a5c43;}}
</style></head><body><div class="b"><b>Oracle</b><p>This shared link has expired or was never here.<br>
<a href="/">return to the oracle &rsaquo;</a></p></div></body></html>""")

# Computed from the static template (whose executable scripts are placeholder-free), so injected
# scripts in share data are never hashed into the policy. The per-share JSON data island is
# non-executable and excluded from hashing, keeping this constant valid for every rendered share.
SHARE_CSP = csp_for(SHARE_PAGE_TMPL)


def render_share_page(sh):
    try:
        data = json.loads(sh["data"])
    except Exception:
        data = {"title": sh["title"] or "Conversation", "messages": []}
    title = (data.get("title") or "Conversation").strip() or "Conversation"
    # OG description: first prompt, condensed
    desc = ""
    for m in data.get("messages", []):
        if m.get("role") == "user" and (m.get("content") or "").strip():
            desc = " ".join((m["content"]).split())[:180]
            break
    if not desc:
        desc = "A shared conversation from ORACLE."
    # Embed the snapshot as JSON, neutralizing any HTML-significant characters so model output
    # cannot break out of the <script> element (defense in depth atop the client-side escaping).
    blob = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    home_label = PUBLIC_URL.split("://", 1)[-1].split("/", 1)[0]
    out = SHARE_PAGE_TMPL
    out = out.replace("__TITLE__", _html_mod.escape(title, quote=True))
    out = out.replace("__OGDESC__", _html_mod.escape(desc, quote=True))
    out = out.replace("__HOMELABEL__", _html_mod.escape(home_label, quote=True))
    out = out.replace("__HOME__", _html_mod.escape(PUBLIC_URL, quote=True))
    out = out.replace("__DATA__", blob)   # user data last: can't collide with other placeholders
    return out


if __name__ == "__main__":
    init_db()
    secret_bytes()   # resolve (and, with env set, scrub the legacy DB copy of) the session secret now
    bootstrap_admin()
    start_backups()
    # the OG/touch icons render in a slow pure-Python pixel loop — warm them off the request path
    threading.Thread(target=lambda: (og_image_png(), apple_icon_png()), daemon=True).start()
    server = ThreadedHTTPServer(("0.0.0.0", PORT), Handler)
    log.info("ORACLE chat on http://localhost:%d", PORT)
    log.info("database: %s", DB_PATH)
    if user_count() == 0:
        log.info("no users yet -> open the page to create the first admin (or set KENOSIS_ADMIN_USER/PASS)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass






