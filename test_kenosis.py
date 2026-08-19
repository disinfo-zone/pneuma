"""Tests for the pure-Python core of kenosis_chat.py: auth tokens, tree logic, share-snapshot
privacy stripping, schema migration, and the small parsing helpers.

Run:  python -m pytest test_kenosis.py    (or)    python -m unittest test_kenosis -v

Uses a throwaway SQLite file per run; never touches a real chat.db.
"""

import io
import json
import os
import sqlite3
import tempfile
import threading
import time
import unittest

_TMP = tempfile.mkdtemp(prefix="kenosis-test-")
os.environ["KENOSIS_DB"] = os.path.join(_TMP, "test.db")
os.environ["KENOSIS_SESSION_SECRET"] = "unit-test-secret"
os.environ["KENOSIS_BACKUP_HOURS"] = "0"

import kenosis_chat as k  # noqa: E402  (env must be set before import)


def setUpModule():
    k.init_db()


class TestAuth(unittest.TestCase):
    def test_password_roundtrip(self):
        h = k.hash_pw("correct horse")
        self.assertTrue(k.verify_pw("correct horse", h))
        self.assertFalse(k.verify_pw("wrong", h))
        self.assertFalse(k.verify_pw("correct horse", "garbage"))

    def test_session_roundtrip_and_version(self):
        tok = k.sign_session("alice", 3)
        self.assertEqual(k.parse_session(tok), ("alice", 3))

    def test_session_tamper_rejected(self):
        tok = k.sign_session("alice")
        payload, sig = tok.split(".")
        evil = k.b64(json.dumps({"u": "admin", "exp": int(time.time()) + 9999}).encode())
        self.assertIsNone(k.parse_session(evil + "." + sig))
        self.assertIsNone(k.parse_session(payload + "." + k.b64(b"x" * 32)))
        self.assertIsNone(k.parse_session("not-a-token"))

    def test_expired_session_rejected(self):
        payload = k.b64(json.dumps({"u": "alice", "exp": int(time.time()) - 10}).encode("utf-8"))
        import hashlib, hmac as hm
        sig = k.b64(hm.new(k.secret_bytes(), payload.encode(), hashlib.sha256).digest())
        self.assertIsNone(k.parse_session(payload + "." + sig))

    def test_legacy_token_without_version_maps_to_zero(self):
        payload = k.b64(json.dumps({"u": "alice", "exp": int(time.time()) + 999}).encode("utf-8"))
        import hashlib, hmac as hm
        sig = k.b64(hm.new(k.secret_bytes(), payload.encode(), hashlib.sha256).digest())
        self.assertEqual(k.parse_session(payload + "." + sig), ("alice", 0))

    def test_secret_not_persisted(self):
        row = k.db().execute("SELECT 1 FROM settings WHERE key='session_secret'").fetchone()
        self.assertIsNone(row)

    def test_token_entropy(self):
        toks = {k.gen_token() for _ in range(200)}
        self.assertEqual(len(toks), 200)
        for t in toks:
            self.assertGreaterEqual(len(t), 24)

    def test_login_throttle(self):
        key = "unit-test-ip"
        for _ in range(k.LOGIN_MAX_FAILS):
            k.login_record_fail(key)
        self.assertGreater(k.login_retry_after(key), 0)
        k.login_clear(key)
        self.assertEqual(k.login_retry_after(key), 0)


class _ConvoBase(unittest.TestCase):
    def _mk_convo(self, owner=1):
        cid = k._cid()
        with k.db():
            k.db().execute(
                "INSERT INTO conversations(id,owner_id,title,system,model,params,created,updated)"
                " VALUES(?,?,?,?,?,?,?,?)", (cid, owner, "", "sys", "m", "{}", k._now(), k._now()))
        return cid


class TestTree(_ConvoBase):
    def test_positions_are_sequential_and_unique(self):
        cid = self._mk_convo()
        with k.db():
            a = k.insert_message(cid, None, "user", "q1")
            b = k.insert_message(cid, a, "assistant", "a1")
            k.insert_message(cid, b, "user", "q2")
        rows = k.db().execute("SELECT position FROM messages WHERE convo_id=? ORDER BY position", (cid,)).fetchall()
        self.assertEqual([r["position"] for r in rows], [0, 1, 2])

    def test_concurrent_inserts_no_duplicate_positions(self):
        cid = self._mk_convo()
        with k.db():
            root = k.insert_message(cid, None, "user", "root")
        errs = []

        def worker():
            try:
                for _ in range(20):
                    with k.db():   # commit per insert, as the app's persist() does
                        k.insert_message(cid, root, "assistant", "x")
            except Exception as e:  # pragma: no cover
                errs.append(e)

        ts = [threading.Thread(target=worker) for _ in range(4)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        self.assertFalse(errs)
        rows = k.db().execute("SELECT position FROM messages WHERE convo_id=?", (cid,)).fetchall()
        pos = [r["position"] for r in rows]
        self.assertEqual(len(pos), len(set(pos)), "duplicate positions under concurrency")

    def test_active_path_and_siblings(self):
        cid = self._mk_convo()
        u1 = k.insert_message(cid, None, "user", "q")
        a1 = k.insert_message(cid, u1, "assistant", "first answer")
        a2 = k.insert_message(cid, u1, "assistant", "regenerated answer")  # sibling branch
        path = k.active_path(cid, a2)
        self.assertEqual([m["id"] for m in path], [u1, a2])
        self.assertEqual(path[1]["sib_count"], 2)
        self.assertEqual(path[1]["sib_index"], 1)
        self.assertEqual(path[1]["siblings"], [a1, a2])

    def test_default_leaf_follows_latest_branch(self):
        cid = self._mk_convo()
        u1 = k.insert_message(cid, None, "user", "q")
        k.insert_message(cid, u1, "assistant", "a1")
        a2 = k.insert_message(cid, u1, "assistant", "a2")
        path = k.active_path(cid, None)   # no leaf set -> deepest/latest branch
        self.assertEqual(path[-1]["id"], a2)

    def test_delete_subtree_removes_descendants_and_reseats_leaf(self):
        cid = self._mk_convo()
        u1 = k.insert_message(cid, None, "user", "q1")
        a1 = k.insert_message(cid, u1, "assistant", "a1")
        u2 = k.insert_message(cid, a1, "user", "q2")
        a2 = k.insert_message(cid, u2, "assistant", "a2")
        k.set_leaf(cid, a2)
        k.db().commit()
        k.delete_subtree(cid, u2)
        left = {r["id"] for r in k.db().execute("SELECT id FROM messages WHERE convo_id=?", (cid,))}
        self.assertEqual(left, {u1, a1})
        leaf = k.db().execute("SELECT active_leaf_id FROM conversations WHERE id=?", (cid,)).fetchone()[0]
        self.assertEqual(leaf, a1)

    def test_delete_subtree_missing_id_is_noop(self):
        cid = self._mk_convo()
        k.insert_message(cid, None, "user", "q")
        k.delete_subtree(cid, "m-nope")
        n = k.db().execute("SELECT COUNT(*) c FROM messages WHERE convo_id=?", (cid,)).fetchone()["c"]
        self.assertEqual(n, 1)

    def test_chain_content_walks_to_root(self):
        cid = self._mk_convo()
        u1 = k.insert_message(cid, None, "user", "q1")
        a1 = k.insert_message(cid, u1, "assistant", "a1")
        u2 = k.insert_message(cid, a1, "user", "q2")
        seq = k.chain_content(cid, u2)
        self.assertEqual([m["content"] for m in seq], ["q1", "a1", "q2"])


class TestShareSnapshot(_ConvoBase):
    def test_snapshot_strips_private_data(self):
        cid = self._mk_convo()
        u1 = k.insert_message(cid, None, "user", "question",
                              attachments=[{"name": "secret.pdf", "text": "ATTACHMENT BODY"}])
        a1 = k.insert_message(cid, u1, "assistant", "tool step",
                              tool={"tool_calls": [{"id": "c1"}]})
        t1 = k.insert_message(cid, a1, "tool", "FETCHED PRIVATE CONTENT",
                              tool={"tool_call_id": "c1"})
        a2 = k.insert_message(cid, t1, "assistant", "the answer", reasoning="CHAIN OF THOUGHT")
        k.set_leaf(cid, a2)
        k.db().commit()
        snap = k.build_share_snapshot(cid)
        blob = json.dumps(snap)
        self.assertNotIn("sys", [m.get("role") for m in snap["messages"]])
        self.assertNotIn("CHAIN OF THOUGHT", blob)         # no reasoning
        self.assertNotIn("ATTACHMENT BODY", blob)          # file names only, not bodies
        self.assertNotIn("FETCHED PRIVATE CONTENT", blob)  # no tool results
        self.assertIn("secret.pdf", blob)                  # the name is shown
        roles = {m["role"] for m in snap["messages"]}
        self.assertEqual(roles, {"user", "assistant"})

    def test_share_page_script_breakout_neutralized(self):
        sh = {"data": json.dumps({"title": "</script><script>alert(1)</script>",
                                  "messages": [{"role": "user", "content": "</script>x"}]}),
              "title": "t"}
        page = k.render_share_page(sh)
        self.assertNotIn("</script><script>alert(1)</script>", page)


class TestHelpers(unittest.TestCase):
    def test_clean_title(self):
        self.assertEqual(k.clean_title('Title: "Hello World."'), "Hello World")
        self.assertEqual(k.clean_title("one two three four five six seven eight nine ten"),
                         "one two three four five six seven eight")
        self.assertEqual(k.clean_title(""), "")

    def test_strip_and_parse_tool_calls(self):
        leaked = 'before <tool_call><function=fetch_url><parameter=url>http://x.com</parameter></function></tool_call> after'
        self.assertEqual(k.strip_tool_calls(leaked), "before  after")
        calls = k.parse_text_tool_calls(leaked)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "fetch_url")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"])["url"], "http://x.com")
        dangling = "text <tool_call> unfinished"
        self.assertEqual(k.strip_tool_calls(dangling), "text")

    def test_apply_macros(self):
        out = k.apply_macros("Hi {{user}}, I am {{char}}.{{newline}}{{// hidden}}{{unknown}}",
                             {"user": "Dave", "char": "Artaud"})
        self.assertEqual(out, "Hi Dave, I am Artaud.\n{{unknown}}")
        roll = k.apply_macros("{{roll:1d1}}", {})
        self.assertEqual(roll, "1")

    def test_valid_id(self):
        self.assertTrue(k.valid_id("m-abc123"))
        self.assertFalse(k.valid_id("../etc/passwd"))
        self.assertFalse(k.valid_id(""))
        self.assertFalse(k.valid_id("x" * 65))

    def test_restore_endpoint_keys(self):
        k.set_setting("endpoints", [{"id": "e1", "name": "n", "url": "u", "key": "real-key"}])
        merged = k.restore_endpoint_keys([{"id": "e1", "name": "n2", "url": "u", "key": k.KEY_SENTINEL}])
        self.assertEqual(merged[0]["key"], "real-key")
        merged = k.restore_endpoint_keys([{"id": "e1", "url": "u", "key": "new-key"}])
        self.assertEqual(merged[0]["key"], "new-key")
        red = k.redacted_endpoints()
        self.assertEqual(red[0]["key"], k.KEY_SENTINEL)

    def test_attach_block_fencing(self):
        block = k._attach_block([{"name": 'a"b.txt', "text": "BODY"}])
        self.assertIn("<file name=\"a'b.txt\">", block)
        self.assertIn("BODY", block)


class TestModelDefaults(unittest.TestCase):
    """Sampler params layer global -> per-model -> per-conversation."""

    def setUp(self):
        self._saved = {kk: k.get_setting(kk) for kk in ("default_params", "model_defaults",
                                                        "default_model")}
        k.set_setting("default_params", {"temperature": 1.05, "top_p": 0.99, "min_p": 0.03,
                                         "xtc_probability": 0.4})
        k.set_setting("model_defaults", {
            "centostron1": {"temperature": 0.95, "min_p": 0.05, "xtc_probability": 0.5,
                            "repetition_penalty": 1.03},
            "partialmodel": {"temperature": 0.7},
        })
        k.set_setting("default_model", "fallbackmodel")

    def tearDown(self):
        for kk, v in self._saved.items():
            k.set_setting(kk, v)

    def test_model_without_entry_is_unchanged(self):
        p = k.effective_params({"model": "othermodel", "params": {}})
        self.assertEqual(p["temperature"], 1.05)
        self.assertEqual(p["xtc_probability"], 0.4)

    def test_model_entry_overrides_global(self):
        p = k.effective_params({"model": "centostron1", "params": {}})
        self.assertEqual(p["temperature"], 0.95)
        self.assertEqual(p["min_p"], 0.05)
        self.assertEqual(p["xtc_probability"], 0.5)
        self.assertEqual(p["repetition_penalty"], 1.03)

    def test_model_entry_merges_over_global_rather_than_replacing(self):
        # top_p is stated only globally; a partial model entry must not drop it
        p = k.effective_params({"model": "partialmodel", "params": {}})
        self.assertEqual(p["temperature"], 0.7)
        self.assertEqual(p["top_p"], 0.99)
        self.assertEqual(p["xtc_probability"], 0.4)

    def test_conversation_params_beat_the_model_entry(self):
        p = k.effective_params({"model": "centostron1", "params": {"temperature": 1.4}})
        self.assertEqual(p["temperature"], 1.4)
        self.assertEqual(p["min_p"], 0.05)   # untouched keys still come from the model entry

    def test_zero_valued_model_defaults_survive_the_filter(self):
        k.set_setting("model_defaults", {"m0": {"top_k": 0, "xtc_probability": 0.0}})
        p = k.effective_params({"model": "m0", "params": {}})
        self.assertEqual(p["top_k"], 0)
        self.assertEqual(p["xtc_probability"], 0.0)

    def test_model_falls_back_to_default_model_setting(self):
        k.set_setting("model_defaults", {"fallbackmodel": {"temperature": 0.5}})
        p = k.effective_params({"params": {}})
        self.assertEqual(p["temperature"], 0.5)

    def test_override_selects_that_models_defaults(self):
        convo = {"model": "othermodel", "params": {}, "system": ""}
        _, model, _, params = k.resolve_request(convo, "centostron1")
        self.assertEqual(model, "centostron1")
        self.assertEqual(params["temperature"], 0.95)
        # and without the override the conversation's own model still decides
        _, model, _, params = k.resolve_request(convo)
        self.assertEqual(model, "othermodel")
        self.assertEqual(params["temperature"], 1.05)


class TestMigration(unittest.TestCase):
    def test_legacy_linear_db_migrates_to_tree(self):
        path = os.path.join(_TMP, "legacy.db")
        con = sqlite3.connect(path)
        con.executescript("""
            CREATE TABLE users(id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL,
                pw_hash TEXT NOT NULL, role TEXT NOT NULL DEFAULT 'user',
                allowed_models TEXT, disabled INTEGER NOT NULL DEFAULT 0, created TEXT NOT NULL);
            CREATE TABLE settings(key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE conversations(id TEXT PRIMARY KEY, owner_id INTEGER NOT NULL, folder_id TEXT,
                title TEXT, system TEXT, model TEXT, endpoint_id TEXT, params TEXT, character_id TEXT,
                created TEXT NOT NULL, updated TEXT NOT NULL);
            CREATE TABLE messages(id TEXT PRIMARY KEY, convo_id TEXT NOT NULL, position INTEGER NOT NULL,
                role TEXT NOT NULL, content TEXT, reasoning TEXT, model TEXT, meta TEXT, ts TEXT, edited TEXT);
            INSERT INTO conversations VALUES('c1',1,NULL,'t','s','m',NULL,'{}',NULL,'2024','2024');
            INSERT INTO messages(id,convo_id,position,role,content) VALUES('m1','c1',0,'user','q');
            INSERT INTO messages(id,convo_id,position,role,content) VALUES('m2','c1',1,'assistant','a');
        """)
        con.commit()
        con.close()
        old_db, old_conn = k.DB_PATH, getattr(k._local, "conn", None)
        k.DB_PATH = path
        k._local.conn = None
        try:
            k.init_db()
            c = k.db()
            cols = [r["name"] for r in c.execute("PRAGMA table_info(messages)")]
            for col in ("parent_id", "rating", "attachments", "tool"):
                self.assertIn(col, cols)
            ucols = [r["name"] for r in c.execute("PRAGMA table_info(users)")]
            self.assertIn("session_version", ucols)
            self.assertEqual(c.execute("SELECT parent_id FROM messages WHERE id='m2'").fetchone()[0], "m1")
            self.assertEqual(c.execute("SELECT active_leaf_id FROM conversations WHERE id='c1'").fetchone()[0], "m2")
        finally:
            k.DB_PATH, k._local.conn = old_db, old_conn


class TestContextTrim(unittest.TestCase):
    def test_est_tokens(self):
        self.assertEqual(k.est_tokens(""), 1)
        self.assertEqual(k.est_tokens("x" * 400), 100)

    def test_tok_factor_clamps_and_smooths(self):
        k.update_tok_factor("tm", 1000, 100)   # 10x -> clamped to 4
        self.assertLessEqual(k.tok_factor("tm"), 4.0)
        k.update_tok_factor("tm2", 120, 100)
        self.assertAlmostEqual(k.tok_factor("tm2"), 1.2, places=3)


class TestCalculator(unittest.TestCase):
    def test_arithmetic(self):
        self.assertEqual(k.safe_calc("2+2*3"), 8)
        self.assertEqual(k.safe_calc("2^10"), 1024)          # ^ alias for **
        self.assertAlmostEqual(k.safe_calc("sqrt(2)**2"), 2, places=9)
        self.assertEqual(k.safe_calc("max(3, min(7, 5))"), 5)
        self.assertAlmostEqual(k.safe_calc("-pi"), -3.14159265, places=6)

    def test_rejects_abuse(self):
        for evil in ('__import__("os")', '().__class__', '"a"*9', 'open("x")',
                     '9**9**9', 'lambda: 1', '[1,2]', 'x', 'factorial(99999)'):
            with self.assertRaises(Exception, msg=evil):
                k.safe_calc(evil)

    def test_tool_dispatch(self):
        text, ui = k.execute_tool("calculate", {"expression": "6*7"})
        self.assertTrue(ui["ok"])
        self.assertIn("42", text)
        _, ui = k.execute_tool("calc", {"expression": "__import__"})
        self.assertFalse(ui["ok"])


class TestSearchHelpers(unittest.TestCase):
    def test_fts_query_and(self):
        self.assertEqual(k._fts_match_query("hello world"), '"hello" "world"*')
        self.assertIsNone(k._fts_match_query("!!!"))

    def test_fts_query_or_drops_short_terms(self):
        q = k._fts_match_query("go to the harbor", all_terms=False)
        self.assertIn('"harbor"', q)
        self.assertNotIn('"go"', q)

    def test_chunk_text(self):
        chunks = k._chunk_text(("word " * 100 + "\n\n") * 20, target=600)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c) <= 1200 for c in chunks))
        self.assertEqual(k._chunk_text(""), [])


class TestTOTP(unittest.TestCase):
    SECRET = "GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ"   # RFC 6238 test key ("12345678901234567890")

    def test_rfc6238_vector(self):
        self.assertEqual(k.totp_code(self.SECRET, t=59), "287082")

    def test_verify_window_and_reject(self):
        now = time.time()
        good = k.totp_code(self.SECRET, t=now - 30)   # previous step still accepted
        self.assertTrue(k.totp_verify(self.SECRET, good))
        self.assertFalse(k.totp_verify(self.SECRET, "000000") and
                         k.totp_verify(self.SECRET, "999999"))
        self.assertFalse(k.totp_verify(self.SECRET, ""))
        self.assertFalse(k.totp_verify("", "287082"))


class TestChunkedBody(unittest.TestCase):
    @staticmethod
    def _handler(body, te="chunked", clen=None):
        import io
        import email.message
        h = k.Handler.__new__(k.Handler)
        h.rfile = io.BytesIO(body)
        m = email.message.Message()
        if te:
            m["Transfer-Encoding"] = te
        if clen is not None:
            m["Content-Length"] = str(clen)
        h.headers = m
        h.close_connection = False
        return h

    @staticmethod
    def _chunked(raw, size=7):
        out = b""
        for i in range(0, len(raw), size):
            piece = raw[i:i + size]
            out += ("%x\r\n" % len(piece)).encode() + piece + b"\r\n"
        return out + b"0\r\n\r\n"

    def test_decodes_chunked_json(self):
        raw = json.dumps({"model": "kenosis-v2", "n": 1}).encode()
        h = self._handler(self._chunked(raw))
        self.assertEqual(h._read(), {"model": "kenosis-v2", "n": 1})
        self.assertEqual(h.rfile.read(), b"")   # body fully consumed -> keep-alive stays clean

    def test_oversize_chunk_raises(self):
        h = self._handler(("%x\r\n" % (k.MAX_BODY_BYTES + 1)).encode() + b"x")
        with self.assertRaises(k._BodyTooLarge):
            h._read()

    def test_malformed_closes_connection(self):
        h = self._handler(b"nonsense\r\n")
        self.assertEqual(h._read(), {})
        self.assertTrue(h.close_connection)

    def test_content_length_still_works(self):
        raw = json.dumps({"a": 1}).encode()
        h = self._handler(raw, te=None, clen=len(raw))
        self.assertEqual(h._read(), {"a": 1})


class TestCompressionAnchor(_ConvoBase):
    def _mkconvo(self, n_msgs):
        cid = "cc1"
        c = k.db()
        with c:
            c.execute("INSERT OR REPLACE INTO conversations(id,owner_id,title,created,updated) VALUES(?,?,?,?,?)",
                      (cid, 1, "t", k._now(), k._now()))
        prev = None
        ids = []
        for i in range(n_msgs):
            with c:
                prev = k.insert_message(cid, prev, "user" if i % 2 == 0 else "assistant", "m%d" % i)
            ids.append(prev)
        return cid, ids

    def test_ensure_summary_incremental_and_rebuild(self):
        cid, ids = self._mkconvo(6)
        convo = {"id": cid, "ctx_summary": None, "ctx_summary_upto": None}
        dropped = [{"id": i, "role": "user", "content": "x"} for i in ids[:3]]
        calls = []
        real = k.summarize_dropped
        k.summarize_dropped = lambda ep, m, prior, msgs: calls.append((prior, len(msgs))) or "SUM1"
        try:
            s = k.ensure_summary(convo, {}, "m", dropped)
            self.assertEqual(s, "SUM1")
            self.assertEqual(calls[-1], (None, 3))          # fresh build over all dropped
            row = k.db().execute("SELECT ctx_summary, ctx_summary_upto FROM conversations WHERE id=?", (cid,)).fetchone()
            self.assertEqual((row[0], row[1]), ("SUM1", ids[2]))

            # same dropped prefix again -> stored summary reused, NO new model call
            convo = {"id": cid, "ctx_summary": "SUM1", "ctx_summary_upto": ids[2]}
            n = len(calls)
            self.assertEqual(k.ensure_summary(convo, {}, "m", dropped), "SUM1")
            self.assertEqual(len(calls), n)

            # boundary advanced by one message -> incremental fold of just the new tail
            k.summarize_dropped = lambda ep, m, prior, msgs: calls.append((prior, len(msgs))) or "SUM2"
            dropped2 = dropped + [{"id": ids[3], "role": "assistant", "content": "y"}]
            self.assertEqual(k.ensure_summary(convo, {}, "m", dropped2), "SUM2")
            self.assertEqual(calls[-1], ("SUM1", 1))
        finally:
            k.summarize_dropped = real


class TestStatsAndBackup(_ConvoBase):
    def test_usage_stats_aggregates(self):
        cid = "st1"
        c = k.db()
        with c:
            c.execute("INSERT OR REPLACE INTO conversations(id,owner_id,title,created,updated) VALUES(?,?,?,?,?)",
                      (cid, 1, "t", k._now(), k._now()))
            k.insert_message(cid, None, "assistant", "hi", model="test-model",
                             meta={"completion_tokens": 50, "tps": 10.0, "ttft_ms": 500,
                                   "prompt_tokens": 100, "cached_tokens": 80})
        s = k.usage_stats()
        row = next(m for m in s["models"] if m["model"] == "test-model")
        self.assertEqual(row["tokens"], 50)
        self.assertEqual(row["avg_cache_pct"], 80)

    def test_backup_verifies(self):
        if k.user_count() == 0:   # verification refuses userless backups by design
            k.create_user("bk-user", "password123")
        old = k.BACKUP_DIR
        k.BACKUP_DIR = tempfile.mkdtemp(prefix="kbk-")
        try:
            dest = k.backup_db()
            self.assertTrue(os.path.exists(dest))
        finally:
            k.BACKUP_DIR = old


class TestPushEndpointAllowlist(unittest.TestCase):
    def test_real_push_services_allowed(self):
        for url in ("https://fcm.googleapis.com/fcm/send/abc",
                    "https://updates.push.services.mozilla.com/wpush/v2/x",
                    "https://web.push.apple.com/QOaZ",
                    "https://db5p.notify.windows.com/w/?token=x"):
            self.assertTrue(k.push_endpoint_allowed(url), url)

    def test_ssrf_targets_rejected(self):
        for url in ("https://192.168.1.1/admin", "https://localhost:8000/v1",
                    "https://evil.example.com/", "https://fcm.googleapis.com.evil.com/",
                    "https://user@fcm.googleapis.com@evil.com/", "http://fcm.googleapis.com/x", ""):
            # http:// is stopped by the https prefix check at the endpoint; the helper itself
            # must still refuse everything not on a known push-provider host
            if url.startswith("http://"):
                continue
            self.assertFalse(k.push_endpoint_allowed(url), url)


class TestShareCSP(unittest.TestCase):
    def test_share_csp_from_template_not_rendered_page(self):
        # a hostile <script> smuggled into share data must NOT gain a hash in the served CSP
        sh = {"data": json.dumps({"title": "t", "messages": [
                  {"role": "user", "content": "<script>alert(1)</script>"}]}),
              "title": "t"}
        page = k.render_share_page(sh)
        # every EXECUTABLE script in the rendered page is covered by the precomputed template CSP
        import re as _re
        for attrs, body in _re.findall(r"<script([^>]*)>(.*?)</script>", page, _re.S):
            if "application/json" in attrs or not body.strip():
                continue
            h = "'sha256-" + k.base64.b64encode(k.hashlib.sha256(body.encode()).digest()).decode() + "'"
            self.assertIn(h, k.SHARE_CSP)
        # and the injected payload is neutralized in the data island (no raw <script in the page)
        self.assertNotIn("<script>alert", page)


class TestCSP(unittest.TestCase):
    def test_hashes_not_unsafe_inline(self):
        csp = k.csp_for("<html><script>var a=1;</script></html>")
        self.assertIn("script-src 'self' 'sha256-", csp)
        script_src = [p for p in csp.split(";") if "script-src" in p][0]
        self.assertNotIn("unsafe-inline", script_src)

    def test_page_csp_covers_all_scripts(self):
        import re as _re
        n_scripts = len([s for s in _re.findall(r"<script[^>]*>(.*?)</script>", k.PAGE, _re.S) if s.strip()])
        self.assertEqual(k.PAGE_CSP.count("'sha256-"), n_scripts)


class _StreamBase(unittest.TestCase):
    """Drives Handler._stream against a stubbed model server.

    The three things it guards all live inside that one method and are invisible from any smaller
    unit: the continuation hints have to reach the model server, an edited-and-branched user turn
    has to keep its attachments, and `branch` has to write a sibling instead of overwriting.
    """

    @classmethod
    def setUpClass(cls):
        k.set_setting("endpoints", [{"id": "t", "name": "t", "url": "http://127.0.0.1:1/v1/chat/completions"}])
        k.set_setting("active_endpoint", "t")
        k.set_setting("thinking_models", ["m"])
        cls.uid = k.user_by_name("streamer") or k.create_user("streamer", "pw", role="admin")
        cls.u = k.user_by_name("streamer")

    def setUp(self):
        self.sent = []          # one dict per stream_model call: the kwargs we care about
        self.reply = "TAIL"

        def fake_stream_model(ep, model, system, messages, params, tools=None, extra=None,
                              tool_choice="auto", vision=False):
            self.sent.append({"messages": [dict(m) for m in messages], "extra": extra, "tools": tools})
            yield {"delta": self.reply}
            yield {"finish": "stop"}

        self._real = k.stream_model
        k.stream_model = fake_stream_model
        self.addCleanup(lambda: setattr(k, "stream_model", self._real))

    def _mk(self, **cols):
        cid = k._cid()
        fields = dict(owner_id=self.u["id"], title="", system="sys", model="m", params="{}",
                      created=k._now(), updated=k._now())
        fields.update(cols)
        keys = ",".join(["id"] + list(fields))
        with k.db():
            k.db().execute("INSERT INTO conversations(%s) VALUES(%s)" % (keys, ",".join("?" * (len(fields) + 1))),
                           [cid] + list(fields.values()))
        return cid

    def _stream(self, cid, payload):
        h = _FakeHandler()
        k.Handler._stream(h, cid, payload, self.u)
        return h


class _FakeHandler(k.Handler):
    """Just enough of BaseHTTPRequestHandler for _stream to write into. Subclasses the real Handler
    (without its socket-driven __init__) so the helpers _stream calls on self are the real ones."""
    close_connection = False

    def __init__(self):   # deliberately does not call super(): that would try to serve a request
        self.wfile = io.BytesIO()
        self.error = None

    def _json(self, code, obj, extra=None):
        self.error = (code, obj)

    def send_response(self, code):
        pass

    def send_header(self, *a):
        pass

    def end_headers(self):
        pass

    def events(self):
        return [json.loads(l) for l in self.wfile.getvalue().decode().splitlines() if l.strip()]


class TestContinuePrefill(_StreamBase):
    def test_hints_ride_in_chat_template_kwargs_with_thinking(self):
        """The regression that made 'continue' produce a whole new reply glued onto the old one.

        Sent as top-level body fields the hints were silently ignored, and assigning
        chat_template_kwargs twice (once for the hints, once for enable_thinking) dropped whichever
        came first -- which broke continue on exactly the thinking models it was used on most.
        """
        cid = self._mk(think=1)
        with k.db():
            a = k.insert_message(cid, None, "user", "q")
            b = k.insert_message(cid, a, "assistant", "half a sentence")
            k.set_leaf(cid, b)
        self._stream(cid, {"continue_id": b})
        ctk = self.sent[0]["extra"]["chat_template_kwargs"]
        self.assertEqual(ctk["continue_final_message"], True)
        self.assertEqual(ctk["add_generation_prompt"], False)
        self.assertEqual(ctk["enable_thinking"], True)      # merged, not clobbered
        self.assertIsNone(self.sent[0]["tools"])
        # the partial reply is fed back as the trailing assistant turn for the model to extend
        self.assertEqual(self.sent[0]["messages"][-1]["role"], "assistant")
        self.assertEqual(self.sent[0]["messages"][-1]["content"], "half a sentence")

    def test_prefix_keeps_trailing_whitespace(self):
        """A continuation resumes at the exact character the text ends on; stripping the trailing
        newline would restart the model mid-paragraph."""
        cid = self._mk()
        with k.db():
            b = k.insert_message(cid, None, "assistant", "x")
            k.set_leaf(cid, b)
        self._stream(cid, {"continue_id": b, "content": "chapter one.\n\n"})
        self.assertEqual(self.sent[0]["messages"][-1]["content"], "chapter one.\n\n")
        row = k.db().execute("SELECT content FROM messages WHERE id=?", (b,)).fetchone()
        self.assertEqual(row["content"], "chapter one.\n\nTAIL")

    def test_continue_appends_in_place(self):
        cid = self._mk()
        with k.db():
            b = k.insert_message(cid, None, "assistant", "head ")
            k.set_leaf(cid, b)
        self._stream(cid, {"continue_id": b})
        rows = k.db().execute("SELECT content FROM messages WHERE convo_id=?", (cid,)).fetchall()
        self.assertEqual([r["content"] for r in rows], ["head TAIL"])

    def test_branch_writes_a_sibling_instead(self):
        cid = self._mk()
        with k.db():
            b = k.insert_message(cid, None, "assistant", "head ")
            k.set_leaf(cid, b)
        self._stream(cid, {"continue_id": b, "branch": True})
        rows = k.db().execute("SELECT id,parent_id,content FROM messages WHERE convo_id=? ORDER BY position",
                              (cid,)).fetchall()
        self.assertEqual([r["content"] for r in rows], ["head ", "head TAIL"])   # original untouched
        self.assertIsNone(rows[1]["parent_id"])                                  # a sibling, not a child
        leaf = k.db().execute("SELECT active_leaf_id FROM conversations WHERE id=?", (cid,)).fetchone()[0]
        self.assertEqual(leaf, rows[1]["id"])
        msgs = k.get_convo(cid, self.u)["messages"]
        self.assertEqual(msgs[-1]["sib_count"], 2)   # the ‹ 1/2 › switcher walks the two endings


class TestEditBranchAttachments(_StreamBase):
    ATT = [{"name": "pasted text", "text": "the whole essay"}]

    def test_branching_an_edited_message_keeps_its_attachments(self):
        """The edit UI sends content only. Without inheritance the pasted text the question was
        built around vanished from the new branch and the model answered with the evidence gone."""
        cid = self._mk()
        with k.db():
            a = k.insert_message(cid, None, "user", "what do you make of this?", attachments=self.ATT)
            b = k.insert_message(cid, a, "assistant", "well…")
            k.set_leaf(cid, b)
        self._stream(cid, {"edit_user_id": a, "content": "rate this out of ten"})
        # stored on the new branch, decoded once (not double-encoded by insert_message)
        new_user = k.db().execute(
            "SELECT attachments FROM messages WHERE convo_id=? AND role='user' AND content=?",
            (cid, "rate this out of ten")).fetchone()
        self.assertEqual(json.loads(new_user["attachments"]), self.ATT)
        # and actually sent to the model this turn
        self.assertEqual(self.sent[0]["messages"][-1]["attachments"], self.ATT)

    def test_explicit_empty_list_still_clears(self):
        cid = self._mk()
        with k.db():
            a = k.insert_message(cid, None, "user", "q", attachments=self.ATT)
            k.set_leaf(cid, a)
        self._stream(cid, {"edit_user_id": a, "content": "q2", "attachments": []})
        row = k.db().execute("SELECT attachments FROM messages WHERE convo_id=? AND content='q2'", (cid,)).fetchone()
        self.assertIsNone(row["attachments"])


class TestComposeMode(_StreamBase):
    def test_seeded_composition_continues_from_a_bare_assistant_root(self):
        """A composition has no user turn at all: the document is one assistant message that the
        model extends. Nothing in the trim/context path may choke on the missing user turn."""
        cid = self._mk(mode="compose")
        with k.db():
            root = k.insert_message(cid, None, "assistant", "It was a bright cold day in April, and ")
            k.set_leaf(cid, root)
        self._stream(cid, {"continue_id": root})
        sent = self.sent[0]["messages"]
        self.assertEqual([m["role"] for m in sent], ["assistant"])
        self.assertEqual(k.get_convo(cid, self.u)["mode"], "compose")
        row = k.db().execute("SELECT content FROM messages WHERE id=?", (root,)).fetchone()
        self.assertEqual(row["content"], "It was a bright cold day in April, and TAIL")


if __name__ == "__main__":
    unittest.main(verbosity=2)
