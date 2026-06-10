"""Tests for the pure-Python core of kenosis_chat.py: auth tokens, tree logic, share-snapshot
privacy stripping, schema migration, and the small parsing helpers.

Run:  python -m pytest test_kenosis.py    (or)    python -m unittest test_kenosis -v

Uses a throwaway SQLite file per run; never touches a real chat.db.
"""

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
