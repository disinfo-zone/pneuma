# Deploying kenosis chat

A single-file Python app (`kenosis_chat.py`) with an embedded UI. Auth + multi-user, data in
one SQLite file. No external services required beyond the model endpoint(s) you configure.

## Run locally (dev)

```bash
pip install requests
python kenosis_chat.py
# open http://localhost:8770  -> first visit shows a one-time admin setup screen
```

State written next to the script: `chat.db` (+ `chat.db-wal`, `chat.db-shm`).
Any conversations in `./chat_conversations/*.json` are imported to the admin on first setup.

## Run with Docker

```bash
# 1. edit docker-compose.yml: set KENOSIS_SESSION_SECRET, KENOSIS_ADMIN_USER/PASS
# 2. build + start
docker compose up -d --build
# 3. open http://<host>:8770
```

The DB persists in `./data/chat.db` on the host (the `./data:/data` volume).

## Behind a Cloudflare tunnel

1. Create a tunnel in Cloudflare Zero Trust → Tunnels.
2. Add a public hostname (e.g. `chat.yourdomain.com`) routing to `http://kenosis-chat:8770`
   (or `http://<lan-ip>:8770` if cloudflared runs outside this compose stack).
3. Keep **`KENOSIS_COOKIE_SECURE: "1"`** — Cloudflare serves the app over HTTPS, so the secure
   cookie is correct. (Set `0` only for plain-http local testing.)
4. Optional extra layer: put **Cloudflare Access** in front of the hostname so only your Google
   accounts / emails can even reach the login page. The app's own auth still gates everything.

If cloudflared is the *only* way in, drop the `ports:` mapping in `docker-compose.yml` (or bind it
to `127.0.0.1:8770:8770`) so the app is not exposed directly on the LAN.

## Security model (what "safe" means here)

- **Full gate:** every page and API requires a valid session; unauthenticated `/` redirects to `/login`.
- **Passwords:** PBKDF2-HMAC-SHA256, 240k iterations, per-user salt. No plaintext stored.
- **Sessions:** HMAC-signed cookies (`HttpOnly`, `SameSite=Lax`, `Secure` in prod), 30-day expiry.
  Set a stable `KENOSIS_SESSION_SECRET` so sessions survive restarts.
- **CSRF:** SameSite=Lax + an Origin/Referer-vs-Host check on every state-changing request.
- **Roles:** `admin` manages endpoints/keys, global defaults, the user-visible model list, and users.
  `user` (your friends) can chat, edit their own system prompt + sampler params, and manage their
  own private characters — but never see API keys, endpoints, or other users' chats.
- **Model exposure:** admins choose a global "user models" allowlist (Settings → User models), with
  optional per-user overrides (Settings → Users). Enforced server-side on create/switch/stream.
- **Characters:** private to each account by default; admins may opt a character to *site-wide*
  (visible to everyone, read-only to non-owners).

There is intentionally no email/password-recovery flow. To reset a forgotten password, an admin
resets it in Settings → Users; to reset the admin, delete `chat.db` (loses data) or update the row
directly via `sqlite3 chat.db`.

## Environment variables

| var | default | purpose |
|-----|---------|---------|
| `KENOSIS_PORT` | `8770` | listen port |
| `KENOSIS_DB` | `./chat.db` | SQLite path |
| `KENOSIS_SESSION_SECRET` | generated + stored in db | cookie signing key (set a stable one in prod) |
| `KENOSIS_COOKIE_SECURE` | off | `1` marks the session cookie `Secure` (use behind HTTPS) |
| `KENOSIS_SESSION_DAYS` | `30` | session lifetime |
| `KENOSIS_ADMIN_USER` / `KENOSIS_ADMIN_PASS` | — | bootstrap an admin on startup (optional) |
| `KENOSIS_SITUATION` | `1` | append the current date (and, with tools off, a no-internet note) to the system prompt for grounding. Day-granular so it preserves prefix caching; set `0` to disable |

The model endpoints themselves (URLs + API keys) are configured in-app under Settings → Endpoints,
not via env, so they live in the database with the rest of the state.
