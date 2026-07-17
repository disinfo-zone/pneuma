# Deploying kenosis chat

A single-file Python app (`kenosis_chat.py`) with an embedded UI. Auth + multi-user, data in
one SQLite file. No external services required beyond the model endpoint(s) you configure.

## Run locally (dev)

```bash
pip install -r requirements.txt
python kenosis_chat.py
# open http://localhost:8770  -> first visit shows a one-time admin setup screen
```

State written next to the script: `chat.db` (+ `chat.db-wal`, `chat.db-shm`).
Any conversations in `./chat_conversations/*.json` are imported to the admin on first setup.

## Run with Docker

```bash
# 1. put KENOSIS_SESSION_SECRET (and seed endpoint/key) in local.env
# 2. the container runs as UID 10001 (non-root) — make the volumes writable once:
sudo mkdir -p data backups && sudo chown -R 10001:10001 data backups
# 3. build + start
docker compose up -d --build
# 4. open http://<host>:8770
```

The DB persists in `./data/chat.db` on the host (the `./data:/data` volume). Automated
backups land in `./backups` (separate volume, so a copy of `./data` alone never includes
them) and rotate per `KENOSIS_BACKUP_KEEP`.

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
  Set a stable `KENOSIS_SESSION_SECRET` so sessions survive restarts — it is read from the
  environment only and **never written to the database**, so DB copies/backups can't forge
  sessions. Changing a password (your own, or an admin reset) bumps a per-user session
  version, revoking every other signed-in device.
- **Login throttling:** failed sign-ins are limited per source IP *and* per target account,
  so a spoofed/rotating `CF-Connecting-IP` header can't brute-force one username. Set
  `KENOSIS_TRUST_PROXY=0` if clients can reach the port without going through Cloudflare.
- **API keys:** endpoint keys are stored server-side and never sent to the browser; the
  settings UI sees a `__stored__` placeholder and only a changed value overwrites the key.
- **Request caps:** bodies above `KENOSIS_MAX_BODY_MB` (default 32 MB) are rejected before
  auth, and malformed `Content-Length` headers are handled defensively.
- **Container:** runs as a non-root user (UID 10001).
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
| `KENOSIS_SESSION_SECRET` | ephemeral | cookie signing key — set a stable one in prod; never persisted to the DB (legacy DB copies are scrubbed when the env var is set) |
| `KENOSIS_COOKIE_SECURE` | off | `1` marks the session cookie `Secure` (use behind HTTPS) |
| `KENOSIS_SESSION_DAYS` | `30` | session lifetime |
| `KENOSIS_ADMIN_USER` / `KENOSIS_ADMIN_PASS` | — | bootstrap an admin on startup (optional) |
| `KENOSIS_BACKUP_HOURS` / `KENOSIS_BACKUP_KEEP` / `KENOSIS_BACKUP_DIR` | `24` / `7` / `./backups` | automated `VACUUM INTO` backups: cadence, retention, location; `0` hours disables |
| `KENOSIS_TRUST_PROXY` | `1` | trust `CF-Connecting-IP`/`X-Forwarded-For` for throttle keys and logs; `0` to use the socket peer only |
| `KENOSIS_PUBLIC_URL` | `https://delphi.disinfo.zone` | absolute base URL for share links and OG tags |
| `KENOSIS_MAX_BODY_MB` | `32` | hard request-body cap, enforced before auth |
| `KENOSIS_SITUATION` | `1` | append the current date (and, with tools off, a no-internet note) to the system prompt for grounding. Day-granular so it preserves prefix caching; set `0` to disable |

The model endpoints themselves (URLs + API keys) are configured in-app under Settings → Endpoints,
not via env, so they live in the database with the rest of the state.

## Backups: verify & restore

Backups are written by `VACUUM INTO` every `KENOSIS_BACKUP_HOURS` (default 24h) to
`KENOSIS_BACKUP_DIR`, and every copy is verified at write time (`PRAGMA integrity_check`
plus a sanity check that it contains users) — a copy that fails verification is deleted
and logged as an ERROR, so watch the logs for `backup verification FAILED`.

To restore:

```bash
# 1. stop the app so nothing writes to the live DB
docker compose stop kenosis-chat

# 2. move the damaged DB aside (keep it for forensics)
mv data/chat.db data/chat.db.broken
rm -f data/chat.db-wal data/chat.db-shm     # stale WAL/SHM must not outlive the main file

# 3. copy the chosen backup into place (backups are plain SQLite files)
cp backups/chat-YYYYMMDD-HHMMSS.db data/chat.db
sudo chown 10001:10001 data/chat.db

# 4. start again — migrations and the FTS index rebuild automatically if needed
docker compose start kenosis-chat
```

Everything lives in that one file (accounts, chats, settings, knowledge, push
subscriptions). Sessions survive a restore as long as `KENOSIS_SESSION_SECRET` is
unchanged. If a user enabled TOTP and lost their authenticator, clear it manually:
`sqlite3 data/chat.db "UPDATE users SET totp_secret=NULL WHERE username='NAME'"`.
