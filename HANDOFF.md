# Spinning up ORACLE (pneuma) - quick start

ORACLE is a single-file Python chat app (`kenosis_chat.py`) for any OpenAI-compatible model
endpoint. One file, one SQLite database (`chat.db`), no build step. Auth-gated and multi-user.

You need: an OpenAI-compatible endpoint to point it at (its URL + optional API key), and either
Docker or Python 3.10+.

---

## Option A - Docker (use this to host it)

1. Edit `docker-compose.yml` and set, under `environment:`
   - `KENOSIS_SESSION_SECRET` - any long random string. Generate one:
     `python -c "import os,base64;print(base64.b64encode(os.urandom(32)).decode())"`
   - `KENOSIS_ADMIN_USER` / `KENOSIS_ADMIN_PASS` - your admin login (or delete these two lines
     and create the admin on the first web visit instead)
   - Add your model endpoint:
     ```
     KENOSIS_SEED_URL: "http://YOUR-MODEL-HOST:8000/v1/chat/completions"
     KENOSIS_SEED_MODELS_URL: "http://YOUR-MODEL-HOST:8000/v1/models"
     KENOSIS_SEED_KEY: "your-api-key-or-leave-blank"
     ```
   - Keep `KENOSIS_COOKIE_SECURE: "1"` if it will be served over HTTPS (it will, behind a
     Cloudflare tunnel). Set it to `"0"` only for plain-http local testing.

2. `docker compose up -d --build`

3. Open `http://SERVER-IP:8770` (or your tunnel hostname). Sign in / create the admin.

The database persists in `./data/chat.db` on the host. Back that file up; it holds all accounts,
chats, and settings.

---

## Option B - bare Python (quick local test)

```bash
pip install -r requirements.txt
python kenosis_chat.py
# open http://localhost:8770  -> first visit creates the admin
```

Point it at your model either by setting the same `KENOSIS_SEED_URL` / `KENOSIS_SEED_KEY` env
vars before running, OR just add the endpoint in the UI: Settings -> Endpoints after you log in.

---

## Cloudflare tunnel (to expose it on the internet)

1. Create a tunnel in the Cloudflare Zero Trust dashboard.
2. Route a public hostname (e.g. `oracle.example.com`) to `http://SERVER-IP:8770`
   (or `http://oracle:8770` if you uncomment the `cloudflared` service in `docker-compose.yml`).
3. Make sure `KENOSIS_COOKIE_SECURE=1` (Cloudflare serves HTTPS).
4. Optional: put Cloudflare Access in front of the hostname for an extra login layer.

Full deploy details + the security model are in `DEPLOY_CHAT.md`.

---

## After it's up (admin tasks, all in the web UI)

- **Add a friend:** Settings -> Users -> add user.
- **Choose which models users can pick:** Settings -> User models (admins always see all).
- **Set the default model / system prompt:** Settings -> Defaults.
- **Add or edit model endpoints + API keys:** Settings -> Endpoints.

## Notes

- All state is in `chat.db`. Deleting it wipes everything and reopens first-run setup.
- No secrets are committed to this repo; real endpoints/keys live only in `chat.db` (gitignored)
  or in the env vars you set.
- Only dependency is `requests`; everything else is the Python standard library.
