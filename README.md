# ORACLE

A small, self-contained, multi-user chat front-end for any OpenAI-compatible model endpoint
(vLLM, MLX, llama.cpp, TGI, or a hosted API). One Python file, one SQLite database, no build step.

Runs privately behind authentication so you can put a local model on the open internet (e.g. via a
Cloudflare tunnel) and share it with a few friends without exposing it to the world.

## Features

- **Full auth gate** - nothing is reachable without signing in. PBKDF2 password hashing,
  HMAC-signed session cookies, Origin/CSRF checks. First-run screen creates the admin.
- **Roles** - the admin manages endpoints/keys, global defaults, the user-visible model list,
  and users. Regular users (your friends) chat, tune their own prompts/params, and keep their own
  private characters; they never see API keys, endpoints, or each other's conversations.
- **Per-user model allowlist** - expose only the models you want to each user.
- **Characters** - saved system prompts, private per account; admins can publish site-wide ones.
- **In-chat branching** - regenerate or edit-and-resend forks the conversation into a tree you
  can navigate with sibling switchers, instead of spawning a new chat.
- **Streaming** with token usage + timing, reasoning capture, markdown rendering.
- **Continue (prefill)** - stop a stream or edit a reply, then hit *continue* to have the model
  keep generating from where the text leaves off; the new tokens are appended in place.
- **Public share** - publish any chat as a frozen snapshot at a clean, sign-in-free reading page
  (`/s/<token>`). The system prompt and later messages stay private; re-snapshot or unshare anytime.
- **Editing, folders (drag-and-drop), multi-select, search, rename, timestamps.**
- **Thumbs up/down ratings** stored in the data and included in exports (handy for RLHF datasets).
- **Export** a single chat (markdown or JSON) or every chat you own (JSON).
- **Appearance** - light/dark theme, body font (serif/sans/mono), text size, text width,
  collapsible + resizable sidebar. Stored per browser.

## Run locally

```bash
pip install -r requirements.txt
python kenosis_chat.py
# open http://localhost:8770  -> first visit creates the admin account
```

Point it at your model with environment variables (or just add the endpoint in Settings after
first launch):

```bash
KENOSIS_SEED_URL="http://your-host:8000/v1/chat/completions" \
KENOSIS_SEED_MODELS_URL="http://your-host:8000/v1/models" \
KENOSIS_SEED_KEY="your-api-key" \
python kenosis_chat.py
```

State is written next to the script: `chat.db` (plus `-wal` / `-shm`). Keep it; it holds every
account, conversation, and setting. It is gitignored.

## Docker + Cloudflare tunnel

```bash
cp docker-compose.yml docker-compose.local.yml   # edit secrets there, or edit in place
docker compose up -d --build
```

Set a long `KENOSIS_SESSION_SECRET`, keep `KENOSIS_COOKIE_SECURE=1` when served over HTTPS, and
point your Cloudflare tunnel's public hostname at `http://oracle:8770`. See `DEPLOY_CHAT.md` for
the full rundown and the security model.

## Environment variables

| var | default | purpose |
|-----|---------|---------|
| `KENOSIS_PORT` | `8770` | listen port |
| `KENOSIS_DB` | `./chat.db` | SQLite path |
| `KENOSIS_SESSION_SECRET` | generated + stored | cookie signing key (set a stable one in prod) |
| `KENOSIS_COOKIE_SECURE` | off | `1` marks the session cookie Secure (use behind HTTPS) |
| `KENOSIS_SESSION_DAYS` | `30` | session lifetime |
| `KENOSIS_ADMIN_USER` / `KENOSIS_ADMIN_PASS` | - | bootstrap an admin on startup |
| `KENOSIS_SEED_URL` / `KENOSIS_SEED_MODELS_URL` / `KENOSIS_SEED_KEY` / `KENOSIS_SEED_NAME` | localhost / empty | first-run seed endpoint |

Dependencies: `requests` (everything else is the Python standard library).

## License

MIT - see [LICENSE](LICENSE).

