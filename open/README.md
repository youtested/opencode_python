# opencode_py

A pure-Python reimplementation of the [opencode](https://github.com/anomalyco/opencode)
coding agent, built for **32-bit ARM (armv7) Android phones running Termux** — but it
also runs fine on any desktop Linux/macOS machine.

- **Zero binary dependencies**: `httpx`, `rich`, `textual`, `click`, `platformdirs`,
  all pure-Python wheels. No pydantic-core / numpy / Rust — installs cleanly on armv7.
- **Free models first**: ships with the free OpenCode Zen model list and free-tier
  providers (Groq, Cerebras, Google AI Studio, OpenRouter `:free`, NVIDIA, Mistral,
  GitHub Models, SambaNova, Together) with automatic **rotation/failover** on
  rate limits.
- **Full terminal TUI** (Textual) plus a `--no-tui` line-mode fallback for low-RAM.

```
                build  plan     (agents)             deepseek-v4-flash-free  (model)
──────────────────────────────────────────────────────────────────────────
> opencode_py
```

---

## Quick start (desktop)

```bash
pip install -e .           # installs the opencode-py command
opencode-py                # launch the TUI
# or headless:
echo 'create hello.py that prints hi' | opencode-py --no-tui
```

Then inside the TUI:

1. Type `/connect`, pick **OpenCode Zen**, and paste your key.
   Get a free key at <https://opencode.ai/auth> (or skip it — Zen's launch free
   models run keyless for a limited time; `--check` will tell you).
2. Type a prompt and watch the agent stream, run tools, and edit files.
3. `Tab` switches between `build` and `plan` agents; `Ctrl+C` interrupts a stream.

Useful flags:

```bash
opencode-py --check                 # ping configured providers, report OK/fail
opencode-py --models                # list live Zen models (free first)
opencode-py --print-config          # show resolved config
opencode-py --model big-pickle      # override model
opencode-py --no-tui -m 'message'   # one-shot headless
```

---

## Install on the phone (Termux, armv7 or arm64)

**Prerequisites**: a 32-bit Android phone with Termux installed
(also works on 64-bit devices).

```bash
pkg install -y git
git clone https://github.com/anomalyco/opencode_py.git   # or your fork's URL
cd opencode_py
bash install_termux.sh
opencode-py
```

`install_termux.sh` installs `python`, `git`, `ripgrep`, `openssh`, then the
pure-Python requirements and the `opencode-py` command itself.

> If you host your own fork, edit `REPO_URL` at the top of `install_termux.sh`
> or pass it: `REPO_URL=... bash install_termux.sh`.

---

## Getting free API keys

All free-tier providers use the standard `chat/completions` HTTP API, so one
generic client serves them all. Each needs its own key, set in `auth.json`
(via `/connect`) or as an env var:

| Provider | Key env var | Get a free key |
|---|---|---|
| OpenCode Zen (free launch models) | `OPENCODE_API_KEY` | <https://opencode.ai/auth> |
| Groq | `GROQ_API_KEY` | <https://console.groq.com/keys> |
| Cerebras | `CEREBRAS_API_KEY` | <https://cloud.cerebras.ai/> |
| Google AI Studio (Gemini) | `GEMINI_API_KEY` | <https://aistudio.google.com/apikey> |
| OpenRouter | `OPENROUTER_API_KEY` | <https://openrouter.ai/keys> |
| NVIDIA NIM | `NVIDIA_API_KEY` | <https://build.nvidia.com/> |
| Mistral | `MISTRAL_API_KEY` | <https://console.mistral.ai/> |
| GitHub Models | `GITHUB_TOKEN` | <https://github.com/settings/tokens> |
| SambaNova | `SAMBANOVA_API_KEY` | <https://cloud.sambanova.ai/> |
| Together | `TOGETHER_API_KEY` | <https://api.together.ai/> |
| Anthropic (paid) | `ANTHROPIC_API_KEY` | <https://console.anthropic.com/> |
| OpenAI (paid) | `OPENAI_API_KEY` | <https://platform.openai.com/api-keys> |
| Ollama (local) | — (no key) | <https://ollama.com/> |

Key precedence: **env var > config > `auth.json`** (`~/.local/share/opencode_py/auth.json`,
chmod 0600). Keys are never logged or committed.

### Rotation / failover

When one provider is rate-limited (`429`) or errors, opencode_py automatically
tries the next lane in your `rotation` list. Configure it in `opencode.json`:

```json
{
  "provider": "opencode",
  "model": "deepseek-v4-flash-free",
  "rotation": [
    { "provider": "opencode", "model": "deepseek-v4-flash-free" },
    { "provider": "groq", "model": "llama-3.3-70b-versatile" },
    { "provider": "cerebras", "model": "llama-3.3-70b" }
  ]
}
```

When a failover happens you'll see a toast / stderr notice like
`[rate limited - switched to groq/llama-3.3-70b-versatile]`.

---

## Config

opencode_py reads `opencode.json` from the current (or project root) directory,
merged over `~/.config/opencode_py/opencode.json`. Supported keys include:

```json
{
  "provider": "opencode",
  "model": "deepseek-v4-flash-free",
  "agent": "build",
  "theme": "opencode-dark",
  "model_read_timeout": 300,
  "permission": { "bash": "allow", "edit": "ask", "webfetch": "ask" },
  "tools": { "disable": ["webfetch"] },
  "autoEdit": true,
  "autoYes": false,
  "rotation": [ { "provider": "opencode", "model": "deepseek-v4-flash-free" } ]
}
```

Slash commands inside the TUI:

```
/help  /new  /init  /clear  /undo  /models  /connect  /permissions
/config  /theme  /resume  /sessions  /agent  /model  /compact  /exit
```

- `/undo` reverts the most recent `edit`/`write` tool call (file-level snapshot,
  last 20 kept).
- `/config validate` checks provider/rotation/permission values for obvious
  mistakes.
- `/compact` drops conversation history down to the last turn to save context.
- `model_read_timeout` (seconds) bounds the gap between streaming chunks — the
  default 300s gives slow reasoning/free models room to think. Reasoning models
  (deepseek, o1/o3, claude-sonnet-4, gemini-2.5, grok-3, ...) automatically get
  a longer 900s gap floor so a long silent think isn't killed by the timeout
  (which would otherwise interrupt their chain of thought and make the model
  look like it "stopped responding"). Text streams live to the screen as it's
  generated (no full-response buffering); an eager `Thinking…` bubble appears
  the instant you hit Enter, and reasoning models collapse to a `+ Thought for
  X.Xs` header when done. Transient timeouts retry automatically (exponential
  backoff, honoring `Retry-After`) before surfacing.
- `auto_retry` (default true) controls that transient-failure retry;
  `auto_retry_count` (default 50) sets how many retries are attempted before the
  error is surfaced, with a `↻ connection dropped — retrying…` hint in the status
  bar. Retries stay on the same model: if the model already ran tools when the
  connection dropped, a short "keep going" nudge is sent so it continues instead
  of replaying the whole task.

### Plugins & MCP (extras)

Config-driven custom tools, no SDK needed:

```json
{
  "plugins": ["my_tools"],
  "mcpServers": {
    "filesystem": { "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"] }
  }
}
```

- **Plugins**: each module in `plugins` should expose `TOOLS = [{"name", "description",
  "parameters", "run"}...]`. Loaded from the current directory / `PYTHONPATH`.
- **MCP servers**: any stdio JSON-RPC server works; each remote tool appears as
  `mcp__<server>__<tool>`. Pure-Python client, no SDK dependency.

---

## Low-RAM tips (old phones)

- Prefer `*-free` / `flash` models (the default free list).
- `--no-tui` drops the Textual UI, saving most of the memory:
  `echo 'fix the tests' | opencode-py --no-tui`
- Start sessions with `/new` or `/clear`; context is trimmed to your configured
  budget automatically (`util/truncate.py`).
- Tool output is capped: `read` (2000 lines / 50 KB), `glob`/`grep` (100 results),
  `webfetch` (5 MB), `bash` (output truncated).
- If output looks frozen, run with `PYTHONUNBUFFERED=1`.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Cannot connect to host` / TLS errors | Your network/certs are stale: `pkg upgrade` then `python -m pip install --upgrade certifi`. |
| `429 Too Many Requests` | Rate-limited — opencode_py auto-rotates to the next lane; add more lanes or wait ~30s. |
| `Out of memory` | Use `--no-tui`, a smaller model, or `/clear` to trim history. |
| TUI renders but no text streams | Old terminal — run headless (`--no-tui`) or use `PYTHONUNBUFFERED=1`. |
| `unknown model` | Run `opencode-py --models`; free launch models can appear/disappear. |
| `pip` says a package is missing | Network — run `pip install -r requirements.txt` again. |

---

## Development

```bash
python -m pytest        # 100+ unit tests (headless, no network needed)
python -m opencode_py.main --check
```

### Project layout

```
opencode_py/
  main.py          CLI entry point (argparse/click)
  config.py        opencode.json loading + deep merge
  auth.py          API key storage (env > config > auth.json 0600)
  commands.py      slash command registry
  permission.py    ask/allow/deny permission engine
  session.py       JSON conversation persistence
  providers/       zen, openai_compat, anthropic, ollama, rotation
  agent/           loop, system prompt, messages, tool-call parsing
  tools/           bash, read, write, edit, glob, grep, webfetch, todo
  tui/             Textual app, chat view, model picker, connect, dialogs
tests/             pytest suite
docs/notes.md      extracted opencode reference notes
```

See `plan.txt` for the full roadmap and design rationale.

## License

MIT. This is an independent, from-scratch Python implementation; it is not the
official opencode project.
