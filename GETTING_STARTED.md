# Getting Started with Cortheon v2

Run **execution-verified `B=4` parallel consensus** on top of any local OpenAI-compatible model (**TensorFold**, **Ollama**, **llama.cpp**, **vLLM**, **LM Studio**) and invoke **`/cortheon`** directly inside **OMP (`oh-my-pi`)**, **Pi (`pi-coding-agent`)**, **OpenCode**, **Claude Code**, **Cursor**, or **Codex**.

## 30-Second Quick Start → Use `/cortheon`

```sh
# 1. Clone & install Cortheon
git clone https://github.com/ashhart/Cortheon.git && cd Cortheon
uv venv .venv && uv pip install --python .venv/bin/python -e .

# 2. Install `/cortheon` across OMP, Pi, OpenCode, Claude Code, Cursor & Codex
.venv/bin/cortheon install-commands --harness all

# 3. Start the proxy against your local model (skip if running TensorFold native --cortheon --cortheon-exec)
.venv/bin/cortheon proxy --upstream http://127.0.0.1:8080/v1 --port 18652
```

Then open your coding harness and type **`/cortheon`** to get going:

```text
# Verify hard math, algorithms, or systems sizing via B=4 execution consensus:
/cortheon 7 servers sit in a ring (0..6); start at 1 (visited); move +1 mod 7 w.p. 2/3 or -1 w.p. 1/3, except at 0 teleport uniformly to 0..6. Exact rational expected ticks to visit all 7?

# Verify invariants first via B=4 consensus, then hand off straight to the agent to build & test (OMP / Pi):
/cortheon build Build ssi_kv.py implementing true MVCC Serializable Snapshot Isolation with SIREAD locks and a 1,000-schedule serializability fuzzer.
```

*(Set `export CORTHEON_URL=http://127.0.0.1:18652/v1` — or `:18651/v1` for TensorFold native — if your server runs on another host/port.)*

---

## Step 1 — Pick Your Local Runtime Mode

### Option A: TensorFold Native (`--cortheon --cortheon-exec`, Recommended)

If you serve your local model with **TensorFold**, Cortheon v2 runs **in-process** with zero proxy hop — giving interactive coding tools full native support alongside `B=4` execution-verified consensus (`ReasonForge`, `CodeForge v5`, `ThinkForge`):

```sh
# 1. Create a lightweight Python sandbox venv for Cortheon's verified execution
python3 -m venv ~/.cortheon-venv
~/.cortheon-venv/bin/pip install sympy numpy z3-solver

# 2. Serve with --parallel 8 (leaves 4 lanes free for your agent harness even during B=4 consensus!)
tensorfold-native serve /path/to/your-model \
  --host 0.0.0.0 --port 18651 \
  --parallel 8 --context 65536 \
  --thinking-budget 1024 --thinking-reserve 2048 \
  --prompt-cache-gib 12 --loop-guard \
  --cortheon --cortheon-exec \
  --cortheon-python ~/.cortheon-venv/bin/python3 \
  --name local-model
```

> **Why `--parallel 8` matters:** Cortheon v2 fires `B=4` parallel streams on verification turns. Running `--parallel 8` reserves 4 free GPU lanes so background agent turns, titles, and edits stay at `queued=0.000s` instead of waiting behind a `B=4` wave. On a 25-digit exact 7-server asymmetric Markov cover-time problem, combining `--parallel 8`, `low`-effort PAL slots, 20s sandbox timeout, and atomic early-abort cut wall time **`7.5×`** (**`271s → 36s`**, hitting `wave1_consensus (2 PAL)` and cancelling the 4th stream automatically).

### Option B: Universal Proxy (`cortheon proxy` — Ollama, llama.cpp, vLLM, LM Studio, TensorFold)

Works with **any** local OpenAI-compatible server:

```sh
# Install Cortheon
git clone https://github.com/ashhart/Cortheon.git
cd Cortheon
uv venv .venv && uv pip install --python .venv/bin/python -e .

# 60-second live Vanilla vs Cortheon duel on your local model
.venv/bin/cortheon showdown --url http://127.0.0.1:8080/v1

# Start the universal OpenAI + Anthropic proxy on :18652
# (Ollama: http://127.0.0.1:11434/v1 | llama.cpp: http://127.0.0.1:8080/v1 | LM Studio: http://127.0.0.1:1234/v1)
.venv/bin/cortheon proxy --upstream http://127.0.0.1:8080/v1 --port 18652
```

Virtual model IDs exposed at `http://127.0.0.1:18652/v1`:
- `cortheon` — auto-routes `fast` / `code` / `reason` / `think`
- `cortheon:reason` — forces `B=4` PAL Python/SymPy/Z3 consensus with Wave-1 early-abort
- `cortheon:code` — forces `CodeForge v5` Behavioral Differential Consensus + Red-Teaming
- `cortheon:think` — forces `ThinkForge` council + Python physics/invariant verifier
- `cortheon:deep` — deeper thinking budgets
- `cortheon:fast` — direct `B=1` passthrough
- **Client-side tool loops (`tools`/`functions`):** `cortheon proxy` transparently streams OpenAI tool-calling turns straight through to upstream so coding harnesses (**OMP**, **Pi**, **OpenCode**, **Cursor**) work out of the box while non-tool/planning/advisor turns and `cortheon solve` engage full `B=4` consensus!

---

## Step 2 — Install `/cortheon` Across Every Harness (One Command)

Install the native **`/cortheon`** slash command across **OMP (`oh-my-pi`)**, **Pi (`pi-coding-agent`)**, **OpenCode**, **Claude Code**, **Cursor**, and **Codex** in one command:

```sh
# Install /cortheon globally across ALL supported harnesses:
.venv/bin/cortheon install-commands --harness all

# Or target a specific harness, OMP profile, or project directory:
.venv/bin/cortheon install-commands --harness omp --profile cortheon
.venv/bin/cortheon install-commands --harness opencode,claude,cursor --project /path/to/repo
```

What you get in every harness:
- **OMP & Pi (`cortheon.ts` extension):** Native **`/cortheon <question>`** with a live inline TUI progress widget, **`/cortheon build <task>`** (verifies via `B=4` consensus first, then hands the verified ground truth straight to the agent to implement & test), plus a first-class **`cortheon_verify`** tool!
- **OpenCode (`.opencode/command/cortheon.md`), Claude Code (`.claude/commands/cortheon.md`), Cursor (`.cursor/commands/cortheon.md`):** Native **`/cortheon <question or task>`** slash command!
- **Codex / Generic Agents (`.agents/skills/cortheon/SKILL.md`):** Full Cortheon v2 skill pack + phase micro-cards!

---

## Step 3 — Connect Your Coding Agent Provider

### 1. OMP (`oh-my-pi`) — Isolated Profile + `Ctrl+P` Cortheon Toggle + Live `tmux` Split

Keep your primary `~/.omp/agent/` config completely untouched by using an isolated `--profile cortheon` profile.

**`~/.omp/profiles/cortheon/agent/models.yml`** (replace `127.0.0.1:18651` with `127.0.0.1:18652` if using `cortheon proxy`, or your LAN server host):

```json
{
  "providers": {
    "cortheon-local": {
      "baseUrl": "http://127.0.0.1:18651/v1",
      "api": "openai-completions",
      "apiKey": "local",
      "authHeader": false,
      "compat": {
        "supportsStore": false,
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": true,
        "thinkingFormat": "openai",
        "reasoningDisableMode": "none-effort",
        "reasoningEffortMap": {
          "minimal": "low",
          "medium": "high",
          "xhigh": "max"
        },
        "supportsUsageInStreaming": true,
        "supportsStrictMode": false,
        "maxTokensField": "max_tokens",
        "replayReasoningContent": true
      },
      "models": [
        {
          "id": "local-model",
          "name": "Local Model + Cortheon v2",
          "reasoning": true,
          "input": ["text"],
          "contextWindow": 65536,
          "maxTokens": 32768,
          "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 }
        },
        {
          "id": "local-model-raw",
          "name": "Local Model Raw Baseline (cortheon=false)",
          "reasoning": true,
          "input": ["text"],
          "contextWindow": 65536,
          "maxTokens": 32768,
          "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
          "compat": {
            "extraBody": { "model": "local-model", "cortheon": false }
          }
        }
      ]
    }
  }
}
```

**`~/.omp/profiles/cortheon/agent/config.yml`**:

```yaml
setupVersion: 3
compaction:
  thresholdPercent: -1
enabledModels:
  - cortheon-local/local-model
  - cortheon-local/local-model-raw
modelRoles:
  default: cortheon-local/local-model
  advisor: cortheon-local/local-model
  plan: cortheon-local/local-model
defaultThinkingLevel: xhigh
providers:
  maxInFlightRequests:
    cortheon-local: 1
```

**Launch OMP (with instant `Ctrl+P` toggle between Cortheon v2 and Raw Baseline):**

```sh
omp --profile cortheon --model cortheon-local/local-model
```

**Optional — Let OMP invoke `B=4` execution-verified consensus mid-session via `AGENTS.md`:**

```markdown
# Cortheon Consensus Helper
For hard math, algorithms, or systems sizing where single-pass reasoning can slip,
run `cortheon solve --url http://127.0.0.1:18651/v1 "<concise self-contained prompt>"`
synchronously before implementing, then verify your code with real tests.
```

**Bonus — Split-pane `tmux` cockpit (interactive OMP on the left, live Cortheon trace on the right):**

```sh
tmux new-session -d -s cortheon-omp "omp --profile cortheon --model cortheon-local/local-model"
tmux set-option -t cortheon-omp mouse on
tmux split-window -h -p 38 -t cortheon-omp:0.0 "tail -n 30 -f /path/to/server-or-proxy.log"
tmux select-pane -t cortheon-omp:0.0
tmux attach -t cortheon-omp
```

---

### 2. Pi (`pi-coding-agent`)

Add Cortheon as a custom OpenAI-compatible provider in **`~/.pi/agent/models.json`**:

```json
{
  "providers": {
    "cortheon": {
      "baseUrl": "http://127.0.0.1:18652/v1",
      "api": "openai-completions",
      "apiKey": "local",
      "compat": {
        "supportsStore": false,
        "supportsDeveloperRole": false,
        "supportsUsageInStreaming": true,
        "maxTokensField": "max_tokens"
      },
      "models": [
        {
          "id": "cortheon",
          "name": "Cortheon v2 (Auto-Route)",
          "reasoning": true,
          "input": ["text"],
          "contextWindow": 65536,
          "maxTokens": 32768,
          "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 }
        },
        {
          "id": "cortheon:reason",
          "name": "Cortheon v2 ReasonForge (B=4 PAL)",
          "reasoning": true,
          "input": ["text"],
          "contextWindow": 65536,
          "maxTokens": 32768,
          "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 }
        },
        {
          "id": "cortheon:code",
          "name": "Cortheon v2 CodeForge v5",
          "reasoning": true,
          "input": ["text"],
          "contextWindow": 65536,
          "maxTokens": 32768,
          "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 }
        }
      ]
    }
  }
}
```

Then launch Pi and install the `/cortheon` slash command + `cortheon_verify` tool:

```sh
.venv/bin/cortheon install-commands --harness pi
pi --model cortheon/cortheon
```

*(If running TensorFold native on `:18651`, point `baseUrl` directly at `http://127.0.0.1:18651/v1` and set `"id"` to your `--name` model ID.)*

---

### 3. OpenCode

Add a custom OpenAI-compatible provider in **`opencode.json`** (project root or `~/.config/opencode/opencode.json`):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "cortheon": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Cortheon v2 (Local)",
      "options": {
        "baseURL": "http://127.0.0.1:18652/v1",
        "apiKey": "local"
      },
      "models": {
        "cortheon": {
          "name": "Cortheon v2 (Auto-Route)",
          "limit": { "context": 65536, "output": 32768 }
        },
        "cortheon:code": {
          "name": "Cortheon v2 CodeForge v5",
          "limit": { "context": 65536, "output": 32768 }
        },
        "cortheon:think": {
          "name": "Cortheon v2 ThinkForge",
          "limit": { "context": 65536, "output": 32768 }
        }
      }
    }
  },
  "model": "cortheon/cortheon"
}
```

Start OpenCode in your repo:

```sh
opencode
```

*(For TensorFold native `--cortheon --cortheon-exec`, set `baseURL` to `http://127.0.0.1:18651/v1` and replace `"cortheon"` with your served model name.)*

---

### 4. Claude Code / Anthropic SDK

`cortheon proxy` also exposes `/v1/messages` (Anthropic Messages API) with SSE keepalive heartbeats:

```sh
# Direct Anthropic SDK / non-tool synthesis queries:
ANTHROPIC_BASE_URL=http://127.0.0.1:18652 ANTHROPIC_API_KEY=local \
  curl -s http://127.0.0.1:18652/v1/messages \
    -H "content-type: application/json" \
    -d '{"model":"cortheon","max_tokens":4096,"messages":[{"role":"user","content":"Find all integers n such that n^2 + 19n + 92 is a perfect square."}]}'
```

For full autonomous tool-using coding inside Claude Code or Codex, install the `/cortheon` slash command & skill so you can type `/cortheon <prompt>` anytime:

```sh
.venv/bin/cortheon install-commands --harness claude,codex
```

---

### 5. Cursor / Continue / OpenAI Python SDK

Point any OpenAI-compatible client at `http://127.0.0.1:18652/v1` (or TensorFold native at `http://127.0.0.1:18651/v1`):

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:18652/v1", api_key="local")
resp = client.chat.completions.create(
    model="cortheon",
    messages=[{"role": "user", "content": "How many length-14 {A,B,C} strings avoid 'AA', 'BB', 'ABC' with #A > #B?"}],
)
print(resp.choices[0].message.content)
```

---

## Step 3 — Verify It Works in 60 Seconds

Try a problem no frozen local model can compute in its head — an asymmetric 7-server Markov cover time with a teleport trap (exact answer is a 25-digit numerator over a 24-digit denominator: `2630600114653531854763423 / 183488738276457059168800`):

```sh
.venv/bin/cortheon solve --url http://127.0.0.1:18651/v1 \
  "7 servers sit in a ring (0..6). A token starts at server 1 (server 1 counts as visited). Each tick it moves +1 mod 7 with prob 2/3 or -1 mod 7 with prob 1/3, except at server 0 it teleports uniformly to one of 0..6 (including 0). Find the exact rational expected ticks E[T] until all 7 servers are visited."
```

You should see Wave-1 PAL scripts lock exact consensus (`2630600114653531854763423/183488738276457059168800`) and cancel remaining straggler streams automatically!
