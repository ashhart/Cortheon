# Security & Sandboxing Model

Cortheon executes model-generated Python code during verification (`ReasonForge`, `CodeForge`, `ThinkForge`) and runs commands inside repositories during experimental agent modes (`agent`, `dev`).

## Default OS Sandbox (`flash_forge/sandbox.py`)

By default, `run_python(...)` executes untrusted model-written code inside an isolated ephemeral temporary directory with OS-level containment:

| Platform | Default Backend | Protections Enforced |
|---|---|---|
| **macOS** | `macos-seatbelt` (`/usr/bin/sandbox-exec`) | Denies **all network access** (`deny network*`); denies **all filesystem writes** outside the ephemeral temp directory (`/dev/null`, `/dev/tty`, `/dev/dtracehelper` excepted); denies reads of sensitive user credential paths (`~/.ssh`, `~/.aws`, `~/.gnupg`, `~/.kube`, `~/.omp`, `~/.pi`, `~/.config`, `~/.netrc`); scrubs secret env vars; enforces `RLIMIT_CPU`, `RLIMIT_FSIZE` (64 MiB), and hard process-group `SIGKILL` timeout. |
| **Linux** | `linux-bwrap` ([Bubblewrap](https://github.com/containers/bubblewrap), when on `PATH`) | Unshares network namespace (`--unshare-net`); read-only root bind mount (`--ro-bind / /`) with writable bind strictly limited to the ephemeral temp directory; `--die-with-parent`; scrubs secret env vars; enforces `RLIMIT_CPU`, `RLIMIT_FSIZE`, and hard process-group `SIGKILL` timeout. |
| **Fallback (Linux without `bwrap` / Windows)** | `subprocess-rlimit` | Isolated temp directory, secret env var scrubbing, POSIX `RLIMIT_CPU` + `RLIMIT_FSIZE` (where supported), hard timeout. **Does not isolate network or host filesystem reads** — install `bwrap` or run inside a container/VM. |

Check your active backend at any time:

```sh
python -c "from flash_forge.sandbox import sandbox_backend; print(sandbox_backend())"
```

Verified automatically in [`tests/test_sandbox_security.py`](tests/test_sandbox_security.py) (mechanically asserting blocked outbound sockets, blocked out-of-directory writes, blocked credential reads, and scrubbed secrets).

## Explicit Opt-Out (Not Recommended)

To disable OS sandboxing explicitly (for debugging inside an already-isolated container or VM):

```sh
export CORTHEON_UNSAFE_NO_SANDBOX=1
```

## Agent Modes (`agent`, `dev`) — Experimental

`cortheon agent` and `cortheon dev` execute shell/test commands against a real repository or ephemeral git worktree. While read-only mode locks down `shell=False` command allowlists and `dev` isolates edits in temporary git worktrees, **always run autonomous agent modes inside a container, VM, or disposable repository clone** — never against production credentials or unbacked-up trees.
