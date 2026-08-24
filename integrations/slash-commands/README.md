# Legacy slash-command examples

These markdown templates are quarantined repository history. Do not install or
run them. They call `cortheon.slash`, a legacy persistent-report engine that is
not included in the Cortheon wheel and does not implement the shipped
memory-only lifecycle.

Use a native Pi, OpenCode, or Codex integration from the main
[README](../../README.md). For another host, run:

```bash
cortheon configure --host generic
```

Copy the resulting MCP entry into the host. The supported cooperative flow is
`cortheon_start`, focused host tool use, `cortheon_observe`, and
`cortheon_complete` or `cortheon_abandon`.
