---
name: cortheon-answer
description: UNSHIPPED legacy example. Do not install or run.
argument-hint: "<task> [:: proposed action]"
agent: build
subtask: false
---

This command is not part of the shipped Cortheon product.

Usage: `/cortheon-answer <task> or /cortheon-answer <task> :: <proposed action>`

Rules:
- Use this before coding when the right library, API, architecture, or current best option matters.
- If the verdict is needs_evidence, do not continue into production code.
- Run the command below with the exact user arguments substituted for `$ARGUMENTS`.
- Reply with the Cortheon output and obey its final Agent Instruction.
- Do not invent sources, APIs, or evidence that are not in the output.

```bash
echo "Legacy Cortheon slash commands are not shipped or supported" >&2
exit 2
```
