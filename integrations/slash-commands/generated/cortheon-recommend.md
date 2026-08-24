---
name: cortheon-recommend
description: UNSHIPPED legacy example. Do not install or run.
argument-hint: "<task>"
agent: build
subtask: false
---

This command is not part of the shipped Cortheon product.

Usage: `/cortheon-recommend <task>`

Rules:
- Use this for narrow package/library selection.
- For open-ended research or scientific claims, prefer /cortheon-answer or /cortheon-research.
- Run the command below with the exact user arguments substituted for `$ARGUMENTS`.
- Reply with the Cortheon output and obey its final Agent Instruction.
- Do not invent sources, APIs, or evidence that are not in the output.

```bash
echo "Legacy Cortheon slash commands are not shipped or supported" >&2
exit 2
```
