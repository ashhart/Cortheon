---
name: cortheon-decide
description: UNSHIPPED legacy example. Do not install or run.
argument-hint: "<task> :: <proposed action>"
agent: build
subtask: false
---

This command is not part of the shipped Cortheon product.

Usage: `/cortheon-decide <task> :: <proposed action>`

Rules:
- Use this when the model is about to act and needs a permission gate.
- The separator :: is required so the substrate can distinguish task from action.
- Run the command below with the exact user arguments substituted for `$ARGUMENTS`.
- Reply with the Cortheon output and obey its final Agent Instruction.
- Do not invent sources, APIs, or evidence that are not in the output.

```bash
echo "Legacy Cortheon slash commands are not shipped or supported" >&2
exit 2
```
