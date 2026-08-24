---
name: cortheon-api
description: UNSHIPPED legacy example. Do not install or run.
argument-hint: "<package> :: <symbol>"
agent: build
subtask: false
---

This command is not part of the shipped Cortheon product.

Usage: `/cortheon-api <package> :: <symbol>`

Rules:
- Use this before writing production code against a package-specific class, method, or function.
- No matches means do not use that symbol unless another source-derived proof is found.
- Run the command below with the exact user arguments substituted for `$ARGUMENTS`.
- Reply with the Cortheon output and obey its final Agent Instruction.
- Do not invent sources, APIs, or evidence that are not in the output.

```bash
echo "Legacy Cortheon slash commands are not shipped or supported" >&2
exit 2
```
