---
name: gen-commit-msg
description: Generate a commit message based on staged changes following VeOmni PR title conventions. Invoke with /gen-commit-msg.
---

# Generate Commit Message

Generate a concise, well-formatted commit message based on the current staged changes.

## Workflow

### Step 1: Analyze Changes

```bash
# Get staged changes
git diff --cached --stat
git diff --cached
```

### Step 2: Determine Module and Type

**Module** (from changed file paths):
- `veomni/arguments/` → `config`
- `veomni/checkpoint/` → `ckpt`
- `veomni/data/` → `data`
- `veomni/distributed/` → `parallel` or `dist`
- `veomni/models/` → `model`
- `veomni/ops/` → `ops`
- `veomni/optim/` or `veomni/schedulers/` → `optim`
- `veomni/trainer/` → `task`
- `veomni/utils/` → `misc`
- `veomni/patchgen/` → `model`
- `tests/` → `ci`
- `docs/` → `docs`
- `.github/` → `ci`
- `configs/` → `config`
- Multiple modules → comma-separated: `[model, data]`

**Type**:
- `feat`: New functionality
- `fix`: Bug fix
- `refactor`: Code restructuring (no behavior change)
- `chore`: Maintenance (deps, CI, configs)
- `test`: Test-only changes

### Step 3: Format

```
[{modules}] {type}: {concise description}

{optional body with details}
```

**Examples**:
- `[model] feat: add Qwen3.5-MoE support with EP parallel plan`
- `[parallel, ops] fix: correct async Ulysses all-to-all tensor shape`
- `[data] refactor: simplify dynamic batching strategy selection`
- `[ci] chore: add transformers v5 compatibility test`
