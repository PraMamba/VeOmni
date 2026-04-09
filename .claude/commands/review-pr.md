---
name: review-pr
description: Review a PR using VeOmni-specific change type detection and review templates. Invoke with /review-pr <PR_NUMBER>.
---

# Review PR

Comprehensive PR review using VeOmni-specific change type detection and
review checklists.

## Usage

```
/review-pr <PR_NUMBER>
```

## Workflow

### Step 1: Fetch PR Information

```bash
gh pr view <PR_NUMBER> --json title,body,files,additions,deletions
gh pr diff <PR_NUMBER>
```

### Step 2: Detect Change Types

Analyze changed files against the change type detection table in
`.claude/data/review-pr-change-types.md`.

Assign a review level:
- **CRITICAL** (Opus): Core distributed, FSDP, MoE EP, checkpoint
- **HIGH** (Opus): Communication patterns, tensor parallel, SP
- **MEDIUM** (Sonnet): Tensor ops, config, trainer, data pipeline
- **LOW** (Haiku): Tests, docs, config-only

### Step 3: Select Review Templates

Based on detected change types, select applicable review templates from
`.claude/data/review-pr-templates.md`.

### Step 4: Execute Review

For each applicable template:
1. Read the relevant changed files
2. Walk through the checklist items
3. Flag any issues found
4. Note positives (good patterns)

### Step 5: Generate Report

```markdown
## PR Review: #{PR_NUMBER}

### Summary
- **Change types detected**: [list]
- **Review level**: CRITICAL/HIGH/MEDIUM/LOW
- **Overall assessment**: APPROVE / REQUEST_CHANGES / COMMENT

### Findings

#### Critical Issues
- ...

#### Suggestions
- ...

#### Positives
- ...
```
