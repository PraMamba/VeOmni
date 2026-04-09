---
name: create-pr
description: Rebase from the latest origin/main, squash commits, and create a PR on GitHub with proper VeOmni title format. Invoke with /create-pr.
---

# Create Pull Request

Rebase from the latest `origin/main`, squash commits, and create a PR on GitHub.

## Usage

```
/create-pr [--draft] [--base <branch>]
```

## Workflow

### Step 1: Verify Prerequisites

```bash
git branch --show-current
git status --short
gh --version
```

- Must NOT be on main/master
- Must have no uncommitted changes

### Step 2: Check for Existing PR

```bash
gh pr view --json number,title,url 2>/dev/null || echo "No existing PR"
```

### Step 3: Fetch and Rebase

```bash
git fetch origin main
git log --oneline HEAD ^origin/main
git rebase origin/main
```

If rebase fails, abort and let user handle manually.

### Step 4: Squash Commits

```bash
MERGE_BASE=$(git merge-base HEAD origin/main)
COMMIT_COUNT=$(git rev-list --count ${MERGE_BASE}..HEAD)
```

If >1 commit, squash into a single commit with combined message.

### Step 5: Generate PR Content

**Title Format**: `[{modules}] {type}: {description}`

- Modules: misc, ci, config, docs, data, dist, omni, logging, model, optim, ckpt, release, task, perf, ops, parallel
- Type: feat, fix, refactor, chore, test
- Breaking: prepend `[BREAKING]`

**Body Template**:
```markdown
### What does this PR do?

> {concise overview}

### Checklist Before Starting

- [ ] Search for similar PRs
- [ ] PR title follows `[{modules}] {type}: {description}` format

### Test

> {validation results}

### API and Usage Example

> {API changes if applicable}

### Design & Code Changes

> {high-level design and change list}

### Checklist Before Submitting

- [ ] Applied pre-commit checks
- [ ] Added/updated documentation
- [ ] Added tests to CI workflow
```

### Step 6: Create PR

```bash
gh pr create --title "<title>" --body "<body>"
```
