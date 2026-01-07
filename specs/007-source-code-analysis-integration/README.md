---
status: planned
created: '2026-01-07'
tags:
  - documentation
  - source-analysis
  - integration
  - lean-spec
  - configuration
priority: high
depends_on:
  - 001-ulysses-sequence-parallelism-analysis
  - 002-experts-parallelism-analysis
  - 003-distributed-checkpoint-analysis
  - 004-group-gemm-kernel-analysis
  - 005-dynamic-batching-strategy-analysis
  - 006-fsdp-fsdp2-backend-analysis
created_at: '2026-01-07T11:48:00.000Z'
updated_at: '2026-01-07T11:48:00.000Z'
---

# Source Code Analysis Documentation Integration

> **Status**: 🔵 Planned · **Priority**: High · **Created**: 2026-01-07 · **Tags**: documentation, source-analysis, integration, lean-spec, configuration

## Overview

This specification describes the integration of VeOmni's comprehensive source code analysis documentation into a unified feature branch. The integration encompasses six core module analysis documents totaling 14,543 lines, along with LeanSpec tooling configuration and AI agent workflow instructions.

### Objectives

1. **Centralize Documentation**: Create a dedicated feature branch (`source_code_analysis`) to house all source code analysis artifacts
2. **Establish Documentation Hub**: Provide unified access to six detailed analysis documents covering VeOmni's distributed training architecture
3. **Configure LeanSpec Tooling**: Set up project-wide specification management infrastructure
4. **Enable AI Agent Workflows**: Integrate agent instructions for automated specification management

### Core Components

The integration includes the following analysis documents and configuration files:

**Analysis Documents** (in `docs/analysis/`):
- `ulysses_sequence_parallelism_analysis.md` (1,212 lines) - Ulysses SP implementation
- `experts_parallelism_analysis.md` (1,899 lines) - MoE expert parallelism
- `distributed_checkpoint_analysis.md` (2,979 lines) - Distributed checkpoint management
- `group_gemm_kernel_analysis.md` (2,821 lines) - GroupGemm optimization kernels
- `dynamic_batching_strategy_analysis.md` (2,443 lines) - Dynamic batching strategies
- `fsdp_fsdp2_backend_analysis.md` (3,189 lines) - FSDP/FSDP2 backend architecture

**Configuration Files**:
- `.lean-spec/config.json` - LeanSpec project configuration
- `.lean-spec/templates/spec-template.md` - Specification template
- `.mcp.json` - MCP server configuration for LeanSpec
- `AGENTS.md` - AI agent workflow instructions
- `CLAUDE.md` - Claude Code project context

### Dependencies

This spec builds upon and consolidates the following individual analysis specifications:
- Spec 001: Ulysses Sequence Parallelism Analysis
- Spec 002: Experts Parallelism Analysis
- Spec 003: Distributed Checkpoint Analysis
- Spec 004: GroupGemm Kernel Analysis
- Spec 005: Dynamic Batching Strategy Analysis
- Spec 006: FSDP/FSDP2 Backend Analysis

## Design

### Architecture

The documentation integration follows a modular architecture with three layers:

```
source_code_analysis/
├── Core Analysis Documents (docs/analysis/)
│   ├── Parallel Computing Layer
│   │   ├── ulysses_sequence_parallelism_analysis.md
│   │   ├── experts_parallelism_analysis.md
│   │   └── fsdp_fsdp2_backend_analysis.md
│   ├── Data & Checkpoint Layer
│   │   ├── distributed_checkpoint_analysis.md
│   │   └── dynamic_batching_strategy_analysis.md
│   └── Performance Optimization Layer
│       └── group_gemm_kernel_analysis.md
├── Specification Management (specs/)
│   ├── 001-ulysses-sequence-parallelism-analysis/
│   ├── 002-experts-parallelism-analysis/
│   ├── 003-distributed-checkpoint-analysis/
│   ├── 004-group-gemm-kernel-analysis/
│   ├── 005-dynamic-batching-strategy-analysis/
│   ├── 006-fsdp-fsdp2-backend-analysis/
│   └── 007-source-code-analysis-integration/ (this spec)
└── Configuration Layer
    ├── .lean-spec/ (LeanSpec configuration)
    ├── .mcp.json (MCP server)
    └── AGENTS.md / CLAUDE.md (AI workflows)
```

### Documentation Relationships

The analysis documents form a dependency graph reflecting VeOmni's architectural layers:

1. **Foundation**: FSDP/FSDP2 backend provides the distributed training foundation
2. **Parallelism**: Ulysses SP and Expert Parallelism build on FSDP for specialized parallelism
3. **Data Management**: Dynamic batching and distributed checkpointing handle data/state
4. **Optimization**: GroupGemm kernels optimize MoE performance

### LeanSpec Configuration

The LeanSpec tooling configuration enables:

- **Flat Directory Structure**: Specs organized with sequential numbering (001-007)
- **Template System**: Standardized spec format with frontmatter metadata
- **AI Agent Integration**: MCP server enables automated spec management via AI agents
- **Dependency Tracking**: Explicit `depends_on` relationships between specifications

### Integration Strategy

The integration uses Git branching to separate analysis documentation from the main development line:

1. **Feature Branch**: `source_code_analysis` branch contains all analysis artifacts
2. **Main Branch Alignment**: `origin/main` stays synchronized with `upstream/main`
3. **Protected History**: Analysis commit (3af1e7e) migrated from main to feature branch
4. **Configuration Co-location**: LeanSpec config files live alongside the code they document

## Plan

### Phase 1: Documentation Verification
- [x] Verify all 6 analysis documents are present in `docs/analysis/`
- [x] Confirm document line counts match expectations (total 14,543 lines)
- [x] Validate document structure and completeness
- [x] Check markdown syntax and internal links

### Phase 2: LeanSpec Setup
- [x] Create `.lean-spec/config.json` with project configuration
- [x] Set up spec template in `.lean-spec/templates/spec-template.md`
- [x] Configure flat directory structure with 3-digit sequence numbers
- [x] Verify template variables (name, status, priority, date)

### Phase 3: Specification Creation
- [x] Create spec 007 directory and README.md
- [x] Write comprehensive spec following template structure
- [x] Document architecture and component relationships
- [x] Ensure spec stays within 2,000-3,500 token optimal range

### Phase 4: Dependency Linking
- [ ] Link spec 007 to spec 001 (Ulysses SP)
- [ ] Link spec 007 to spec 002 (Experts Parallelism)
- [ ] Link spec 007 to spec 003 (Distributed Checkpoint)
- [ ] Link spec 007 to spec 004 (GroupGemm Kernel)
- [ ] Link spec 007 to spec 005 (Dynamic Batching)
- [ ] Link spec 007 to spec 006 (FSDP/FSDP2 Backend)

### Phase 5: Configuration Integration
- [x] Add `.mcp.json` with LeanSpec MCP server configuration
- [x] Create `AGENTS.md` with AI agent workflow instructions
- [x] Create `CLAUDE.md` with project context for Claude Code
- [x] Verify MCP server can connect and manage specs

### Phase 6: Git Branch Management
- [x] Create `source_code_analysis` feature branch from commit 3af1e7e
- [x] Apply stashed configuration files to feature branch
- [x] Reset `main` branch to align with `upstream/main`
- [x] Verify branch history and file locations

### Phase 7: Finalization
- [ ] Commit spec 007 and configuration files
- [ ] Push `source_code_analysis` branch to origin
- [ ] Force-push `main` branch to origin (with --force-with-lease)
- [ ] Verify remote repositories are synchronized

## Test

### LeanSpec Tooling Validation
- [ ] Test MCP server connection: `npx @leanspec/mcp --project /home/scbjtfy/VeOmni`
- [ ] Verify spec listing shows all 7 specs
- [ ] Check spec 007 frontmatter parsing (status, tags, priority)
- [ ] Validate template variable substitution

### Dependency Relationship Validation
- [ ] Confirm spec 007 `depends_on` includes specs 001-006
- [ ] Verify dependency graph is acyclic (no circular dependencies)
- [ ] Check that `lean-spec deps 007-source-code-analysis-integration` shows all 6 dependencies
- [ ] Ensure dependency links are bidirectional (reverse lookup works)

### Documentation Link Integrity
- [ ] Verify all internal document links resolve correctly
- [ ] Check cross-references between specs match actual spec numbers
- [ ] Validate file paths in documentation match repository structure
- [ ] Test that relative links work from spec directory context

### Git Branch Verification
- [ ] Confirm `source_code_analysis` branch contains commit 3af1e7e
- [ ] Verify `main` branch HEAD matches `upstream/main`
- [ ] Check that docs analysis files exist only on feature branch
- [ ] Validate that configuration files are present on feature branch

### Token Count Compliance
- [ ] Measure spec 007 token count (target: 2,000-3,500)
- [ ] If >3,500 tokens, consider splitting into sub-specs
- [ ] Ensure content maintains high signal-to-noise ratio
- [ ] Verify no redundant information duplicates other specs

## Notes

### Design Decisions

**Why Feature Branch?**
Analysis documentation serves a different purpose than code: it provides deep-dive technical reference rather than implementation guidance. Keeping it on a separate branch:
- Keeps main branch focused on code and immediate docs
- Allows analysis updates independent of code changes
- Provides clean integration point for future analysis work

**Why Consolidate 6 Specs into 1?**
The individual specs (001-006) document specific analyses. This meta-spec (007) documents the *integration* of those analyses into the project's workflow and tooling. It adds value by:
- Explaining the documentation architecture
- Configuring the tooling infrastructure
- Establishing the dependency graph
- Providing navigation/discovery for all analyses

**Token Budget Consideration**
Current token count: ~2,800 tokens (within optimal range). Content focuses on integration logic rather than duplicating analysis details, which are already covered in specs 001-006.

### Alternative Approaches Considered

1. **Merge All Analysis Docs into Main**: Rejected because analysis docs are large (14K+ lines) and would clutter the main branch history
2. **Create Individual Branches per Analysis**: Rejected due to branch management complexity and difficulty tracking relationships
3. **No LeanSpec Integration**: Rejected because spec management provides valuable structure and AI agent automation

### Future Enhancements

- **Automated Sync**: Set up CI/CD to sync analysis docs between branches if needed
- **Search Integration**: Add full-text search across all analysis documents
- **Visualization**: Generate dependency graphs and architecture diagrams from specs
- **Metrics Dashboard**: Track documentation coverage and staleness metrics
