#!/bin/bash
# Hook script to remind updating expert agents when related code changes
# Called by Claude Code PostToolUse hook

# Check if jq is available
if ! command -v jq &> /dev/null; then
    exit 0
fi

# Read JSON input from stdin
INPUT=$(cat)

# Extract file path from tool input
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

if [ -z "$FILE_PATH" ]; then
    exit 0
fi

# Define mappings: code path pattern -> expert agent file
check_expert_update() {
    local file="$1"
    local reminder_file=""
    local reminder_desc=""

    # FSDP / FSDP2 related
    if [[ "$file" == *"veomni/distributed/fsdp"* ]] || \
       [[ "$file" == *"veomni/distributed/fsdp2"* ]] || \
       [[ "$file" == *"veomni/distributed/torch_parallelize"* ]] || \
       [[ "$file" == *"veomni/distributed/parallel_state"* ]]; then
        reminder_file="fsdp-expert.md"
        reminder_desc="FSDP/Parallel State"
    fi

    # Sequence Parallel related
    if [[ "$file" == *"veomni/distributed/sequence_parallel"* ]]; then
        reminder_file="sequence-parallel-expert.md"
        reminder_desc="Sequence Parallel (Ulysses)"
    fi

    # MoE / Expert Parallel related
    if [[ "$file" == *"veomni/distributed/moe"* ]] || \
       [[ "$file" == *"veomni/ops/fused_moe"* ]] || \
       [[ "$file" == *"veomni/ops/group_gemm"* ]] || \
       [[ "$file" == *"parallel_plan.py"* ]]; then
        reminder_file="moe-expert.md"
        reminder_desc="MoE/Expert Parallel"
    fi

    # Model loading / patching related
    if [[ "$file" == *"veomni/models/"* ]] || \
       [[ "$file" == *"veomni/patchgen/"* ]]; then
        reminder_file="model-expert.md"
        reminder_desc="Model Loading/Patching"
    fi

    # Trainer / Callback related
    if [[ "$file" == *"veomni/trainer/"* ]]; then
        reminder_file="trainer-expert.md"
        reminder_desc="Trainer/Callbacks"
    fi

    # Data pipeline related
    if [[ "$file" == *"veomni/data/"* ]]; then
        reminder_file="data-pipeline-expert.md"
        reminder_desc="Data Pipeline"
    fi

    # Output reminder if matched
    if [ -n "$reminder_file" ]; then
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "Expert Update Reminder"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "Modified: $file"
        echo "Consider updating: .claude/agents/$reminder_file ($reminder_desc)"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
    fi
}

check_expert_update "$FILE_PATH"
