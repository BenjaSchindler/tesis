#!/bin/bash
# Quick Launch Template for SMOTE-LLM Experiments on GCP
#
# Usage:
#   ./launch_experiment.sh <vm_name> <seed> <output_prefix> [variant]
#
# Examples:
#   ./launch_experiment.sh my-vm 42 baseline
#   ./launch_experiment.sh my-vm 42 variante_a --prompts-per-cluster 5
#   ./launch_experiment.sh my-vm 42 variante_b --multi-temperature-ensemble
#   ./launch_experiment.sh my-vm 42 variante_c --llm-model gpt-5-mini-2025-08-07 --reasoning-effort minimal

set -e

# Load toolkit
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/gcp_toolkit.sh"

# Parse arguments
VM_NAME="${1:-}"
SEED="${2:-42}"
PREFIX="${3:-experiment}"
shift 3 2>/dev/null || true
EXTRA_ARGS="$*"

# Validate
if [ -z "$VM_NAME" ]; then
    echo "Usage: $0 <vm_name> <seed> <output_prefix> [extra_args...]"
    echo ""
    echo "Examples:"
    echo "  $0 my-vm 42 baseline"
    echo "  $0 my-vm 42 variante_a --prompts-per-cluster 5"
    echo "  $0 my-vm 42 variante_b --multi-temperature-ensemble"
    exit 1
fi

check_api_key || exit 1

# Launch
gcp_quick_launch "$VM_NAME" "$SEED" "$PREFIX" $EXTRA_ARGS
