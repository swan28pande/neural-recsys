#!/usr/bin/env bash
# Run the full experiment pipeline: train all models and compare results.
#
# Usage:
#   bash scripts/run_experiment.sh [config_file]
#
# This trains BPR-MF, NeuMF, and Two-Tower models sequentially,
# evaluates each on the test set, and prints a comparison table.

set -euo pipefail

CONFIG="${1:-configs/default.yaml}"
RESULTS_DIR="results"

echo "============================================================"
echo "  Neural Collaborative Filtering - Experiment Pipeline"
echo "============================================================"
echo ""
echo "Config: ${CONFIG}"
echo ""

# Ensure data is downloaded
echo "Step 1: Checking dataset..."
bash scripts/download_data.sh
echo ""

# Create results directory
mkdir -p "${RESULTS_DIR}"

# Models to train
MODELS=("bpr_mf" "neumf" "two_tower")
MODEL_NAMES=("BPR-MF" "NeuMF" "Two-Tower")

echo "Step 2: Training models..."
echo ""

for i in "${!MODELS[@]}"; do
    model="${MODELS[$i]}"
    name="${MODEL_NAMES[$i]}"

    echo "------------------------------------------------------------"
    echo "  Training ${name} (${model})"
    echo "------------------------------------------------------------"

    python -m src.train \
        --config "${CONFIG}" \
        --model "${model}" \
        2>&1 | tee "${RESULTS_DIR}/${model}_output.log"

    echo ""
    echo "${name} training complete."
    echo ""
done

echo "============================================================"
echo "  All experiments complete!"
echo "  Logs saved to: ${RESULTS_DIR}/"
echo "  Checkpoints saved to: checkpoints/"
echo "============================================================"
echo ""
echo "To view results:"
echo "  cat ${RESULTS_DIR}/*_output.log | grep -A5 'Test Results'"
