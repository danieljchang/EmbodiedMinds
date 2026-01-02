#!/usr/bin/env bash

# Run EmbodiedBench-style evaluation for multiple VLM-backed agents.
# For each model, this script:
#   1. Loads the corresponding trained AgentModel checkpoint
#   2. Runs evaluate_embodiedbench.py
#   3. Produces a separate JSON + CSV report and a per-model log file
#
# Assumptions:
# - You have already trained one agent per VLM configuration and saved
#   checkpoints in the "checkpoints/" directory using the names below.
# - Each checkpoint was evaluated/trained with the corresponding VLM adapter.
# - Data root points to the EB-Man trajectory dataset.
#
# This matches the models you listed from the EmbodiedBench paper:
#   - Llama 3.2-11B-Vision-Ins
#   - InternVL 2.5-8B
#   - InternVL 3-8B
#   - Qwen 2-VL-7B-Ins
#   - Qwen 2.5-VL-7B-Ins
#   - Ovis 2-16B
#   - gemma-3-12B-it

set -euo pipefail

DATA_ROOT="./data/EB-Man_trajectory_dataset"
OUTPUT_DIR="./logs"
S3_BUCKET="11777-h1"
MAX_EPISODES=100
SUCCESS_THRESHOLD=0.8

mkdir -p "${OUTPUT_DIR}"

# Map human-readable model names to checkpoint stems.
# Each checkpoint file is expected at: checkpoints/<CKPT_STEM>.pt
declare -A CKPTS
CKPTS["llama-3.2-11b-vision-ins"]="agent_best_llama32_11b_vision_ins"
CKPTS["internvl-2.5-8b"]="agent_best_internvl25_8b"
CKPTS["internvl-3-8b"]="agent_best_internvl3_8b"
CKPTS["qwen2-vl-7b-ins"]="agent_best_qwen2_vl_7b_ins"
CKPTS["qwen2.5-vl-7b-ins"]="agent_best_qwen25_vl_7b_ins"
CKPTS["ovis2-16b"]="agent_best_ovis2_16b"
CKPTS["gemma-3-12b-it"]="agent_best_gemma3_12b_it"

MODELS=(
  "llama-3.2-11b-vision-ins"
  "internvl-2.5-8b"
  "internvl-3-8b"
  "qwen2-vl-7b-ins"
  "qwen2.5-vl-7b-ins"
  "ovis2-16b"
  "gemma-3-12b-it"
)

echo "==============================================="
echo " Running EmbodiedBench-style evaluation for VLM-backed agents"
echo " Data root:      ${DATA_ROOT}"
echo " Output dir:     ${OUTPUT_DIR}"
echo " S3 bucket:      ${S3_BUCKET}"
echo " Max episodes:   ${MAX_EPISODES}"
echo " Success thresh: ${SUCCESS_THRESHOLD}"
echo "==============================================="
echo

for MODEL_NAME in "${MODELS[@]}"; do
  CKPT_STEM="${CKPTS[${MODEL_NAME}]}"
  CKPT_PATH="checkpoints/${CKPT_STEM}.pt"

  echo "-------------------------------------------------"
  echo "Model: ${MODEL_NAME}"
  echo "Checkpoint: ${CKPT_PATH}"

  if [[ ! -f "${CKPT_PATH}" ]]; then
    echo "  WARNING: Checkpoint not found, skipping."
    echo
    continue
  fi

  LOG_FILE="${OUTPUT_DIR}/evaluation_${CKPT_STEM}.log"

  echo "  Writing detailed log to: ${LOG_FILE}"
  echo "  Running evaluation..."

  python3 evaluate_embodiedbench.py \
    --checkpoint "${CKPT_PATH}" \
    --data-root "${DATA_ROOT}" \
    --max-episodes "${MAX_EPISODES}" \
    --success-threshold "${SUCCESS_THRESHOLD}" \
    --output-dir "${OUTPUT_DIR}" \
    --s3-bucket "${S3_BUCKET}" \
    > "${LOG_FILE}" 2>&1

  echo "  Done. See:"
  echo "    - ${LOG_FILE}"
  echo "    - ${OUTPUT_DIR}/embodiedbench_evaluation_${CKPT_STEM}.json"
  echo "    - ${OUTPUT_DIR}/embodiedbench_summary_${CKPT_STEM}.csv"
  echo
done

echo "==============================================="
echo "All requested evaluations have been processed."
echo "==============================================="



