#!/usr/bin/env bash
set -euo pipefail

# Common official-MACE LoRA launcher for conservative total-energy/force fitting.
# Adaptation lanes differ only in input files, epoch count, and run name.

: "${FOUNDATION_MODEL:?Set FOUNDATION_MODEL}"
: "${TRAIN_FILE:?Set TRAIN_FILE}"
: "${VALID_FILE:?Set VALID_FILE}"
: "${RUN_NAME:?Set RUN_NAME}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR}"

MODEL_KIND="${MODEL_KIND:-MACE}"
DEFAULT_DTYPE="${DEFAULT_DTYPE:-float64}"
TEST_FILE="${TEST_FILE:-${VALID_FILE}}"
ENERGY_KEY="${ENERGY_KEY:-REF_energy}"
FORCES_KEY="${FORCES_KEY:-REF_forces}"
SEED="${SEED:-17}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-2}"
VALID_BATCH_SIZE="${VALID_BATCH_SIZE:-${BATCH_SIZE}}"
LEARNING_RATE="${LEARNING_RATE:-5.0e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1.0e-8}"
DEVICE="${DEVICE:-cuda}"
PATIENCE="${PATIENCE:-1000000}"
LES_ARGUMENTS="${LES_ARGUMENTS:-}"

command -v mace_run_train >/dev/null 2>&1 || {
  echo "mace_run_train was not found; activate the pinned MACE environment." >&2
  exit 1
}

for required_file in "${FOUNDATION_MODEL}" "${TRAIN_FILE}" "${VALID_FILE}" "${TEST_FILE}"; do
  [[ -f "${required_file}" ]] || {
    echo "Missing required file: ${required_file}" >&2
    exit 1
  }
done

if [[ "${MODEL_KIND}" == "MACELES" && -n "${LES_ARGUMENTS}" && ! -f "${LES_ARGUMENTS}" ]]; then
  echo "LES_ARGUMENTS was supplied but does not exist: ${LES_ARGUMENTS}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

args=(
  "--name=${RUN_NAME}"
  "--work_dir=${OUTPUT_DIR}"
  "--model=${MODEL_KIND}"
  "--foundation_model=${FOUNDATION_MODEL}"
  "--foundation_model_elements=True"
  "--train_file=${TRAIN_FILE}"
  "--valid_file=${VALID_FILE}"
  "--test_file=${TEST_FILE}"
  "--energy_key=${ENERGY_KEY}"
  "--forces_key=${FORCES_KEY}"
  "--E0s=foundation"
  "--lora=True"
  "--lora_rank=32"
  "--lora_alpha=32.0"
  "--energy_weight=1.0"
  "--forces_weight=100.0"
  "--lr=${LEARNING_RATE}"
  "--weight_decay=${WEIGHT_DECAY}"
  "--batch_size=${BATCH_SIZE}"
  "--valid_batch_size=${VALID_BATCH_SIZE}"
  "--max_num_epochs=${EPOCHS}"
  "--patience=${PATIENCE}"
  "--ema"
  "--ema_decay=0.995"
  "--amsgrad"
  "--clip_grad=10.0"
  "--restart_latest"
  "--save_all_checkpoints"
  "--save_cpu"
  "--eval_interval=1"
  "--default_dtype=${DEFAULT_DTYPE}"
  "--device=${DEVICE}"
  "--seed=${SEED}"
)

# The released converted MACELES-OFF checkpoint uses the published default LES
# reconstruction. An LES YAML is accepted only when deliberately supplied.
if [[ -n "${LES_ARGUMENTS}" ]]; then
  args+=("--les_arguments=${LES_ARGUMENTS}")
fi

{
  echo "run_name=${RUN_NAME}"
  echo "model_kind=${MODEL_KIND}"
  echo "foundation_model=${FOUNDATION_MODEL}"
  echo "foundation_model_elements=True"
  echo "train_file=${TRAIN_FILE}"
  echo "valid_file=${VALID_FILE}"
  echo "test_file=${TEST_FILE}"
  echo "epochs=${EPOCHS}"
  echo "batch_size=${BATCH_SIZE}"
  echo "valid_batch_size=${VALID_BATCH_SIZE}"
  echo "seed=${SEED}"
  echo "default_dtype=${DEFAULT_DTYPE}"
  echo "learning_rate=${LEARNING_RATE}"
  echo "weight_decay=${WEIGHT_DECAY}"
  echo "les_arguments=${LES_ARGUMENTS:-<published-default>}"
  printf 'command='
  printf '%q ' mace_run_train "${args[@]}"
  printf '\n'
} | tee "${OUTPUT_DIR}/launch_manifest.txt"

mace_run_train "${args[@]}"
