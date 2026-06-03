#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_SH="${CONDA_SH:-/media/home/smx/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-sharp}"
if [[ -x "${CONDA_SH}" ]]; then
  eval "$("${CONDA_SH}" shell.bash hook)"
  conda activate "${CONDA_ENV}"
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"

CHECKPOINT="${CHECKPOINT:-${1:-}}"
if [[ -z "${CHECKPOINT}" || ! -f "${CHECKPOINT}" ]]; then
  echo "ERROR: pass a checkpoint as arg1 or set CHECKPOINT=/path/to/step_XXXXXXX.pt" >&2
  exit 1
fi

export OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs/validation}"
export RUN_NAME="${RUN_NAME:-unisharp_validation_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${OUT_ROOT}/${RUN_NAME}"
mkdir -p "${RUN_DIR}"

export GPU_IDS="${GPU_IDS:-0}"
export VALIDATION_JOBS_PER_GPU="${VALIDATION_JOBS_PER_GPU:-1}"
export VALIDATION_BATCH_SIZE="${VALIDATION_BATCH_SIZE:-1}"
export VALIDATION_FAST_METRICS="${VALIDATION_FAST_METRICS:-1}"
export VALIDATION_MAX_GROUPS="${VALIDATION_MAX_GROUPS:-0}"
export SEED="${SEED:-42}"
export MAX_INDEX_GAP="${MAX_INDEX_GAP:-10}"
export PAIR_MAX_TRANSLATION_M="${PAIR_MAX_TRANSLATION_M:-0.5}"
export PAIR_MIN_OVERLAP="${PAIR_MIN_OVERLAP:-0.6}"
export PANO_POSE_FLIP_CONVENTION="${PANO_POSE_FLIP_CONVENTION:-flip_yz_negate_rel_z}"

DEFAULT_VALIDATION_MANIFEST_DIR="${REPO_ROOT}/validation_manifests"
if [[ -d "${REPO_ROOT}/../validation_manifests" ]]; then
  DEFAULT_VALIDATION_MANIFEST_DIR="${REPO_ROOT}/../validation_manifests"
fi
export VALIDATION_MANIFEST_DIR="${VALIDATION_MANIFEST_DIR:-${DEFAULT_VALIDATION_MANIFEST_DIR}}"
export VALIDATION_PSEUDO_DEPTH_ROOT="${VALIDATION_PSEUDO_DEPTH_ROOT:-/media/team_data/ML4_team/datasets/sharp/validation_unik3d_pseudo_depth}"
export RE10K_PSEUDO_DEPTH_ROOT="${RE10K_PSEUDO_DEPTH_ROOT:-/media/team_data/ML4_team/datasets/nopose/re10k_unik3d_pseudo_depth/test}"

export DATA_ROOT_RE10K="${DATA_ROOT_RE10K:-/media/team_data/ML4_team/datasets/nopose/re10k}"
export DATA_ROOT_DL3DV="${DATA_ROOT_DL3DV:-/media/team_data/ML4_team/datasets/sharp/DL3DV-ALL-960P}"
export DATA_ROOT_HM3D="${DATA_ROOT_HM3D:-/media/team_data/ML4_team/datasets/panogs}"
export DATA_ROOT_REPLICA="${DATA_ROOT_REPLICA:-/media/team_data/ML4_team/datasets/sharp/replica}"
export DATA_ROOT_SIM="${DATA_ROOT_SIM:-/media/team_data/ML4_team/datasets/smx_sim}"
export SIM_POSE_ROOT="${SIM_POSE_ROOT:-/media/team_data/ML4_team/datasets/smx_sim/30cm}"
DEFAULT_DATASET_MANIFEST_DIR="${REPO_ROOT}/dataset_manifests"
if [[ -d "${REPO_ROOT}/../dataset_manifests" ]]; then
  DEFAULT_DATASET_MANIFEST_DIR="${REPO_ROOT}/../dataset_manifests"
fi
export WILD_ROOTS_FILE="${WILD_ROOTS_FILE:-${DEFAULT_DATASET_MANIFEST_DIR}/wildrgbd_roots.txt}"
export DATA_ROOT_SCANNETPP="${DATA_ROOT_SCANNETPP:-/media/home/songmeixi_insta360.com/sharpgs/scannetpp}"
export DATA_ROOT_SCANETPP_FISHEYE="${DATA_ROOT_SCANETPP_FISHEYE:-/media/team_data/ML4_team/datasets/scan}"
export DATA_ROOT_TAT="${DATA_ROOT_TAT:-/media/team_data/ML4_team/datasets/TAT/tanks_and_temples}"

DATASETS_CSV="${DATASETS:-re10k,dl3dv,hm3d,sim,wildrgbd}"
IFS=',' read -r -a DATASET_ARR <<< "${DATASETS_CSV}"
IFS=',' read -r -a GPU_ID_ARR <<< "${GPU_IDS}"
if [[ "${VALIDATION_JOBS_PER_GPU}" -lt 1 ]]; then
  echo "ERROR: VALIDATION_JOBS_PER_GPU must be >= 1" >&2
  exit 1
fi

data_root_for_dataset() {
  case "$1" in
    re10k) echo "${DATA_ROOT_RE10K}" ;;
    dl3dv) echo "${DATA_ROOT_DL3DV}" ;;
    hm3d)
      if [[ -d "${DATA_ROOT_HM3D}/test" ]]; then
        echo "${DATA_ROOT_HM3D}/test"
      else
        echo "${DATA_ROOT_HM3D}"
      fi
      ;;
    replica) echo "${DATA_ROOT_REPLICA}" ;;
    sim) echo "${DATA_ROOT_SIM}" ;;
    wildrgbd) echo "${WILD_ROOTS_FILE}" ;;
    scannetpp) echo "${DATA_ROOT_SCANNETPP}" ;;
    scanetpp_fisheye) echo "${DATA_ROOT_SCANETPP_FISHEYE}" ;;
    smx_sim_fisheye) echo "${DATA_ROOT_SIM}" ;;
    tat) echo "${DATA_ROOT_TAT}" ;;
    *) echo "Unknown dataset: $1" >&2; return 1 ;;
  esac
}

extra_args_for_dataset() {
  case "$1" in
    re10k) echo "--re10k-pseudo-depth-root ${RE10K_PSEUDO_DEPTH_ROOT}" ;;
    sim) echo "--sim-pose-root ${SIM_POSE_ROOT}" ;;
    *) echo "" ;;
  esac
}

run_dataset() {
  local gpu_id="$1"
  local dataset="$2"
  local data_root
  local out_dir
  local manifest

  data_root="$(data_root_for_dataset "${dataset}")"
  out_dir="${RUN_DIR}/${dataset}"
  manifest="${VALIDATION_MANIFEST_DIR}/${dataset}.txt"

  local cmd=(
    python -m unisharp.validation.run_validation
    --checkpoint "${CHECKPOINT}"
    --dataset "${dataset}"
    --data-root "${data_root}"
    --device "cuda:0"
    --out-dir "${out_dir}"
    --validation-batch-size "${VALIDATION_BATCH_SIZE}"
    --validation-pseudo-depth-root "${VALIDATION_PSEUDO_DEPTH_ROOT}"
    --max-index-gap "${MAX_INDEX_GAP}"
    --pair-max-translation-m "${PAIR_MAX_TRANSLATION_M}"
    --pair-min-overlap "${PAIR_MIN_OVERLAP}"
    --seed "${SEED}"
  )
  if [[ -f "${manifest}" ]]; then
    cmd+=(--manifest-file "${manifest}")
  fi
  if [[ "${VALIDATION_MAX_GROUPS}" != "0" ]]; then
    cmd+=(--manifest-max-groups "${VALIDATION_MAX_GROUPS}")
  fi
  if [[ "${VALIDATION_FAST_METRICS}" == "1" ]]; then
    cmd+=(--fast-metrics)
  fi
  read -r -a extra_args <<< "$(extra_args_for_dataset "${dataset}")"
  if [[ "${#extra_args[@]}" -gt 0 && -n "${extra_args[0]:-}" ]]; then
    cmd+=("${extra_args[@]}")
  fi

  echo "Validating ${dataset} on GPU ${gpu_id}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" PANO_POSE_FLIP_CONVENTION="${PANO_POSE_FLIP_CONVENTION}" "${cmd[@]}"
}

worker() {
  local worker_id="$1"
  local gpu_index=$(( worker_id % ${#GPU_ID_ARR[@]} ))
  local gpu_id="${GPU_ID_ARR[${gpu_index}]}"
  local total_workers=$(( ${#GPU_ID_ARR[@]} * VALIDATION_JOBS_PER_GPU ))
  local idx
  for idx in "${!DATASET_ARR[@]}"; do
    if (( idx % total_workers == worker_id )); then
      run_dataset "${gpu_id}" "${DATASET_ARR[${idx}]}"
    fi
  done
}

echo "UniSharp validation"
echo "  CHECKPOINT=${CHECKPOINT}"
echo "  RUN_DIR=${RUN_DIR}"
echo "  DATASETS=${DATASETS_CSV}"
echo "  GPU_IDS=${GPU_IDS}"

TOTAL_WORKERS=$(( ${#GPU_ID_ARR[@]} * VALIDATION_JOBS_PER_GPU ))
PIDS=()
for worker_id in $(seq 0 $((TOTAL_WORKERS - 1))); do
  worker "${worker_id}" &
  PIDS+=("$!")
done

STATUS=0
for pid in "${PIDS[@]}"; do
  wait "${pid}" || STATUS=1
done

if [[ "${STATUS}" -ne 0 ]]; then
  echo "One or more validation workers failed." >&2
  exit "${STATUS}"
fi

echo "Validation finished."
echo "Outputs: ${RUN_DIR}"
