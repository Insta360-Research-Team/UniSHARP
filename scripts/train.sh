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

export OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs}"
export RUN_NAME="${RUN_NAME:-unisharp_gt_override_$(date +%Y%m%d_%H%M%S)}"

export SEED="${SEED:-260602}"

export STEPS="${STEPS:-1000000}"
export WARMUP="${WARMUP:-75000}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export NUM_WORKERS="${NUM_WORKERS:-1}"
export GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
export MASTER_PORT="${MASTER_PORT:-29531}"
export DEVICE="${DEVICE:-cuda}"
export DDP_TIMEOUT_HOURS="${DDP_TIMEOUT_HOURS:-8}"

export LR0="${LR0:-1.2e-4}"
export LR1="${LR1:-1.6e-5}"
export UNIK3D_DECODER_LR0="${UNIK3D_DECODER_LR0:-2.5e-5}"
export UNIK3D_DECODER_LR1="${UNIK3D_DECODER_LR1:-2.5e-6}"
export UNIK3D_ENCODER_LR0="${UNIK3D_ENCODER_LR0:-1.5e-6}"
export UNIK3D_ENCODER_LR1="${UNIK3D_ENCODER_LR1:-1.5e-7}"
export GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
export MAX_STEP_GRAD_NORM="${MAX_STEP_GRAD_NORM:-100000.0}"

export INITIALIZER_STRIDE="${INITIALIZER_STRIDE:-1}"
export INITIALIZER_SCALE_FACTOR="${INITIALIZER_SCALE_FACTOR:-1.5}"
export DELTA_RHO_LIMIT="${DELTA_RHO_LIMIT:-2.0}"

export MAX_INDEX_GAP="${MAX_INDEX_GAP:-10}"
export MAX_DEPTH_M="${MAX_DEPTH_M:-100.0}"
export PINHOLE_TRAIN_SIZE="${PINHOLE_TRAIN_SIZE:-0}"
export TRAIN_RESIZE_MULTIPLE="${TRAIN_RESIZE_MULTIPLE:-256}"
export SIM_MAX_LONG_EDGE="${SIM_MAX_LONG_EDGE:-512}"
export RE10K_PSEUDO_FAR_DEPTH_INVALID_M="${RE10K_PSEUDO_FAR_DEPTH_INVALID_M:-30.0}"
export SIM_FAR_DEPTH_INVALID_M="${SIM_FAR_DEPTH_INVALID_M:-30.0}"
export SIM_FAR_DEPTH_INVALID_MAX_FRAC="${SIM_FAR_DEPTH_INVALID_MAX_FRAC:-1.0}"
export SCANETPP_FISHEYE_FAR_DEPTH_INVALID_M="${SCANETPP_FISHEYE_FAR_DEPTH_INVALID_M:-30.0}"

export LAMBDA_COLOR="${LAMBDA_COLOR:-1.0}"
export LAMBDA_ALPHA="${LAMBDA_ALPHA:-1.5}"
export LAMBDA_PERCEP="${LAMBDA_PERCEP:-1.0}"
export LAMBDA_DEPTH="${LAMBDA_DEPTH:-0.5}"
export LAMBDA_TV="${LAMBDA_TV:-1.0}"
export LAMBDA_GRAD="${LAMBDA_GRAD:-1.0}"
export LAMBDA_GRAD_IMG="${LAMBDA_GRAD_IMG:-0.2}"
export LAMBDA_EDGE_RGB="${LAMBDA_EDGE_RGB:-0.0}"
export LAMBDA_DELTA="${LAMBDA_DELTA:-1.0}"
export LAMBDA_DELTA_RHO="${LAMBDA_DELTA_RHO:-0.01}"
export LAMBDA_SPLAT="${LAMBDA_SPLAT:-1.0}"
export LAMBDA_EDGE_SPLAT="${LAMBDA_EDGE_SPLAT:-0.0}"
export LAMBDA_GRID="${LAMBDA_GRID:-0.05}"
export LAMBDA_AUX_RAY="${LAMBDA_AUX_RAY:-3.0}"
export LAMBDA_AUX_DEPTH_SCALE="${LAMBDA_AUX_DEPTH_SCALE:-3.0}"
export LAMBDA_AUX_DEPTH2_SCALE="${LAMBDA_AUX_DEPTH2_SCALE:-1.0}"

export SAVE_EVERY="${SAVE_EVERY:-5000}"
export VIS_EVERY="${VIS_EVERY:-500}"
export LOG_EVERY="${LOG_EVERY:-50}"


export DATASET_WEIGHT_RE10K="${DATASET_WEIGHT_RE10K:-1.0}"
export DATASET_WEIGHT_HM3D="${DATASET_WEIGHT_HM3D:-1.0}"
export DATASET_WEIGHT_SIM="${DATASET_WEIGHT_SIM:-1.0}"
export DATASET_WEIGHT_WILDRGBD="${DATASET_WEIGHT_WILDRGBD:-1.0}"
export DATASET_WEIGHT_DL3DV="${DATASET_WEIGHT_DL3DV:-1.0}"
export DATASET_WEIGHT_SCANETPP="${DATASET_WEIGHT_SCANETPP:-0.0}"

export DATA_ROOT_RE10K="${DATA_ROOT_RE10K:-/media/team_data/ML4_team/datasets/nopose/re10k}"
export RE10K_PSEUDO_DEPTH_ROOT="${RE10K_PSEUDO_DEPTH_ROOT:-/media/team_data/ML4_team/datasets/nopose/re10k_unik3d_pseudo_depth}"
export DATA_ROOT_HM3D="${DATA_ROOT_HM3D:-/media/team_data/ML4_team/datasets/panogs}"
export DATA_ROOT_SIM="${DATA_ROOT_SIM:-/media/team_data/ML4_team/datasets/smx_sim}"
export SIM_POSE_ROOT="${SIM_POSE_ROOT:-/media/team_data/ML4_team/datasets/smx_sim/30cm}"
export DATA_ROOT_DL3DV="${DATA_ROOT_DL3DV:-/media/team_data/ML4_team/datasets/sharp/DL3DV-ALL-960P}"
export DATA_ROOT_DL3DV_DEPTH="${DATA_ROOT_DL3DV_DEPTH:-/media/team_data/ML4_team/datasets/sharp/DL3DV-ALL-960P_da3_outputs}"
export DATA_ROOT_SCANETPP="${DATA_ROOT_SCANETPP:-/media/team_data/ML4_team/datasets/scan}"

DEFAULT_DATASET_MANIFEST_DIR="${REPO_ROOT}/dataset_manifests"
if [[ -d "${REPO_ROOT}/../dataset_manifests" ]]; then
  DEFAULT_DATASET_MANIFEST_DIR="${REPO_ROOT}/../dataset_manifests"
fi
export DATASET_MANIFEST_DIR="${DATASET_MANIFEST_DIR:-${DEFAULT_DATASET_MANIFEST_DIR}}"
if [[ ! -f "${DATASET_MANIFEST_DIR}/sim_train_scenes.txt" && -f "${REPO_ROOT}/../dataset_manifests/sim_train_scenes.txt" ]]; then
  export DATASET_MANIFEST_DIR="${REPO_ROOT}/../dataset_manifests"
fi
export WILD_ROOTS_FILE="${WILD_ROOTS_FILE:-${DATASET_MANIFEST_DIR}/wildrgbd_roots.txt}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export NCCL_NET="${NCCL_NET:-Socket}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

IFS=',' read -r -a GPU_ID_ARR <<< "${GPU_IDS}"
if [[ "${#GPU_ID_ARR[@]}" -gt 1 ]]; then
  LAUNCH_CMD=(torchrun --nproc_per_node="${#GPU_ID_ARR[@]}" --master_port="${MASTER_PORT}")
else
  LAUNCH_CMD=(python)
fi

echo "UniSharp training: run=${RUN_NAME} out=${OUT_ROOT} gpu=${GPU_IDS}"
echo "  branch=gt-override scratch_unik3d_pretrained"
echo "  datasets: re10k=${DATASET_WEIGHT_RE10K} hm3d=${DATASET_WEIGHT_HM3D} sim=${DATASET_WEIGHT_SIM} wildrgbd=${DATASET_WEIGHT_WILDRGBD} dl3dv=${DATASET_WEIGHT_DL3DV} scanetpp=${DATASET_WEIGHT_SCANETPP}"

exec "${LAUNCH_CMD[@]}" -m unisharp.cli train-feature \
  --out-root "${OUT_ROOT}" \
  --run-name "${RUN_NAME}" \
  --steps "${STEPS}" \
  --warmup "${WARMUP}" \
  --lr0 "${LR0}" \
  --lr1 "${LR1}" \
  --unik3d-lr0 "${UNIK3D_DECODER_LR0}" \
  --unik3d-lr1 "${UNIK3D_DECODER_LR1}" \
  --unik3d-encoder-lr0 "${UNIK3D_ENCODER_LR0}" \
  --unik3d-encoder-lr1 "${UNIK3D_ENCODER_LR1}" \
  --grad-clip-norm "${GRAD_CLIP_NORM}" \
  --max-step-grad-norm "${MAX_STEP_GRAD_NORM}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --device "${DEVICE}" \
  --ddp-timeout-hours "${DDP_TIMEOUT_HOURS}" \
  --max-index-gap "${MAX_INDEX_GAP}" \
  --max-depth-m "${MAX_DEPTH_M}" \
  --sim-far-depth-invalid-m "${SIM_FAR_DEPTH_INVALID_M}" \
  --sim-far-depth-invalid-max-frac "${SIM_FAR_DEPTH_INVALID_MAX_FRAC}" \
  --sim-max-long-edge "${SIM_MAX_LONG_EDGE}" \
  --pinhole-train-size "${PINHOLE_TRAIN_SIZE}" \
  --train-resize-multiple "${TRAIN_RESIZE_MULTIPLE}" \
  --scanetpp-fisheye-far-depth-invalid-m "${SCANETPP_FISHEYE_FAR_DEPTH_INVALID_M}" \
  --initializer-stride "${INITIALIZER_STRIDE}" \
  --initializer-scale-factor "${INITIALIZER_SCALE_FACTOR}" \
  --delta-rho-limit "${DELTA_RHO_LIMIT}" \
  --lambda-color "${LAMBDA_COLOR}" \
  --lambda-alpha "${LAMBDA_ALPHA}" \
  --lambda-percep "${LAMBDA_PERCEP}" \
  --lambda-depth "${LAMBDA_DEPTH}" \
  --lambda-tv "${LAMBDA_TV}" \
  --lambda-grad "${LAMBDA_GRAD}" \
  --lambda-grad-img "${LAMBDA_GRAD_IMG}" \
  --lambda-edge-rgb "${LAMBDA_EDGE_RGB}" \
  --lambda-delta "${LAMBDA_DELTA}" \
  --lambda-delta-rho "${LAMBDA_DELTA_RHO}" \
  --lambda-splat "${LAMBDA_SPLAT}" \
  --lambda-edge-splat "${LAMBDA_EDGE_SPLAT}" \
  --lambda-grid "${LAMBDA_GRID}" \
  --lambda-aux-ray "${LAMBDA_AUX_RAY}" \
  --lambda-aux-depth-scale "${LAMBDA_AUX_DEPTH_SCALE}" \
  --lambda-aux-depth2-scale "${LAMBDA_AUX_DEPTH2_SCALE}" \
  --dataset-weight-re10k "${DATASET_WEIGHT_RE10K}" \
  --dataset-weight-hm3d "${DATASET_WEIGHT_HM3D}" \
  --dataset-weight-sim "${DATASET_WEIGHT_SIM}" \
  --dataset-weight-wildrgbd "${DATASET_WEIGHT_WILDRGBD}" \
  --dataset-weight-dl3dv "${DATASET_WEIGHT_DL3DV}" \
  --dataset-weight-scanetpp "${DATASET_WEIGHT_SCANETPP}" \
  --data-root-re10k "${DATA_ROOT_RE10K}" \
  --re10k-pseudo-depth-root "${RE10K_PSEUDO_DEPTH_ROOT}" \
  --re10k-pseudo-far-depth-invalid-m "${RE10K_PSEUDO_FAR_DEPTH_INVALID_M}" \
  --data-root-hm3d "${DATA_ROOT_HM3D}" \
  --data-root-sim "${DATA_ROOT_SIM}" \
  --sim-pose-root "${SIM_POSE_ROOT}" \
  --wild-roots-file "${WILD_ROOTS_FILE}" \
  --data-root-dl3dv "${DATA_ROOT_DL3DV}" \
  --data-root-dl3dv-depth "${DATA_ROOT_DL3DV_DEPTH}" \
  --data-root-scanetpp "${DATA_ROOT_SCANETPP}" \
  --dataset-manifest-dir "${DATASET_MANIFEST_DIR}" \
  --save-every "${SAVE_EVERY}" \
  --vis-every "${VIS_EVERY}" \
  --log-every "${LOG_EVERY}" \
  --seed "${SEED}"
