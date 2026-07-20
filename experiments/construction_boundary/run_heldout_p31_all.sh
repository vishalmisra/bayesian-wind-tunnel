#!/usr/bin/env bash
# Launch all 15 held-out-prefix cells (5 buckets x 3 seeds) at p=31
# across the 8 RTX 4090s on tokenprobe. Max 3 cells per GPU; the rest
# queue. Each cell writes its own log.
#
# Usage (on tokenprobe):
#   cd ~/bayesg/Nature-Paper-Round6/code
#   bash run_heldout_p31_all.sh [P]
#   tail -F logs/p31/*.log
#
# Optional arg: modulus (default 31). Use 17 to sanity-check the original
# replication; use 43 for the larger second pass.

set -u

P=${1:-31}
SEEDS=(42 1337 2024)
BUCKETS=(0 1 2 3 4)
N_GPUS=8
MAX_PER_GPU=${MAX_PER_GPU:-2}
BATCH_SIZE=${BATCH_SIZE:-16}
N_STEPS=${N_STEPS:-150000}

# Resolve paths
HERE=$(cd "$(dirname "$0")" && pwd)
SCRIPT="${HERE}/recurrence_heldout_prefix.py"
SOURCE_DIR=${SOURCE_DIR:-${HERE}}
OUT_ROOT=${OUT_ROOT:-${HERE}/results/heldout_prefix_p${P}}
LOG_DIR=${LOG_DIR:-${OUT_ROOT}/logs}

mkdir -p "${LOG_DIR}"

CONDA_INIT=${CONDA_INIT:-"source ${HOME}/miniconda3/etc/profile.d/conda.sh && conda activate bayesg"}

# Build the full work list: one entry per (seed, bucket) cell.
declare -a CELLS
for seed in "${SEEDS[@]}"; do
  for bucket in "${BUCKETS[@]}"; do
    CELLS+=("${seed}:${bucket}")
  done
done
TOTAL=${#CELLS[@]}

echo "Launching ${TOTAL} cells across ${N_GPUS} GPUs (cap ${MAX_PER_GPU}/GPU)"
echo "Modulus  : p=${P}"
echo "Script   : ${SCRIPT}"
echo "Source   : ${SOURCE_DIR}"
echo "Outputs  : ${OUT_ROOT}/seed_<seed>_holdout_<bucket>/"
echo "Logs     : ${LOG_DIR}/seed<seed>_holdout<bucket>_gpu<id>.log"

# In-flight counters per GPU.
declare -A IN_FLIGHT
for ((g=0; g<N_GPUS; g++)); do IN_FLIGHT[$g]=0; done

# PIDs per cell so we can wait at the end if desired.
declare -a PIDS

# Round-robin with capacity cap. When all GPUs are saturated we wait for
# any background job to finish, decrement that GPU's counter, then retry.
launched=0
for cell in "${CELLS[@]}"; do
  seed=${cell%:*}
  bucket=${cell#*:}

  # Find a GPU with capacity.
  gpu=-1
  while true; do
    for ((g=0; g<N_GPUS; g++)); do
      if (( IN_FLIGHT[$g] < MAX_PER_GPU )); then
        gpu=$g
        break
      fi
    done
    if (( gpu >= 0 )); then break; fi
    # All saturated: wait for any child to finish, then re-scan.
    wait -n
    # Re-scan in-flight by checking which PIDs are still alive.
    for ((i=0; i<${#PIDS[@]}; i++)); do
      pid=${PIDS[$i]}
      if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
        # decrement that PID's GPU
        g=${PID_GPU[$pid]:-}
        if [[ -n "${g}" ]]; then
          IN_FLIGHT[$g]=$(( IN_FLIGHT[$g] - 1 ))
          unset PID_GPU[$pid]
          PIDS[$i]=""
        fi
      fi
    done
  done

  cell_out="${OUT_ROOT}/seed_${seed}_holdout_${bucket}"
  cell_log="${LOG_DIR}/seed${seed}_holdout${bucket}_gpu${gpu}.log"
  mkdir -p "${cell_out}"

  echo "[$(date +%H:%M:%S)] launch seed=${seed} bucket=${bucket} -> cuda:${gpu}  log=${cell_log}"

  # nohup so the cell survives if the launching shell exits.
  nohup bash -c "${CONDA_INIT} && \
    python -u '${SCRIPT}' \
      --p ${P} \
      --seed ${seed} \
      --holdout_bucket ${bucket} \
      --batch_size ${BATCH_SIZE} \
      --n_steps ${N_STEPS} \
      --device cuda:${gpu} \
      --source_dir '${SOURCE_DIR}' \
      --output_dir '${cell_out}' \
      2>&1" > "${cell_log}" 2>&1 &
  pid=$!
  PIDS+=("${pid}")
  declare -A PID_GPU 2>/dev/null || true
  PID_GPU[$pid]=$gpu
  IN_FLIGHT[$gpu]=$(( IN_FLIGHT[$gpu] + 1 ))
  launched=$(( launched + 1 ))

  # Small stagger so 24 dataloaders don't fight for the same import lock.
  sleep 2
done

echo
echo "All ${launched} cells launched. In-flight per GPU: ${IN_FLIGHT[*]}"
echo "PIDs: ${PIDS[*]}"
echo
echo "Tail any log with:   tail -F ${LOG_DIR}/seed<seed>_holdout<bucket>_gpu<g>.log"
echo "Wait for all to finish: wait"
echo
echo "Aggregate when done:"
echo "  python -c \"import json,glob,numpy as np; "\
"rs=[json.load(open(p)) for p in sorted(glob.glob('${OUT_ROOT}/seed_*_holdout_*/metrics.json'))]; "\
"tr=[r['final_train_bucket']['kl_bits_mean'] for r in rs]; "\
"ho=[r['final_heldout_bucket']['kl_bits_mean'] for r in rs]; "\
"print(f'cells={len(rs)}  train KL={np.mean(tr):.6f}+/-{np.std(tr):.6f}  heldout KL={np.mean(ho):.6f}+/-{np.std(ho):.6f}')\""

# Optional: uncomment to block until all cells finish.
# wait
