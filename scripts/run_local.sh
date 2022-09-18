#!/bin/bash
# Runs experiments on a local computer.
#
# Usage:
#   bash scripts/run_local.sh CONFIG SEED NUM_WORKERS [RELOAD_PATH]
# Example:
#   # 8 workers with configuration config/foo.gin and seed 1.
#   bash scripts/run_local.sh config/foo.gin 1 8
#
#   # 4 workers with configuration config/foo.gin and seed 2, and reloading from
#   # old_dir/.
#   bash scripts/run_local.sh config/foo.gin 2 4 old_dir/

print_header() {
  echo
  echo "------------- $1 -------------"
}

# Prints "=" across an entire line.
print_thick_line() {
  printf "%0.s=" $(seq 1 `tput cols`)
  echo
}

print_header "Create logging directory"
DATE="$(date +'%Y-%m-%d_%H-%M-%S')"
LOGDIR="local_logs/local_${DATE}"
mkdir -p "$LOGDIR"
echo "LOCAL Log directory: ${LOGDIR}"

#
# Parse command line flags.
#

CONFIG="$1"
SEED="$2"
NUM_WORKERS="$3"
RELOAD_PATH="$4"
if [ -z "${NUM_WORKERS}" ]
then
  echo "Usage: bash scripts/run_local.sh CONFIG SEED NUM_WORKERS [RELOAD_PATH]"
  exit 1
fi

if [ -n "$RELOAD_PATH" ]; then
  RELOAD_ARG="--reload-dir ${RELOAD_PATH}"
else
  RELOAD_ARG=""
fi

set -u  # Uninitialized vars are error.

#
# Setup env.
#

# Remove MuJoCo locks if needed.
bash scripts/rm_mujoco_lock.sh

#
# Run the experiment.
#

SINGULARITY_OPTS="--cleanenv --no-home --bind $PWD"
SCHEDULER_JSON="./$(date +'%Y-%m-%d_%H-%M-%S')_scheduler_info.json"
PIDS_TO_KILL=()

# Use different port number so multiple jobs can start on one node. 8786
# is the default port. Then add offset of 10 and then the seed.
SCHEDULER_PORT=$((8786 + 10 + $SEED))

print_header "Starting Dask scheduler"
SCHEDULER_OUT="$LOGDIR/scheduler.out"
echo "(Output goes to $SCHEDULER_OUT)"
singularity exec $SINGULARITY_OPTS container.sif \
  dask-scheduler \
    --port $SCHEDULER_PORT \
    --scheduler-file $SCHEDULER_JSON &> "$SCHEDULER_OUT" &
PIDS_TO_KILL+=("$!")
sleep 2 # Wait for scheduler to start.

print_header "Starting Dask workers"
WORKER_OUT="$LOGDIR/workers.out"
echo "(Output goes to $WORKER_OUT)"
singularity exec $SINGULARITY_OPTS container.sif \
  dask-worker \
    --scheduler-file $SCHEDULER_JSON \
    --nprocs $NUM_WORKERS \
    --nthreads 1 &> "$WORKER_OUT" &
PIDS_TO_KILL+=("$!")
sleep 5

print_header "Running experiment"
EXPERIMENT_OUT="$LOGDIR/experiment.out"
echo "(Output goes to $EXPERIMENT_OUT)"
echo
print_thick_line
singularity exec $SINGULARITY_OPTS --nv container.sif \
  python -m src.main \
    --config "$CONFIG" \
    --address "127.0.0.1:$SCHEDULER_PORT" \
    --local-logdir "$LOGDIR" \
    --seed "$SEED" \
    $RELOAD_ARG 2>&1 | tee "$EXPERIMENT_OUT"
print_thick_line

#
# Clean Up.
#

print_header "Cleanup"
for pid in ${PIDS_TO_KILL[*]}
do
  kill -9 "${pid}"
done

rm $SCHEDULER_JSON

print_header "Logdir: $LOGDIR"
echo "\
Once this script has terminated, move these outputs to the experiment's
logging directory using:

  bash scripts/local_postprocess.sh $LOGDIR
"
