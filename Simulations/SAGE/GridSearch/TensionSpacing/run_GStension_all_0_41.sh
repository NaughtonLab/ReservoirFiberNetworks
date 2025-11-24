#!/bin/bash
#SBATCH --job-name=GStension_Array
#SBATCH --array=0-41
#SBATCH --cpus-per-task=96
#SBATCH --mem-per-cpu=2G
#SBATCH --time=3-00:00:00
#SBATCH --partition=normal_q
#SBATCH --account=naughton
#SBATCH -o /projects/naughton/Apoorva/fiber_network_project/out/%x_%A_%a.out
#SBATCH -e /projects/naughton/Apoorva/fiber_network_project/out/%x_%A_%a.err

module reset
module load python3

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS

SRUN_CORES=8
MAX_PARALLEL=6
running=0

# Determine range based on SLURM_ARRAY_TASK_ID
# Simple formula: Each ID handles a block of 10
ID=$SLURM_ARRAY_TASK_ID
START=$((ID * 10))
END=$((START + 9))

# Cap the end index at 41
if [ $END -gt 41 ]; then
  END=41
fi

BASE_FILE_NAME='fiber_config_GSTensionSpacing'
FILE_NO=($(seq $START $END))
FILE="${BASE_FILE_NAME}.py"

source /apps/common/software/Miniforge3/24.11.3-0/bin/activate elastica_venv_310
cd /projects/naughton/Apoorva/fiber_network_project/fiber_network/ || { echo "Cannot cd into dir"; exit 1; }

echo "Job Array ID: $ID"
echo "Processing indices: ${FILE_NO[@]}"
echo "Using config file: $FILE"

for N in "${FILE_NO[@]}"; do
  
  if [[ ! -f "$FILE" ]]; then
    echo "WARNING: $FILE not found, skipping"
    continue
  fi

  echo "Launching $FILE with --grid_idx $N ..."
  # launch a single-task joblet bound to SRUN_CORES CPUs
  srun --ntasks=1 --cpus-per-task="${SRUN_CORES}" \
       --cpu-bind=cores --exclusive \
       python3 "$FILE" --grid_idx "$N" &

  # portable throttle (avoids 'wait -n' which some clusters lack)
  while [[ $(jobs -rp | wc -l) -ge $MAX_PARALLEL ]]; do
    sleep 0.5
  done
done

wait
echo "Fiber GS finished for Array ID $ID"
