#!/bin/bash
#SBATCH --job-name=opVsthread_2x2
#SBATCH --cpus-per-task=60
#SBATCH --mem-per-cpu=2G
#SBATCH --time=24:00:00
#SBATCH --partition=normal_q
#SBATCH --account=naughton
#SBATCH -o out/%x_%j.out
#SBATCH -e out/%x_%j.err

module reset
module load python3

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS

cd /projects/naughton/Apoorva/fiber_network
python3 fiber_config_num_ops_vs_threads.py --num_threads 2