#!/bin/bash
#SBATCH --job-name=GSthreadspacing_eval_NL
#SBATCH --cpus-per-task=30
#SBATCH --exclusive
#SBATCH -t 24:00:00
#SBATCH -p normal_q
#SBATCH -A naughton
#SBATCH -o /projects/naughton/Apoorva/fiber_network_project/out/%x_%j.out
#SBATCH -e /projects/naughton/Apoorva/fiber_network_project/out/%x_%j.err

#load module
module reset
module load python3

#compile
source activate elastica_venv_310
cd /projects/naughton/Apoorva/fiber_network_project/fiber_network/Simulations/SAGE/GridSearch/ThreadSpacing/
python3 GSThreadSpacing_data_eval_NL.py