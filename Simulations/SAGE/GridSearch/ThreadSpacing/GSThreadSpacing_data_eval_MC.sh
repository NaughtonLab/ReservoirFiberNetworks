#!/bin/bash

#SBATCH --cpus-per-task=30
#SBATCH --exclusive
#SBATCH -t 24:00:00
#SBATCH -p normal_q
#SBATCH -A naughton

#load module
module reset
module load python3

#compile
source /apps/common/software/Miniforge3/24.11.3-0/bin/activate elastica_venv_310
cd /projects/naughton/Apoorva/fiber_network_project/fiber_network/Simulations/SAGE/GridSearch/ThreadSpacing/
python3 GSThreadSpacing_data_eval_MC.py