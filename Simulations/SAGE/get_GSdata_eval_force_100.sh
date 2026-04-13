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
cd /projects/naughton/Apoorva/fiber_network
python3 get_GSData_eval_force.py