#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=24:00:00

# set name of job
#SBATCH --job-name=hpo_measures

# set number of GPUs
#SBATCH --gres=gpu:1

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=charles.london@wolfson.ox.ac.uk

# run the application

module load python3/anaconda
source activate generalization

python3 bo_algorithms.py --seed="$1" --objective="$2" --epochs="$3" --dataset="$4" --bo_method="tpe"
python3 get_final_performance.py --seed="$1" --objective="$2" --epochs="$3" --dataset="$4" --bo_method="tpe"