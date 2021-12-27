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

seeds="0 17 43"
epochs="1 5 10"

for s in $seeds; do
  for e in $epochs; do
    python3 bo_algorithms.py --seed="$s" --objective="$1" --epochs="$e" --dataset="$2" --bo_method="tpe"
    python3 get_final_performance.py --seed="$s" --objective="$1" --epochs="$e" --dataset="$2" --bo_method="tpe"
  done
done