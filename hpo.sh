#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=48:00:00

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

batches="1 10 100"
seeds="0 17 43"

for s in $seeds; do
  for b in $batches; do
    python3 bo_algorithms.py --seed="$s" --objective="$1" --batches="$b" --dataset="$2" --bo_method="tpe"
    python3 get_final_performance.py --seed="$s" --objective="$1" --batches="$b" --dataset="$2" --bo_method="tpe"
  done
done