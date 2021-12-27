#!/bin/bash

measures='CE_VAL PATH_NORM_OVER_EXPONENTIAL_MARGIN SPEC_INIT_MAIN_EXPONENTIAL_MARGIN MAG_INIT'
datasets="cifar10 svhn cifar100"

for m in $measures; do
  for d in $datasets; do
    sbatch -p small hpo.sh "$m" "$d"
  done
done