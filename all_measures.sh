#!/bin/bash

measures='CE_TRAIN TRAIN_ACC VAL_ACC SOTL PATH_NORM PARAM_NORM FRO_DIST DIST_SPEC_INIT_FFT PACBAYES_INIT MAG_FLATNESS'

for m in $measures; do
  sbatch -p small hpo.sh "$m" 1 "cifar10"
done