# BO packages to tune hyperparamers

### Dependencies
* scipy
* scikit-learn
* gpy 
* gpyopt
* hyperopt
* hpbandster

### Instruction

Define the objective function to be minimised following the format in `objective.py`

E.g. To run TPE-BO on tuning fully connected network hyperparameters (FCNetTuning) for 60 optimisation iterations starting from 30 initial observation data (i.e. 30 (hyperparam, validation error) pairs):

```python bo_algorithms --task_name=FCNetTuning --bo_method=tpe --n_iters=60 --n_init=30``` 
