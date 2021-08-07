import argparse
import os
from hpbandster.optimizers.bohb import BOHB
import hpbandster.core.nameserver as hpns
from hpbandster.core.worker import Worker
from hyperopt import fmin, tpe, Trials
import pickle
import numpy as np
import random
import GPyOpt
from objective_funcs.objective import TuneNN
from objective_funcs.config import objective_type
import torch
from functools import partial


parser = argparse.ArgumentParser()
parser.add_argument('--objective', default='val_acc', type=objective_type, help='specifies objective function to use')
parser.add_argument('--epochs', default=5, type=int, help='number of epochs to run before calculating measure')
parser.add_argument('--dataset', default="cifar10", type=str, help='specifies the dataset to run on')
parser.add_argument('--bo_method', default="tpe", type=str, help='bo method used: bohb, tpe, gpbo')
parser.add_argument('--n_iters', default=20, type=int, help='number of BO iterations for optimization method')
parser.add_argument('--output_path', default="./results/", type=str,
                    help='specifies the path where the results will be saved')
parser.add_argument('--data_dir', default="./data/", type=str, help='specifies the path to the datasets')
parser.add_argument('--n_init', type=int, default=5, help='number of initial data')
parser.add_argument('--seed', type=int, default=42, help='random seed')

args = parser.parse_args()

print(args)
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)

# warmup GPU for cost timing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
a = torch.rand((3000, 3000), device=device)
b = torch.rand((3000, 3000), device=device)
torch.matmul(a, b)
del a
del b

# define the objective problem
b = TuneNN(objective=args.objective, data_dir=args.data_dir, dataset=args.dataset, seed=args.seed,
           bo_method=args.bo_method)

# create the result saving path
output_dir = os.path.join(args.output_path, args.dataset, args.objective.name)
os.makedirs(output_dir, exist_ok=True)
result_path = os.path.join(output_dir, "{}_epochs_{}_seed_{}.pickle".format(args.bo_method, args.epochs, args.seed))
bo_results = []

if args.bo_method == 'bohb':

    # ---- BO strategy A: BOHB ------

    # define the search space
    search_space = b.get_configuration_space()

    # modify the objective function format to apply this BO method
    class MyWorker(Worker):
        def compute(self, config, budget, **kwargs):
            y, cost = b.eval(config, budget=int(budget))
            return ({
                'loss': float(y),
                'info': float(cost)})

    # specify configurations of BOHB:
    hb_run_id = f'{args.seed}'
    min_bandwidth = 0.3
    num_workers = 1
    min_budget = 30
    max_budget = 90

    # initialise BOHB
    NS = hpns.NameServer(run_id=hb_run_id, host='localhost', port=0)
    ns_host, ns_port = NS.start()
    workers = []
    for i in range(num_workers):
        w = MyWorker(nameserver=ns_host, nameserver_port=ns_port,
                     run_id=hb_run_id, id=i)
        w.run(background=True)
        workers.append(w)

    bohb = BOHB(configspace=search_space, run_id=hb_run_id,
                eta=3, min_budget=min_budget, max_budget=max_budget,
                nameserver=ns_host, nameserver_port=ns_port,
                ping_interval=10, min_bandwidth=min_bandwidth)

    # run BOHB
    results = bohb.run(int(args.n_iters+args.n_init), min_n_workers=num_workers)
    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()

    # process the returned results to give the same format
    bo_results = []
    curr_best = np.inf
    curr_best_hyperparam = None
    for item in results.data.items():
        key, datum = item
        budget = datum.budget
        hyperparams = datum.config
        object_values = datum.results[budget]['loss']
        if object_values < curr_best:
            curr_best = object_values
            curr_best_hyperparam = hyperparams

        if budget == max_budget:
            bo_results.append((hyperparams, object_values))

    print(f'Best hyperparams={curr_best_hyperparam} with objective value={curr_best}')

elif args.bo_method == 'tpe':

    # ------ BO strategy B: TPE ------

    # define the search space
    search_space = b.get_search_space()

    # initialise and run TPE
    trials = Trials()
    objective = partial(b.eval, epochs=args.epochs)
    best_hyperparam = fmin(objective, space=search_space, algo=tpe.suggest, max_evals=int(args.n_iters+args.n_init),
                           trials=trials)
    # process the returned results to give the same format
    bo_results = [{'config': item['config'], 'loss': item['loss'], 'cost': item['cost']} for item in trials.results]
    best_objective_value = np.min([y['loss'] for y in trials.results])
    print(f'Best hyperparams={best_hyperparam} with objective value={best_objective_value}')

elif args.bo_method == 'gpbo':

    # ------ BO strategy C: GPyOpt ------

    # define the search space
    search_space = b.get_search_space()

    # initialise and run GPyOpt
    gpyopt = GPyOpt.methods.BayesianOptimization(f=b.eval, domain=search_space,
                                                 initial_design_numdata=args.n_init,
                                                 acquisition_type='EI', model_type='GP',
                                                 model_update_interval=5, verbosity=True,
                                                 normalize_Y=False)
    gpyopt.run_optimization(max_iter=args.n_iters)

    # process the returned results to give the same format
    Y_queried = gpyopt.Y
    X_queried = gpyopt.X
    best_hyperparam = np.atleast_2d(gpyopt.x_opt)
    best_objective_value = gpyopt.Y_best[-1]

    bo_results = []
    for j, x in enumerate(X_queried):
        bo_results.append((x, Y_queried[j]))

    print(f'Best hyperparams={best_hyperparam} with objective value={best_objective_value}')

pickle.dump(bo_results, open(result_path, 'wb'))
