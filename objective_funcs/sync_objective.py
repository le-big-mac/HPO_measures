import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from typing import Dict, List, Optional, Tuple
import logging
from logging.handlers import RotatingFileHandler
import sys
import time
import os
import ConfigSpace
from hyperopt import hp, STATUS_OK
from copy import deepcopy


class sync_func(object):

    def __init__(self, task='egg', seed=None, bo_method='gp'):

        self.bo_method = bo_method
        self.task = task
        self.seed = seed

    def objective_function(self, config, epochs=1):

        # minimise validation error
        if self.task == 'egg':
            # egg_true_location = np.array([[1.0, 0.7895]])
            # egg_true_min = np.array([[-9.59640593]])
            x0 = config['x0'] * 512
            x1 = config['x1'] * 512
            term1 = -(x1 + 47) * np.sin(np.sqrt(np.abs(x1 + x0 / 2 + 47)))
            term2 = -x0 * np.sin(np.sqrt(np.abs(x0 - (x1 + 47))))
            y = (term1 + term2) / 100
            c = 0

        return y, c

    def eval(self, config, budget=1):

        if self.bo_method == 'tpe':
            config_standard = deepcopy(config)
            for h in self.space.get_hyperparameters():
                if type(h) == ConfigSpace.hyperparameters.OrdinalHyperparameter:
                    config_standard[h.name] = h.sequence[int(config[h.name])]
                elif type(h) == ConfigSpace.hyperparameters.UniformIntegerHyperparameter:
                    config_standard[h.name] = int(config[h.name])
                elif type(h) == ConfigSpace.hyperparameters.UniformFloatHyperparameter:
                    config_standard[h.name] = config[h.name]

            y, c = self.objective_function(config_standard, budget)
            return {
                'config': config,
                'loss': y,
                'cost': c,
                'status': STATUS_OK}

        elif self.bo_method == 'bohb':
            y, c = self.objective_function(config, budget)
            return y, c

        elif self.bo_method == 'gpbo':
            config_standard = {}
            for j, h in enumerate(self.search_space):
                config_standard[h['name']] = config[:, j]

            y, c = self.objective_function(config_standard, budget)
            return y

    def get_search_space(self):
        space = self.get_configuration_space()
        self.space = space
        if self.bo_method == 'tpe':
            search_space = {}
            for h in space.get_hyperparameters():
                if type(h) == ConfigSpace.hyperparameters.OrdinalHyperparameter:
                    search_space[h.name] = hp.quniform(h.name, 0, len(h.sequence) - 1, q=1)
                elif type(h) == ConfigSpace.hyperparameters.CategoricalHyperparameter:
                    search_space[h.name] = hp.choice(h.name, h.choices)
                elif type(h) == ConfigSpace.hyperparameters.UniformIntegerHyperparameter:
                    search_space[h.name] = hp.quniform(h.name, h.lower, h.upper, q=1)
                elif type(h) == ConfigSpace.hyperparameters.UniformFloatHyperparameter:
                    search_space[h.name] = hp.uniform(h.name, h.lower, h.upper)
        elif self.bo_method == 'gpbo':
            search_space = []
            for h in space.get_hyperparameters():
                if type(h) == ConfigSpace.hyperparameters.CategoricalHyperparameter:
                    type_name = 'categorical'
                    domain = h.choices
                elif type(h) == ConfigSpace.hyperparameters.UniformFloatHyperparameter:
                    type_name = 'continuous'
                    domain = (h.lower, h.upper)
                elif type(h) == ConfigSpace.hyperparameters.UniformIntegerHyperparameter:
                    type_name = 'discrete'
                    domain = tuple(range(h.lower, h.upper))
                h_dict = {'name': h.name, 'type': type_name, 'domain': domain}
                search_space.append(h_dict)

        self.search_space = search_space
        return search_space

    @staticmethod
    def get_configuration_space():
        space = ConfigSpace.ConfigurationSpace()
        space.add_hyperparameter(ConfigSpace.UniformFloatHyperparameter(f"x0", 0, 1))
        space.add_hyperparameter(ConfigSpace.UniformFloatHyperparameter(f"x1", 0, 1))

        return space

