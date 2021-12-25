import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import ConfigSpace
from hyperopt import hp, STATUS_OK
from copy import deepcopy
from objective_funcs.models import NiN
from objective_funcs.config import ObjectiveType
from objective_funcs.dataset_helpers import get_dataloaders
from objective_funcs.measures import get_objective
from objective_funcs.measures import ACC
from objective_funcs.config import DatasetType


class TuneNN(object):

    def __init__(self, objective: ObjectiveType, data_dir: str, dataset='cifar10', seed=0, bo_method='gp',
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):

        self.objective = objective
        self.bo_method = bo_method
        self.dataset = dataset
        self.dataset_type = DatasetType[dataset.upper()]
        self.seed = seed
        torch.manual_seed(self.seed)
        self.device = device

        self.train_dataset, self.train_eval_loader, self.val_loader, self.test_loader \
            = get_dataloaders(data_dir, dataset, objective in [ObjectiveType.VAL_ACC, ObjectiveType.CE_VAL],
                              self.device)

    def objective_function(self, config, batches=10):
        # minimise validation error
        model = NiN(self.dataset_type, config['depth'], 8, 25, True, 0)
        model.to(self.device)
        init_model = deepcopy(model)
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9, weight_decay=0)
        # optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=0)
        train_loader = DataLoader(self.train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        # Train
        batch_losses = []

        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx > batches:
                break

            data, target = data.to(self.device), target.to(self.device)

            optimizer.zero_grad()

            logits = model(data)
            cross_entropy = F.cross_entropy(logits, target)

            batch_losses.append(cross_entropy.item())
            cross_entropy.backward()

            optimizer.step()

        val_error = get_objective(self.objective, model, init_model, self.train_eval_loader, self.val_loader,
                                  batch_losses, self.device, self.seed)

        end.record()
        torch.cuda.synchronize()
        run_cost = start.elapsed_time(end)

        del model
        del init_model

        return val_error, run_cost

    def eval(self, config, batches=10):

        if self.bo_method == 'tpe':
            config_standard = deepcopy(config)
            for h in self.space.get_hyperparameters():
                if type(h) == ConfigSpace.hyperparameters.OrdinalHyperparameter:
                    config_standard[h.name] = h.sequence[int(config[h.name])]
                elif type(h) == ConfigSpace.hyperparameters.UniformIntegerHyperparameter:
                    config_standard[h.name] = int(config[h.name])
                elif type(h) == ConfigSpace.hyperparameters.UniformFloatHyperparameter:
                    config_standard[h.name] = float(config[h.name])

            y, c = self.objective_function(config_standard, batches)
            return {
                'config': config,
                'loss': y,
                'cost': c,
                'status': STATUS_OK}

        elif self.bo_method == 'bohb':
            y, c = self.objective_function(config, batches)
            return y, c

        elif self.bo_method == 'gp':
            config_standard = {}
            for j, h in enumerate(self.search_space):
                config_standard[h['name']] = config[j]

            y, c = self.objective_function(config_standard, batches)
            return y, c

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

        elif self.bo_method == 'gp':
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
        space.add_hyperparameter(ConfigSpace.UniformFloatHyperparameter(f"lr", 1e-3, 1e-1))
        space.add_hyperparameter(ConfigSpace.UniformIntegerHyperparameter(f"batch_size", 32, 256))
        space.add_hyperparameter(ConfigSpace.UniformIntegerHyperparameter(f"depth", 2, 5))
        # space.add_hyperparameter(ConfigSpace.UniformFloatHyperparameter(f"dropout_rate", 0.0, 0.9))

        return space
