import pickle
import argparse
import numpy as np
import random
import torch
import os
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from objective_funcs.models import NiN
from objective_funcs.config import objective_type
from objective_funcs.dataset_helpers import get_dataloaders
from objective_funcs.measures import ACC


parser = argparse.ArgumentParser()
parser.add_argument('--objective', default='val_acc', type=objective_type, help='specifies objective function to use')
parser.add_argument('--epochs', default=5, type=int, help='number of epochs to run before calculating measure')
parser.add_argument('--dataset', default="cifar10", type=str, help='specifies the dataset to run on')
parser.add_argument('--bo_method', default="tpe", type=str, help='bo method used: bohb, tpe, gpbo')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--data_dir', default="./data/", type=str, help='specifies the path to the datasets')
parser.add_argument('--output_path', default="./results/", type=str,
                    help='specifies the path where the results will be saved')

args = parser.parse_args()
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_dir = os.path.join(args.output_path, args.dataset, args.objective.name)
result_path = os.path.join(output_dir, "{}_epochs_{}_seed_{}.pickle".format(args.bo_method, args.epochs, args.seed))

with open(result_path, "rb") as f:
    hp_configs = pickle.load(f)

best_hparams = min(hp_configs, key=lambda x: x['loss'])['config']

model = NiN(int(best_hparams['depth']), 8, 25, True, 0)
model.to(device)
model.train()
optimizer = optim.SGD(model.parameters(), lr=float(best_hparams["lr"]), momentum=0.9, weight_decay=0)
train_dataset, train_eval_loader, _, test_loader = get_dataloaders(args.data_dir, args.dataset, False, device)
train_loader = DataLoader(train_dataset, batch_size=int(best_hparams['batch_size']), shuffle=True, num_workers=0)

for _ in range(200):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        logits = model(data)
        cross_entropy = F.cross_entropy(logits, target)

        cross_entropy.backward()

        optimizer.step()

    train_acc = ACC(model, train_eval_loader, device)
    if train_acc > 0.99:
        break

test_acc = ACC(model, test_loader, device)

with open(os.path.join(output_dir, "{}_epochs_{}.txt".format(args.bo_method, args.epochs)), 'a+') as f:
    f.write("Seed: {}, Test acc: {}\n".format(args.seed, test_acc))
