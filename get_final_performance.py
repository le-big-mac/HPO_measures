import pickle
import argparse
import numpy as np
import random
import torch
import os
import wandb
import hashlib
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from copy import deepcopy

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
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_dir = os.path.join(args.output_path, args.dataset, args.objective.name)
result_path = os.path.join(output_dir, "{}_epochs_{}_seed_{}_Adam.pickle".format(args.bo_method, args.epochs, args.seed))
output_file = os.path.join(output_dir, "{}_epochs_{}_Adam.txt".format(args.bo_method, args.epochs))

if args.seed == 0:
    try:
        os.remove(output_file)
    except OSError:
        pass

with open(result_path, "rb") as f:
    hp_configs = pickle.load(f)

best_hparams = min(hp_configs, key=lambda x: x['loss'])['config']

config = {"lr": float(best_hparams["lr"]), "batch_size": int(best_hparams['batch_size']),
          "depth": int(best_hparams['depth']), "seed": args.seed, "dataset": args.dataset, "objective": args.objective,
          "epochs": args.epochs}
group = {"dataset": args.dataset, "objective": args.objective, "epochs": args.epochs}
wandb.init(project="hpo_measures_new", config=config, group=hashlib.md5(str(group).encode('utf-8')).hexdigest())

model = NiN(config["depth"], 8, 25, True, 0)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=0)
train_dataset, train_eval_loader, _, test_loader = get_dataloaders(args.data_dir, args.dataset, False, device)
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0)
len_loader = len(train_loader)

print("Depth: {}".format(config["depth"]))
print("Lr: {}".format(config["lr"]))
print("Batch size: {}".format(config["batch_size"]))
print()
print(model)

step = 0
for epoch in range(300):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        step = (epoch - 1) * len_loader + batch_idx

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        logits = model(data)
        cross_entropy = F.cross_entropy(logits, target)
        wandb.log({"batch_loss": cross_entropy.item()}, step=step)

        cross_entropy.backward()

        optimizer.step()

    acc_model = deepcopy(model)
    train_acc = ACC(acc_model, train_eval_loader, device)
    test_acc = ACC(acc_model, test_loader, device)
    wandb.log({"train_accuracy": train_acc, "test_accuracy": test_acc}, step=step)

    print("Epoch: {}".format(epoch))
    print("Train acc: {}".format(train_acc))
    print("Test acc: {}". format(test_acc))
    if train_acc > 0.99:
        break

model.eval()
test_acc = ACC(model, test_loader, device)

with open(output_file, 'a+') as f:
    f.write("Config: {}, Seed: {}, Test acc: {}\n".format(best_hparams, args.seed, test_acc))
