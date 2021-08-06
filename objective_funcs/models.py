from torch import nn
from objective_funcs.dataset_helpers import get_dataloaders
from objective_funcs.config import ObjectiveType as OT
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from objective_funcs.measures import ACC

class Simple_NN(nn.Module):
    def __init__(self, input_size, layer1_size, layer2_size, layer3_size, dropout_rate):
        super(Simple_NN, self).__init__()
        self.layer_1 = nn.Sequential(
            nn.Linear(input_size, layer1_size),
            nn.BatchNorm1d(layer1_size),
            nn.ReLU(True),
            nn.Dropout(p=dropout_rate))
        self.layer_2 = nn.Sequential(
            nn.Linear(layer1_size, layer2_size),
            nn.BatchNorm1d(layer2_size),
            nn.ReLU(True),
            nn.Dropout(p=dropout_rate))
        self.layer_3 = nn.Sequential(
            nn.Linear(layer2_size, layer3_size),
            nn.BatchNorm1d(layer3_size),
            nn.ReLU(True),
            nn.Dropout(p=dropout_rate))
        self.regressor = nn.Linear(layer3_size, 1)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        y = self.regressor(x)
        return y


class NiNBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int, batch_norm: bool, dropout_prob: float) -> None:
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(planes) if batch_norm else lambda x: x
        self.dp1 = nn.Dropout2d(p=dropout_prob)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(planes) if batch_norm else lambda x: x
        self.dp2 = nn.Dropout2d(p=dropout_prob)

        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(planes) if batch_norm else lambda x: x
        self.dp3 = nn.Dropout2d(p=dropout_prob)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dp1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dp2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dp3(x)

        return x


class NiN(nn.Module):
    def __init__(self, depth: int, width: int, base_width: int, batch_norm: bool, dropout_prob: float) -> None:
        super().__init__()

        self.base_width = base_width

        blocks = []
        blocks.append(NiNBlock(3, self.base_width * width, batch_norm, dropout_prob))
        for _ in range(depth - 1):
            blocks.append(NiNBlock(self.base_width * width, self.base_width * width, batch_norm, dropout_prob))
        self.blocks = nn.Sequential(*blocks)

        self.conv = nn.Conv2d(self.base_width * width, 10, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(10) if batch_norm else lambda x: x
        self.dp = nn.Dropout2d(p=dropout_prob)
        self.relu = nn.ReLU(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.blocks(x)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dp(x)

        x = self.avgpool(x)

        return x.squeeze()


def get_converged_performance(config, device, data_dir, dataset_type):
    model = NiN(config['depth'], 8, 25, True, 0)
    model.to(device)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9, weight_decay=0)
    train_dataset, train_eval_loader, _, test_loader = get_dataloaders(data_dir, dataset_type, False, device)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)

    for _ in range(300):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

            optimizer.zero_grad()

            logits = model(data)
            cross_entropy = F.cross_entropy(logits, target)

            cross_entropy.backward()

            optimizer.step()

        train_acc = -ACC(model, train_eval_loader, device)
        if train_acc > 0.99:
            break

    test_acc = -ACC(model, test_loader, device)

    return test_acc
