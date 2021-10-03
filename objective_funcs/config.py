from enum import Enum
import argparse
from typing import Tuple


class ObjectiveType(Enum):
    CE_TRAIN = 1
    TRAIN_ACC = 2
    MAG_FLATNESS = 3
    PATH_NORM = 4
    VAL_ACC = 5
    SOTL = 6
    PARAM_NORM = 7
    FRO_DIST = 8
    L2_DIST = 9
    LOG_PROD_OF_SPEC = 10
    PACBAYES_INIT = 11
    DIST_SPEC_INIT_FFT = 12


def objective_type(key):
    try:
        return ObjectiveType[key.upper()]
    except KeyError:
        raise argparse.ArgumentError()


class DatasetType(Enum):
    CIFAR10 = (1, (3, 32, 32), 10)
    CIFAR100 = (2, (3, 32, 32), 100)
    SVHN = (3, (3, 32, 32), 10)

    def __init__(self, id: int, image_shape: Tuple[int, int, int], num_classes: int):
        self.D = image_shape
        self.K = num_classes