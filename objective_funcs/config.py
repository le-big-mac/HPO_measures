from enum import Enum
import argparse
from typing import Tuple


class ObjectiveType(Enum):
    CE_TRAIN = 1
    CE_VAL = 2
    TRAIN_ACC = 3
    VAL_ACC = 4
    PATH_NORM = 5
    PATH_NORM_OVER_EXPONENTIAL_MARGIN = 6
    SPEC_INIT_MAIN_EXPONENTIAL_MARGIN = 7
    SOTL = 8
    PACBAYES_INIT = 11
    MAG_FLATNESS = 12
    MAG_INIT = 13
    DIST_SPEC_INIT_FFT = 14
    FRO_DIST = 15


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