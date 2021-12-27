from contextlib import contextmanager
from copy import deepcopy
import math
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data.dataloader import DataLoader

from objective_funcs.config import ObjectiveType as OT
from objective_funcs.models import NiN


# Adapted from https://github.com/bneyshabur/generalization-bounds/blob/master/measures.py
@torch.no_grad()
def _reparam(model):
    def in_place_reparam(model, prev_layer=None):
        for child in model.children():
            prev_layer = in_place_reparam(child, prev_layer)
            if child._get_name() == 'Conv2d':
                prev_layer = child
            elif child._get_name() == 'BatchNorm2d':
                scale = child.weight / ((child.running_var + child.eps).sqrt())
                prev_layer.bias.copy_(child.bias + (scale * (prev_layer.bias - child.running_mean)))
                perm = list(reversed(range(prev_layer.weight.dim())))
                prev_layer.weight.copy_((prev_layer.weight.permute(perm) * scale).permute(perm))
                child.bias.fill_(0)
                child.weight.fill_(1)
                child.running_mean.fill_(0)
                child.running_var.fill_(1)
        return prev_layer

    model = deepcopy(model)
    in_place_reparam(model)
    return model


@contextmanager
def _perturbed_model(
        model: NiN,
        sigma: float,
        rng,
        magnitude_eps: Optional[float] = None
):
    device = next(model.parameters()).device
    if magnitude_eps is not None:
        noise = [torch.normal(0, sigma ** 2 * torch.abs(p) ** 2 + magnitude_eps ** 2, generator=rng) for p in
                 model.parameters()]
    else:
        noise = [torch.normal(0, sigma ** 2, p.shape, generator=rng).to(device) for p in model.parameters()]
    model = deepcopy(model)
    try:
        [p.add_(n) for p, n in zip(model.parameters(), noise)]
        yield model
    finally:
        [p.sub_(n) for p, n in zip(model.parameters(), noise)]
        del model


# Adapted from https://drive.google.com/file/d/1_6oUG94d0C3x7x2Vd935a2QqY-OaAWAM/view
def _pacbayes_sigma(
        model: NiN,
        dataloader: DataLoader,
        accuracy: float,
        seed: int,
        magnitude_eps: Optional[float] = None,
        search_depth: int = 15,
        montecarlo_samples: int = 10,
        accuracy_displacement: float = 0.1,
        displacement_tolerance: float = 1e-2,
) -> float:
    lower, upper = 0, 2
    sigma = 1

    BIG_NUMBER = 10348628753
    device = next(model.parameters()).device
    rng = torch.Generator(device=device) if magnitude_eps is not None else torch.Generator()
    rng.manual_seed(BIG_NUMBER + seed)

    for _ in range(search_depth):
        sigma = (lower + upper) / 2
        accuracy_samples = []
        for _ in range(montecarlo_samples):
            with _perturbed_model(model, sigma, rng, magnitude_eps) as p_model:
                loss_estimate = 0
                for data, target in dataloader:
                    logits = p_model(data)
                    pred = logits.data.max(1, keepdim=True)[1]  # get the index of the max logits
                    batch_correct = pred.eq(target.data.view_as(pred)).type(torch.FloatTensor).cpu()
                    loss_estimate += batch_correct.sum()
                loss_estimate /= len(dataloader.dataset)
                accuracy_samples.append(loss_estimate)
        displacement = abs(np.mean(accuracy_samples) - accuracy)
        if abs(displacement - accuracy_displacement) < displacement_tolerance:
            break
        elif displacement > accuracy_displacement:
            # Too much perturbation
            upper = sigma
        else:
            # Not perturbed enough to reach target displacement
            lower = sigma
    return sigma


def _pacbayes_bound(reference_vec: Tensor, sigma) -> Tensor:
    return (reference_vec.norm(p=2) ** 2) / (2 * sigma ** 2)


def _pacbayes_mag_bound(reference_vec: Tensor, dist_w_vec, mag_sigma, mag_eps, omega) -> Tensor:
    numerator = mag_eps ** 2 + (mag_sigma ** 2 + 1) * (reference_vec.norm(p=2) ** 2) / omega
    denominator = mag_eps ** 2 + mag_sigma ** 2 * dist_w_vec ** 2
    return 1 / 2 * (numerator / denominator).log().sum()


def get_weights_only(model):
    blacklist = {'bias', 'bn'}
    return [p for name, p in model.named_parameters() if all(x not in name for x in blacklist)]


def get_reshaped_weights(weights: List[Tensor]) -> List[Tensor]:
    # If the weight is a tensor (e.g. a 4D Conv2d weight), it will be reshaped to a 2D matrix
    return [p.view(p.shape[0], -1) for p in weights]


def get_vec_params(weights: List[Tensor]) -> Tensor:
    return torch.cat([p.view(-1) for p in weights], dim=0)


def _spectral_norm_fft(kernel: Tensor, input_shape: Tuple[int, int]) -> Tensor:
    # PyTorch conv2d filters use Shape(out,in,kh,kw)
    # [Sedghi 2018] code expects filters of Shape(kh,kw,in,out)
    # Pytorch doesn't support complex FFT and SVD, so we do this in numpy
    np_kernel = np.einsum('oihw->hwio', kernel.data.cpu().numpy())
    transforms = np.fft.fft2(np_kernel, input_shape, axes=[0, 1])  # Shape(ih,iw,in,out)
    singular_values = np.linalg.svd(transforms, compute_uv=False)  # Shape(ih,iw,min(in,out))
    spec_norm = singular_values.max()
    return torch.tensor(spec_norm, device=kernel.device)


def _margin(model, dataloader) -> Tensor:
        model = deepcopy(model)
        m = len(dataloader.dataset)
        margins = []
        for data, target in dataloader:
            logits = model(data)
            correct_logit = logits[torch.arange(logits.shape[0]), target].clone()
            logits[torch.arange(logits.shape[0]), target] = float('-inf')
            max_other_logit = logits.data.max(1).values  # get the index of the max logits
            margin = correct_logit - max_other_logit
            margins.append(margin)
        del model
        return torch.cat(margins).kthvalue(m // 10)[0].item()


def CE_TRAIN(train_history):
    return train_history[-1]


def CE_VAL(model, loader, device):
    model.eval()
    loss = 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        logits = model(data)
        loss += F.cross_entropy(logits, target).item()

    return loss


@torch.no_grad()
def ACC(model, loader, device):
    model.eval()
    num_correct = 0
    len_loader = len(loader.dataset)

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        logits = model(data)

        pred = logits.data.max(1, keepdim=True)[1]  # get the index of the max logits
        batch_correct = pred.eq(target.data.view_as(pred)).type(torch.FloatTensor).cpu()
        num_correct += batch_correct.sum()

    return num_correct.item() / len_loader


def SOTL(train_history):
    return sum(train_history)


def PATH_NORM(model, device):
    model = deepcopy(model)
    model.eval()
    for param in model.parameters():
        if param.requires_grad:
            param.data.pow_(2)
    x = torch.ones([1] + [3, 32, 32], device=device)
    x = model(x)
    del model
    return x.sum().item()


def PATH_NORM_OVER_EXPONENTIAL_MARGIN(model, train_loader, device):
    return PATH_NORM(model, device) / np.exp(-_margin(model, train_loader))


def SPEC_INIT_MAIN_EXPONENTIAL_MARGIN(model, init_model, train_loader):
    weights = get_weights_only(model)
    input_shape = (32, 32)
    dist_init_weights = [p - q for p, q in zip(weights, get_weights_only(init_model))]
    dist_reshaped_weights = get_reshaped_weights(dist_init_weights)

    fft_spec_norms = torch.cat([_spectral_norm_fft(p, input_shape).unsqueeze(0) ** 2 for p in weights])
    dist_fro_norms = torch.cat([p.norm('fro').unsqueeze(0) ** 2 for p in dist_reshaped_weights])
    margin = _margin(model, train_loader)

    spec_init = fft_spec_norms.log().sum() + 2 * margin + (dist_fro_norms / fft_spec_norms).sum().log()
    return spec_init.item()


def FRO_DIST(model, init_model):
    weights = get_weights_only(model)
    dist_init_weights = [p - q for p, q in zip(weights, get_weights_only(init_model))]
    dist_reshaped_weights = get_reshaped_weights(dist_init_weights)
    dist_fro_norms = torch.cat([p.norm('fro').unsqueeze(0) ** 2 for p in dist_reshaped_weights])

    return dist_fro_norms.sum().item()


def PARAM_NORM(model):
    weights = get_weights_only(model)
    reshaped_weights = get_reshaped_weights(weights)
    fro_norms = torch.cat([p.norm('fro').unsqueeze(0) ** 2 for p in reshaped_weights])

    return fro_norms.sum().item()


def PACBAYES_INIT(model, init_model, train_eval_loader, train_acc, seed):
    sigma = _pacbayes_sigma(model, train_eval_loader, train_acc, seed)

    weights = get_weights_only(model)
    dist_init_weights = [p - q for p, q in zip(weights, get_weights_only(init_model))]
    dist_w_vec = get_vec_params(dist_init_weights)

    return _pacbayes_bound(dist_w_vec, sigma).item()


def MAG_FLATNESS(model, train_eval_loader, train_acc, seed):
    mag_eps = 1e-3
    mag_sigma = _pacbayes_sigma(model, train_eval_loader, train_acc, seed, mag_eps)

    return torch.tensor(1 / mag_sigma ** 2).item()


def MAG_INIT(model, init_model, train_eval_loader, train_acc, seed):
    mag_eps = 1e-3
    mag_sigma = _pacbayes_sigma(model, train_eval_loader, train_acc, seed, mag_eps)

    weights = get_weights_only(model)
    dist_init_weights = [p - q for p, q in zip(weights, get_weights_only(init_model))]
    dist_w_vec = get_vec_params(dist_init_weights)
    omega = len(dist_w_vec)

    return _pacbayes_mag_bound(dist_w_vec, dist_w_vec, mag_sigma, mag_eps, omega).item()


def DIST_SPEC_INIT_FFT(model, init_model):
    weights = get_weights_only(model)
    dist_init_weights = [p - q for p, q in zip(weights, get_weights_only(init_model))]

    input_shape = (32, 32)
    fft_dist_spec_norms = torch.cat([_spectral_norm_fft(p, input_shape).unsqueeze(0) ** 2 for p in dist_init_weights])

    return fft_dist_spec_norms.sum().item()


@torch.no_grad()
def get_objective(objective: OT, model, init_model, train_eval_loader, val_loader, train_history, device, seed):
    model = _reparam(model)
    init_model = _reparam(init_model)

    if objective == OT.CE_TRAIN:
        return CE_TRAIN(train_history)
    elif objective == OT.CE_VAL:
        return CE_VAL(model, val_loader, device)
    elif objective == OT.TRAIN_ACC:
        return -ACC(model, train_eval_loader, device)
    elif objective == OT.VAL_ACC:
        return -ACC(model, val_loader, device)
    elif objective == OT.PATH_NORM:
        return PATH_NORM(model, device)
    elif objective == OT.PATH_NORM_OVER_EXPONENTIAL_MARGIN:
        return PATH_NORM_OVER_EXPONENTIAL_MARGIN(model, train_eval_loader, device)
    elif objective == OT.SPEC_INIT_MAIN_EXPONENTIAL_MARGIN:
        return SPEC_INIT_MAIN_EXPONENTIAL_MARGIN(model, init_model, train_eval_loader)
    elif objective == OT.SOTL:
        return SOTL(train_history)
    elif objective == OT.PACBAYES_INIT:
        return PACBAYES_INIT(model, init_model, train_eval_loader, ACC(model, train_eval_loader, device), seed)
    elif objective == OT.MAG_FLATNESS:
        return MAG_FLATNESS(model, train_eval_loader, ACC(model, train_eval_loader, device), seed)
    elif objective == OT.MAG_INIT:
        return MAG_INIT(model, init_model, train_eval_loader, ACC(model, train_eval_loader, device), seed)
    elif objective == OT.DIST_SPEC_INIT_FFT:
        return -DIST_SPEC_INIT_FFT(model, init_model)
    elif objective == OT.FRO_DIST:
        return FRO_DIST(model, init_model)
    else:
        raise KeyError
