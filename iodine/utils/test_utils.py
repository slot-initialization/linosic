import torch
import torchmetrics as tom
from ari_metric import ARI
from torch.nn import DataParallel
from torchvision import transforms as T
from matplotlib import pyplot as plt


def stack_images(img_stack: torch.tensor, fill=1, pad=1):
    N, C, H, W = img_stack.shape
    flat_img = torch.full((C, H+2*pad, (W+2*pad)*N), fill_value=fill).to(torch.float)
    for i in range(N):
        part_img = img_stack[i]
        flat_img[:, 1:H+1, 1 + i*(W+1): (i+1)*(W+1)] = part_img
    return flat_img


def image_to_gradient(image):
    N, C, H, W = image.shape
    image = image
    image = image - torch.min(image)
    image = image / torch.max(image)
    image = image.repeat(1, 3, 1, 1)
    high_value = torch.tensor([1., 1., 0.], device=image.device)[None, :, None, None].repeat(1, 1, H, W)
    low_value = torch.tensor([0., 0., 0.5], device=image.device)[None, :, None, None].repeat(1, 1, H, W)
    high_image = image * high_value
    low_image = (1-image) * low_value
    image = high_image + low_image
    return image


def plot_images(image: torch.tensor, gradient=False, invert=False):

    assert len(image.shape) == 4
    if gradient:
        image = image_to_gradient(image)
    image = stack_images(image, fill=1, pad=1)
    if invert:
        image = 1 - image
    return image


def stickbreaking_process(recons, pred_mask, truth_mask):
    recons = recons[None, ...].permute(0, 1, 3, 4, 2)
    pred_mask = pred_mask[None, ...].permute(0, 1, 3, 4, 2)
    truth_mask = truth_mask[None, ...].squeeze(2)
    B, K, H, W, CM = pred_mask.size()
    _, _, _, _, CR = recons.size()
    _, TK, _, _ = truth_mask.size()
    truth_mask = truth_mask[..., None]
    truth_mask_sum = truth_mask.sum(dim=2).sum(dim=2).squeeze(0).repeat(1, K) + 1e-12
    overlap = torch.einsum('bthwc,bphwc->tp', truth_mask, pred_mask)
    overlap = overlap / truth_mask_sum
    max_overlap = torch.max(overlap, dim=0)[1]

    pred_mask_embed = torch.zeros((B, TK, K, H, W, CM), device=truth_mask.device)
    for i in range(pred_mask.shape[1]):
        pred_mask_embed[:, max_overlap[i], i, :, :, :] = pred_mask[:, i]
    recons_embed = recons[:, None, :, :, :, :].repeat(1, TK, 1, 1, 1, 1)
    recons_embed = pred_mask_embed * recons_embed
    pred_mask = pred_mask_embed.sum(dim=2)
    recons = recons_embed.sum(dim=2)
    return recons[0].permute(0, 3, 1, 2), pred_mask[0].permute(0, 3, 1, 2)