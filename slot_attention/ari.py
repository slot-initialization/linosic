import torch
import numpy as np
from torch import arange as ar
from scipy.special import comb
import torchvision
from matplotlib import pyplot as plt


def compute_mask_ari(mask0, mask1):
    mask0 = mask0[:, None].byte()
    mask1 = mask1[None, :].byte()
    agree = mask0 & mask1
    table = agree.sum(dim=-1).sum(dim=-1)
    a = table.sum(axis=1)
    b = table.sum(axis=0)
    n = a.sum()
    comb_a = comb(a, 2).sum()
    comb_b = comb(b, 2).sum()
    comb_n = comb(n, 2)
    comb_table = comb(table, 2).sum()

    if (comb_b == comb_a == comb_n == comb_table):
        # the perfect case
        ari = 1.0
    else:
        ari = (
                (comb_table - comb_a * comb_b / comb_n) /
                (0.5 * (comb_a + comb_b) - (comb_a * comb_b) / comb_n)
        )

    return ari


def ari_calc(truth_mask, pred_mask):
    aris = 0
    B, K, H, W = pred_mask.size()
    _, TK, _, _ = truth_mask.size()

    #print(truth_mask.shape)
    #truth_mask_sum = truth_mask.sum(dim=1)[:, None, ...].repeat(1, K, 1, 1)
    #pred_mask = pred_mask * truth_mask_sum
    # print(truth_mask.shape, pred_mask.shape)

    #print(truth_mask_fg_sum)
    #print(truth_mask_fg_sum.shape)
    truth_mask_sum = truth_mask.sum(dim=2).sum(dim=2).squeeze(0)[..., None].repeat(1, K) + 1e-12
    #print(truth_mask_sum)
    overlap = torch.einsum('bthw,bphw->tp', truth_mask, pred_mask)
    #print(overlap)
    overlap = overlap / truth_mask_sum
    #print(overlap)
    max_overlap = torch.max(overlap, dim=0)[1]
    #print(max_overlap)

    pred_mask_embed = torch.zeros((B, TK, K, H, W), device=truth_mask.device)
    for i in range(pred_mask.shape[1]):
        pred_mask_embed[:, max_overlap[i], i, :, :] = pred_mask[:, i] * truth_mask[:, max_overlap[i]]
    pred_mask_embed[:, -1, :, :, :] = 0
    pred_mask = pred_mask_embed.sum(dim=2)
    truth_mask[:, -1, :, :] = 0
    #for i in range(TK):
    #   pred_mask_show = pred_mask[0, i, :, :].cpu().numpy()
    #   #print(truth_mask_fg_sum[0, i, :, :].sum())
    #   plt.imshow(pred_mask_show)
    #   plt.show()
    #   pred_mask_show = truth_mask[0, i, :, :].cpu().numpy()
    #   # print(truth_mask_fg_sum[0, i, :, :].sum())
    #   plt.imshow(pred_mask_show)
    #   plt.show()

    #print(torch.max(pred_mask), torch.min(pred_mask))
    max_index = torch.argmax(pred_mask, dim=1)
    pred_mask = torch.zeros_like(pred_mask, device=pred_mask.device)
    pred_mask[ar(B)[:, None, None], max_index, ar(H)[None, :, None], ar(W)[None, None, :]] = 1.0
    for b in range(B):
        this_ari = compute_mask_ari(truth_mask[b].detach().cpu(), pred_mask[b].detach().cpu())
        aris += this_ari
    aris = aris / B
    return aris


