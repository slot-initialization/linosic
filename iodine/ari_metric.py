import torch
import numpy as np
from torch import arange as ar
from scipy.special import comb
import torchvision
from matplotlib import pyplot as plt

class ARI:
    def compute_mask_ari(self, mask0, mask1):
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

    def ari_calc(self, truth_mask, pred_mask):
        aris = 0
        B, K, H, W = pred_mask.size()
        _, TK, _, _ = truth_mask.size()

        truth_mask_sum = truth_mask.sum(dim=2).sum(dim=2).squeeze(0)[..., None].repeat(1, K) + 1e-12
        overlap = torch.einsum('bthw,bphw->tp', truth_mask, pred_mask)
        overlap = overlap / truth_mask_sum
        max_overlap = torch.max(overlap, dim=0)[1]

        pred_mask_embed = torch.zeros((B, TK, K, H, W), device=truth_mask.device)
        for i in range(pred_mask.shape[1]):
            pred_mask_embed[:, max_overlap[i], i, :, :] = pred_mask[:, i] * truth_mask[:, max_overlap[i]]
        pred_mask_embed[:, -1, :, :, :] = 0
        pred_mask = pred_mask_embed.sum(dim=2)
        truth_mask[:, -1, :, :] = 0

        max_index = torch.argmax(pred_mask, dim=1)
        pred_mask = torch.zeros_like(pred_mask, device=pred_mask.device)
        pred_mask[ar(B)[:, None, None], max_index, ar(H)[None, :, None], ar(W)[None, None, :]] = 1.0
        for b in range(B):
            this_ari = self.compute_mask_ari(truth_mask[b].detach().cpu(), pred_mask[b].detach().cpu())
            aris += this_ari
        aris = aris / B
        return aris