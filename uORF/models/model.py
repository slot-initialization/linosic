import math
from .op import conv2d_gradfix
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
from torchvision.models import vgg16
from torch import autograd
import numpy as np


class Encoder(nn.Module):
    def __init__(self, input_nc=3, z_dim=64, bottom=False):

        super().__init__()

        self.bottom = bottom

        if self.bottom:
            self.enc_down_0 = nn.Sequential(nn.Conv2d(input_nc + 4, z_dim, 3, stride=1, padding=1),
                                            nn.ReLU(True))
        self.enc_down_1 = nn.Sequential(nn.Conv2d(z_dim if bottom else input_nc+4, z_dim, 3, stride=2 if bottom else 1, padding=1),
                                        nn.ReLU(True))
        self.enc_down_2 = nn.Sequential(nn.Conv2d(z_dim, z_dim, 3, stride=2, padding=1),
                                        nn.ReLU(True))
        self.enc_down_3 = nn.Sequential(nn.Conv2d(z_dim, z_dim, 3, stride=2, padding=1),
                                        nn.ReLU(True))
        self.enc_up_3 = nn.Sequential(nn.Conv2d(z_dim, z_dim, 3, stride=1, padding=1),
                                      nn.ReLU(True),
                                      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.enc_up_2 = nn.Sequential(nn.Conv2d(z_dim*2, z_dim, 3, stride=1, padding=1),
                                      nn.ReLU(True),
                                      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.enc_up_1 = nn.Sequential(nn.Conv2d(z_dim * 2, z_dim, 3, stride=1, padding=1),
                                      nn.ReLU(True))

    def forward(self, x):
        """
        input:
            x: input image, Bx3xHxW
        output:
            feature_map: BxCxHxW
        """
        W, H, B = x.shape[3], x.shape[2], x.shape[0]
        X = torch.linspace(-1, 1, W)
        Y = torch.linspace(-1, 1, H)
        y1_m, x1_m = torch.meshgrid([Y, X])
        x2_m, y2_m = 2 - x1_m, 2 - y1_m  # Normalized distance in the four direction
        pixel_emb = torch.stack([x1_m, x2_m, y1_m, y2_m]).to(x.device).repeat(B, 1, 1, 1)  # 1x4xHxW
        x_ = torch.cat([x, pixel_emb], dim=1)

        if self.bottom:
            x_down_0 = self.enc_down_0(x_)
            x_down_1 = self.enc_down_1(x_down_0)
        else:
            x_down_1 = self.enc_down_1(x_)
        x_down_2 = self.enc_down_2(x_down_1)
        x_down_3 = self.enc_down_3(x_down_2)
        x_up_3 = self.enc_up_3(x_down_3)
        x_up_2 = self.enc_up_2(torch.cat([x_up_3, x_down_2], dim=1))
        feature_map = self.enc_up_1(torch.cat([x_up_2, x_down_1], dim=1))  # BxCxHxW
        return feature_map


class Decoder(nn.Module):
    def __init__(self, n_freq=5, input_dim=33+64, z_dim=64, n_layers=3, locality=True, locality_ratio=4/7, fixed_locality=False):
        """
        freq: raised frequency
        input_dim: pos emb dim + slot dim
        z_dim: network latent dim
        n_layers: #layers before/after skip connection.
        locality: if True, for each obj slot, clamp sigma values to 0 outside obj_scale.
        locality_ratio: if locality, what value is the boundary to clamp?
        fixed_locality: if True, compute locality in world space instead of in transformed view space
        """
        super().__init__()
        self.n_freq = n_freq
        self.locality = locality
        self.locality_ratio = locality_ratio
        self.fixed_locality = fixed_locality
        self.out_ch = 4
        before_skip = [nn.Linear(input_dim, z_dim), nn.ReLU(True)]
        after_skip = [nn.Linear(z_dim+input_dim, z_dim), nn.ReLU(True)]
        for i in range(n_layers-1):
            before_skip.append(nn.Linear(z_dim, z_dim))
            before_skip.append(nn.ReLU(True))
            after_skip.append(nn.Linear(z_dim, z_dim))
            after_skip.append(nn.ReLU(True))
        self.f_before = nn.Sequential(*before_skip)
        self.f_after = nn.Sequential(*after_skip)
        self.f_after_latent = nn.Linear(z_dim, z_dim)
        self.f_after_shape = nn.Linear(z_dim, self.out_ch - 3)
        self.f_color = nn.Sequential(nn.Linear(z_dim, z_dim//4),
                                     nn.ReLU(True),
                                     nn.Linear(z_dim//4, 3))
        before_skip = [nn.Linear(input_dim, z_dim), nn.ReLU(True)]
        after_skip = [nn.Linear(z_dim + input_dim, z_dim), nn.ReLU(True)]
        for i in range(n_layers - 1):
            before_skip.append(nn.Linear(z_dim, z_dim))
            before_skip.append(nn.ReLU(True))
            after_skip.append(nn.Linear(z_dim, z_dim))
            after_skip.append(nn.ReLU(True))
        after_skip.append(nn.Linear(z_dim, self.out_ch))
        self.b_before = nn.Sequential(*before_skip)
        self.b_after = nn.Sequential(*after_skip)

    def forward(self, sampling_coor_bg, sampling_coor_fg, z_slots, fg_transform):
        """
        1. pos emb by Fourier
        2. for each slot, decode all points from coord and slot feature
        input:
            sampling_coor_bg: Px3, P = #points, typically P = NxDxHxW
            sampling_coor_fg: (K-1)xPx3
            z_slots: KxC, K: #slots, C: #feat_dim
            fg_transform: If self.fixed_locality, it is 1x4x4 matrix nss2cam0, otherwise it is 1x3x3 azimuth rotation of nss2cam0
        """
        B, K, C = z_slots.shape
        P = sampling_coor_bg.shape[1]

        if self.fixed_locality:
            outsider_idx = torch.any(sampling_coor_fg.abs() > self.locality_ratio, dim=-1)  # Bx(K-1)xP
            sampling_coor_fg = torch.cat([sampling_coor_fg, torch.ones_like(sampling_coor_fg[:, :, :, 0:1])], dim=-1)  # Bx(K-1)xPx4
            sampling_coor_fg = torch.matmul(fg_transform[:, None, ...], sampling_coor_fg[..., None])  # Bx(K-1)xPx4x1
            sampling_coor_fg = sampling_coor_fg.squeeze(-1)[:, :, :, :3]  # Bx(K-1)xPx3
        else:
            sampling_coor_fg = torch.matmul(fg_transform[:, None, ...], sampling_coor_fg[..., None])  # Bx(K-1)xPx3x1
            sampling_coor_fg = sampling_coor_fg.squeeze(-1)  # (K-1)xPx3
            outsider_idx = torch.any(sampling_coor_fg.abs() > self.locality_ratio, dim=-1)  # Bx(K-1)xP

        z_bg = z_slots[:, 0:1, :]  # Bx1xC
        z_fg = z_slots[:, 1:, :]  # Bx(K-1)xC
        query_bg = sin_emb(sampling_coor_bg, n_freq=self.n_freq)  # BxPx60, 60 means increased-freq feat dim
        input_bg = torch.cat([query_bg, z_bg.expand(-1, P, -1)], dim=2)  # BxPx(60+C)

        sampling_coor_fg_ = sampling_coor_fg.flatten(start_dim=1, end_dim=2)  # Bx((K-1)xP)x3
        query_fg_ex = sin_emb(sampling_coor_fg_, n_freq=self.n_freq)  # Bx((K-1)xP)x60
        z_fg_ex = z_fg[:, :, None, :].expand(-1, -1, P, -1).flatten(start_dim=1, end_dim=2)  # Bx((K-1)xP)xC
        input_fg = torch.cat([query_fg_ex, z_fg_ex], dim=2)  # Bx((K-1)xP)x(60+C)

        tmp = self.b_before(input_bg)
        bg_raws = self.b_after(torch.cat([input_bg, tmp], dim=2)).view([B, 1, P, self.out_ch])  # BxPx5 -> Bx1xPx5
        tmp = self.f_before(input_fg)
        tmp = self.f_after(torch.cat([input_fg, tmp], dim=2))  # Bx((K-1)xP)x64
        latent_fg = self.f_after_latent(tmp)  # Bx((K-1)xP)x64
        fg_raw_rgb = self.f_color(latent_fg).view([B, K-1, P, 3])  # Bx((K-1)xP)x3 -> Bx(K-1)xPx3
        fg_raw_shape = self.f_after_shape(tmp).view([B, K - 1, P])  # Bx((K-1)xP)x1 -> Bx(K-1)xP, density

        if self.locality:
            fg_raw_shape[outsider_idx] *= 0
        fg_raws = torch.cat([fg_raw_rgb, fg_raw_shape[..., None]], dim=-1)  # Bx(K-1)xPx4

        all_raws = torch.cat([bg_raws, fg_raws], dim=1)  # BxKxPx4
        raw_masks = F.relu(all_raws[:, :, :, -1:], True)  # BxKxPx1
        masks = raw_masks / (raw_masks.sum(dim=1) + 1e-5)[:, None, ...]  # BxKxPx1
        raw_rgb = (all_raws[:, :, :, :3].tanh() + 1) / 2
        raw_sigma = raw_masks

        unmasked_raws = torch.cat([raw_rgb, raw_sigma], dim=3)  # BxKxPx4

        masked_raws = unmasked_raws * masks
        #print(masked_raws)
        raws = masked_raws.sum(dim=1)

        return raws, masked_raws, unmasked_raws, masks


class SlotAttentionOwn(nn.Module):
    def __init__(self, num_slots, in_dim=64, slot_dim=64, iters=3, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.slot_dim = slot_dim
        self.n_clusters = 20
        self.Kmeans = KMeansPP(n_clusters=self.n_clusters, max_iter=100, return_lbl=True)
        self.norm_feat_cc = nn.LayerNorm(self.slot_dim)
        self.to_hash = nn.Sequential(nn.Linear(self.slot_dim, self.slot_dim),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(self.slot_dim, self.slot_dim//2),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(self.slot_dim//2, 1))
        #self.res_net_mlps = nn.ModuleList([nn.Sequential(nn.Linear(slot_dim*2, hidden_dim),
        #                                                 nn.ReLU(inplace=True),
        #                                                 nn.Linear(hidden_dim, slot_dim))
        #                                   for _ in range(self.num_slots)])
        self.gru_fg = nn.GRUCell(slot_dim, slot_dim)
        self.gru_bg = nn.GRUCell(slot_dim, slot_dim)

        hidden_dim = max(slot_dim, hidden_dim)

        self.to_res_fg = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, slot_dim)
        )
        self.to_res_bg = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, slot_dim)
        )
        # self.slot_norm = nn.LayerNorm([self.num_slots, self.slot_dim])
        # self.linear_norm = lambda max_v, min_v, x: torch.div(torch.add(min_v, x), max_v)*2 - 1
        self.eye = torch.eye(self.n_clusters, device='cuda')
        self.eye_slot = torch.eye(self.num_slots, device='cuda')
        # self.max_pool = nn.MaxPool1d(kernel_size=5, return_indices=True)

    def forward(self, feat, num_slots=None):
        B, N, D = feat.shape
        K = num_slots if num_slots is not None else self.num_slots
        slots_b, attn_b = [], []
        for feat_b in feat[:]:
            cluster_centers, cc_labels = self.Kmeans(feat_b)  # 10x64, 1x64*64
            feat_cc = self.norm_feat_cc(torch.cat((feat_b, cluster_centers), dim=0))
            feat_b, cluster_centers = feat_cc[:feat.shape[1], :], feat_cc[feat.shape[1]:, :]
            cc_labels = self.eye[cc_labels]  # 64*64x10
            #print(cc_labels)
            #print(cc_labels.shape)
            feat_labels = torch.einsum('nd,nl->ndl', feat_b, cc_labels)  # 64*64x64x10
            #print(feat_labels.shape)
            hashes = self.to_hash(cluster_centers).view(self.n_clusters)  # .softmax(dim=-1) + self.eps  # 10
            hashes = torch.sub(hashes, hashes.min())
            hashes = torch.div(hashes, hashes.max()) * 2 - 1  # 10
            # print(torch.round(hashes, decimals=2))
            slot_map = torch.zeros((self.n_clusters, self.num_slots), device=feat.device)  # 10x5
            for i in range(self.num_slots):
                j = i * 0.5 - 1
                # print(slot_map[:, i].shape, torch.square(torch.sub(hashes, j)).shape)
                slot_map[:, i] = -torch.square(torch.sub(hashes, j)) + 4
                #print(slot_map[:, i])

            slot_max, idx = torch.max(slot_map, dim=-1)
            print(idx)
            slot_map_idx = self.eye_slot[idx] + self.eps
            slot_map = torch.mul(slot_map, slot_map_idx)
            slot_map = torch.div(slot_map, slot_max[..., None])

            attn = torch.einsum('nl,ls->sn', cc_labels, slot_map)  # 5x64*64
            attn_weights = torch.div(attn, attn.sum(dim=-1, keepdim=True))
            feat_slots = torch.einsum('ndl,ls->nsd', feat_labels, slot_map)  # 64*64x5x64
            feat_slots = torch.einsum('nsd,sn->nsd', feat_slots, attn_weights)  # 64*64x5x64
            feat_slots = feat_slots.sum(dim=0)
            slot_map = torch.div(slot_map, slot_map.sum(dim=0))
            slots = torch.einsum('ld,ls->lsd', cluster_centers, slot_map)  # 10x5x64
            slots = slots.sum(dim=0)  # 5x64
            # slots = feat_slots[0]  # 5x64
            bg_update = feat_slots[0][None, ...]
            bg_slots = slots[0][None, ...]
            fg_update = feat_slots[1:]
            fg_slots = slots[1:]
            bg_slots = self.gru_bg(bg_update, bg_slots)  # 1x64
            fg_slots = self.gru_fg(fg_update, fg_slots)  # 4x64
            bg_slots = bg_slots + self.to_res_bg(bg_slots)
            fg_slots = fg_slots + self.to_res_fg(fg_slots)
            #print(bg_slots.shape, fg_slots.shape)
            slots = torch.cat((bg_slots, fg_slots), dim=0)

            slots_b.append(slots)
            attn_b.append(attn)
        slots_b, attn_b = torch.stack(slots_b, dim=0), torch.stack(attn_b, dim=0)  # Bx5x64 Bx5x64*64

        return slots_b, attn_b


class SlotAttentionRegular(nn.Module):
    def __init__(self, num_slots, in_dim=64, slot_dim=64, iters=3, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = slot_dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, slot_dim))
        init.xavier_uniform_(self.slots_logsigma)
        self.slots_mu_bg = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slots_logsigma_bg = nn.Parameter(torch.zeros(1, 1, slot_dim))
        init.xavier_uniform_(self.slots_logsigma_bg)

        self.to_k = nn.Linear(in_dim, slot_dim, bias=False)
        self.to_v = nn.Linear(in_dim, slot_dim, bias=False)
        self.to_q = nn.Sequential(nn.LayerNorm(slot_dim), nn.Linear(slot_dim, slot_dim, bias=False))
        self.to_q_bg = nn.Sequential(nn.LayerNorm(slot_dim), nn.Linear(slot_dim, slot_dim, bias=False))

        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.gru_bg = nn.GRUCell(slot_dim, slot_dim)

        hidden_dim = max(slot_dim, hidden_dim)

        self.to_res = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, slot_dim)
        )
        self.to_res_bg = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, slot_dim)
        )

        self.norm_feat = nn.LayerNorm(in_dim)
        self.slot_dim = slot_dim

    def forward(self, feat, num_slots=None):
        """
        input:
            feat: visual feature with position information, BxNxC
        output: slots: BxKxC, attn: BxKxN
        """
        B, _, _ = feat.shape
        K = num_slots if num_slots is not None else self.num_slots

        mu = self.slots_mu.expand(B, K-1, -1)
        sigma = self.slots_logsigma.exp().expand(B, K-1, -1)
        slot_fg = mu + sigma * torch.randn_like(mu)
        mu_bg = self.slots_mu_bg.expand(B, 1, -1)
        sigma_bg = self.slots_logsigma_bg.exp().expand(B, 1, -1)
        slot_bg = mu_bg + sigma_bg * torch.randn_like(mu_bg)

        feat = self.norm_feat(feat)
        k = self.to_k(feat)
        v = self.to_v(feat)

        attn = None
        for _ in range(self.iters):
            slot_prev_bg = slot_bg
            slot_prev_fg = slot_fg
            q_fg = self.to_q(slot_fg)
            q_bg = self.to_q_bg(slot_bg)

            dots_fg = torch.einsum('bid,bjd->bij', q_fg, k) * self.scale
            dots_bg = torch.einsum('bid,bjd->bij', q_bg, k) * self.scale
            dots = torch.cat([dots_bg, dots_fg], dim=1)  # BxKxN
            attn = dots.softmax(dim=1) + self.eps  # BxKxN
            attn_bg, attn_fg = attn[:, 0:1, :], attn[:, 1:, :]  # Bx1xN, Bx(K-1)xN
            attn_weights_bg = attn_bg / attn_bg.sum(dim=-1, keepdim=True)  # Bx1xN
            attn_weights_fg = attn_fg / attn_fg.sum(dim=-1, keepdim=True)  # Bx(K-1)xN

            updates_bg = torch.einsum('bjd,bij->bid', v, attn_weights_bg)
            updates_fg = torch.einsum('bjd,bij->bid', v, attn_weights_fg)

            slot_bg = self.gru_bg(
                updates_bg.reshape(-1, self.slot_dim),
                slot_prev_bg.reshape(-1, self.slot_dim)
            )
            slot_bg = slot_bg.reshape(B, -1, self.slot_dim)
            slot_bg = slot_bg + self.to_res_bg(slot_bg)

            slot_fg = self.gru(
                updates_fg.reshape(-1, self.slot_dim),
                slot_prev_fg.reshape(-1, self.slot_dim)
            )
            slot_fg = slot_fg.reshape(B, -1, self.slot_dim)
            slot_fg = slot_fg + self.to_res(slot_fg)

        slots = torch.cat([slot_bg, slot_fg], dim=1)
        return slots, attn


class SlotAttention(nn.Module):
    def __init__(self, opt, num_slots, num_cluster=10, in_dim=64, slot_dim=64, iters=3, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.num_slots = num_slots
        self.num_cluster = num_cluster
        self.iters = iters
        self.eps = eps
        self.scale = slot_dim ** -0.5

        self.to_k = nn.Linear(in_dim, slot_dim, bias=False)
        self.to_v = nn.Linear(in_dim, slot_dim, bias=False)
        self.to_q = nn.Sequential(nn.LayerNorm(slot_dim), nn.Linear(slot_dim, slot_dim, bias=False))
        self.to_q_bg = nn.Sequential(nn.LayerNorm(slot_dim), nn.Linear(slot_dim, slot_dim, bias=False))

        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.gru_bg = nn.GRUCell(slot_dim, slot_dim)

        hidden_dim = max(slot_dim, hidden_dim)

        self.to_res = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, slot_dim)
        )
        self.to_res_bg = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, slot_dim)
        )
# SlotInitPseudoWeightsOld
        self.norm_feat = nn.LayerNorm(in_dim)
        self.slot_dim = slot_dim
        if opt.slot_init == "kmeans":
            self.SlotInitModule = SlotInitKmeans(num_slots=self.num_slots, num_cluster=self.num_cluster, slot_dim=self.slot_dim)
        elif opt.slot_init == "PseudoWeights":
            self.SlotInitModule = SlotInitPseudoWeights(num_slots=self.num_slots, num_cluster=self.num_cluster, slot_dim=self.slot_dim)
        elif opt.slot_init == "PseudoWeightsOld":
            self.SlotInitModule = SlotInitPseudoWeightsOld(num_slots=self.num_slots, num_cluster=self.num_cluster, slot_dim=self.slot_dim)
        elif opt.slot_init == "direct":
            self.SlotInitModule = SlotInitDirect(num_slots=self.num_slots, num_cluster=self.num_cluster, slot_dim=self.slot_dim)
        elif opt.slot_init == "directms":
            self.SlotInitModule = SlotInitDirectMS(num_slots=self.num_slots, num_cluster=self.num_cluster, slot_dim=self.slot_dim)
        elif opt.slot_init == "mlp":
            self.SlotInitModule = SlotInitMlp(num_slots=self.num_slots, num_cluster=self.num_cluster, slot_dim=self.slot_dim)
        elif opt.slot_init == "mlpms":
            self.SlotInitModule = SlotInitMlpMS(num_slots=self.num_slots, num_cluster=self.num_cluster, slot_dim=self.slot_dim)
        elif opt.slot_init == "PseudoWeightsOldms":
            self.SlotInitModule = SlotInitPseudoWeightsOldMS(num_slots=self.num_slots, num_cluster=self.num_cluster, slot_dim=self.slot_dim)
        else:
            print("Error: Wrong slot_init argument, valid ones are kmeans, pooling, pixelattn, vectorattn, kmeansexp")

    def forward(self, feat, num_slots=None):
        """
        input:
            feat: visual feature with position information, BxNxC
        output: slots: BxKxC, attn: BxKxN
        """
        B, _, _ = feat.shape
        # .flatten(start_dim=1, end_dim=1)
        # feat = self.norm_feat(feat)
        slots = torch.stack([self.SlotInitModule(feat_b.clone()) for feat_b in feat[:]], dim=0)
        #print(slots.shape)

        feat_slots = self.norm_feat(torch.cat((feat, slots), dim=1))
        feat, slots = feat_slots[:, :feat.shape[1], :], feat_slots[:, feat.shape[1]:, :]
        # slot_bg, slot_fg = slots[:, 0, :][None, ...], slots[:, 1:, :]
        slot_bg, slot_fg = slots[:, 0:1, :], slots[:, 1:, :]
        # add this as argument to initmodule: feat.clone().flatten(start_dim=0, end_dim=1)
        k = self.to_k(feat)
        v = self.to_v(feat)
        attn = None
        for _ in range(self.iters):
            slot_prev_bg = slot_bg
            slot_prev_fg = slot_fg
            q_fg = self.to_q(slot_fg)
            q_bg = self.to_q_bg(slot_bg)

            dots_fg = torch.einsum('bid,bjd->bij', q_fg, k)**2 * self.scale
            dots_bg = torch.einsum('bid,bjd->bij', q_bg, k)**2 * self.scale
            dots = torch.cat([dots_bg, dots_fg], dim=1)  # BxKxN
            attn = dots.softmax(dim=1) + self.eps  # BxKxN
            attn_bg, attn_fg = attn[:, 0:1, :], attn[:, 1:, :]  # Bx1xN, Bx(K-1)xN
            attn_weights_bg = attn_bg / attn_bg.sum(dim=-1, keepdim=True)  # Bx1xN
            attn_weights_fg = attn_fg / attn_fg.sum(dim=-1, keepdim=True)  # Bx(K-1)xN

            updates_bg = torch.einsum('bjd,bij->bid', v, attn_weights_bg)
            updates_fg = torch.einsum('bjd,bij->bid', v, attn_weights_fg)

            slot_bg = self.gru_bg(
                updates_bg.reshape(-1, self.slot_dim),
                slot_prev_bg.reshape(-1, self.slot_dim)
            )

            slot_bg = slot_bg.reshape(B, -1, self.slot_dim)
            slot_bg = slot_bg + self.to_res_bg(slot_bg)

            slot_fg = self.gru(
                updates_fg.reshape(-1, self.slot_dim),
                slot_prev_fg.reshape(-1, self.slot_dim)
            )
            slot_fg = slot_fg.reshape(B, -1, self.slot_dim)
            slot_fg = slot_fg + self.to_res(slot_fg)

        slots = torch.cat([slot_bg, slot_fg], dim=1)
        return slots, attn


class KMeansPP(nn.Module):
    def __init__(self, n_clusters, max_iter=100, tol=0.0001, return_lbl=False, device=torch.device('cuda')):
        super().__init__()
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.return_lbl = return_lbl
        self.centroids = None
        self.lbl = None
        self.device = device

    def forward(self, X, centroids=None):
        self.centroids = self.centroids_init(X, centroids)
        for i in range(self.max_iter):
            centroid_added = False
            new_centroids, used_centroids = self.kmeans_step(X, self.centroids)
            centr_shift = self.calc_centr_shift(new_centroids, used_centroids)
            if new_centroids.shape[0] < self.n_clusters:
                self.centroids = self.centroids_init(X, new_centroids)
                centroid_added = True
            else:
                self.centroids = new_centroids
            if (centr_shift <= self.tol) and (not centroid_added):
                if self.return_lbl:
                    _, lbl = self.calc_dist_lbl(X, self.centroids)
                    return self.centroids, lbl
                return self.centroids
        if self.return_lbl:
            _, lbl = self.calc_dist_lbl(X, self.centroids)
            return self.centroids, lbl
        return self.centroids

    def kmeans_step(self, X, centroids):
        old_centroids = centroids
        _, lbl = self.calc_dist_lbl(X, old_centroids)
        lbl_mask, elem_per_lbl, used_lbls = self.create_lblmask_elemperlbl_usedlbl(lbl)
        x_rep = X.repeat(self.n_clusters, 1, 1)
        einsum = torch.einsum('abc,ab->abc', x_rep, lbl_mask)
        lbl_einsum_sum = torch.sum(einsum, dim=1)
        mean_sum = torch.divide(lbl_einsum_sum, elem_per_lbl)
        new_centroids = mean_sum[[~torch.any(mean_sum.isnan(), dim=1)]]
        used_centroids = old_centroids[[~torch.any(mean_sum.isnan(), dim=1)]]
        return new_centroids, used_centroids,

    def centroids_init(self, X, centroids):
        if centroids is None:
            # centroids = X[torch.randint(0, X.shape[0], (1,))]
            centroids = X[0: 1]
        while centroids.shape[0] < self.n_clusters:
            outlier_coor = self.calc_outlier_coor(X, centroids)
            outlier = X[outlier_coor, :][None, ...]
            centroids = torch.cat((centroids, outlier), dim=0)
        return centroids

    def calc_dist_lbl(self, X, centroids):
        sq_dist = torch.cdist(centroids, X, 2)
        min_sq_dist, lbl = torch.min(sq_dist, dim=0)
        return min_sq_dist, lbl

    def calc_outlier_coor(self, X, centroids):
        sq_dist, _ = self.calc_dist_lbl(X, centroids)
        argmax_dist = torch.argmax(sq_dist)
        return argmax_dist

    def create_lblmask_elemperlbl_usedlbl(self, lbl):
        used_lbls = torch.arange(self.n_clusters, device=self.device).view(self.n_clusters, 1)
        lbl_mask = used_lbls.repeat(1, lbl.shape[0])
        lbl_mask = torch.subtract(lbl_mask, lbl)
        lbl_mask = lbl_mask.eq(0)#.type(torch.int)
        elem_per_lbl = torch.sum(lbl_mask, dim=1).view(self.n_clusters, 1)
        return lbl_mask, elem_per_lbl, used_lbls

    def calc_centr_shift(self, centroids_1, centroids_2):
        shift = torch.subtract(centroids_1, centroids_2).abs().pow(2)
        shift = torch.sum(shift)
        return shift


def euclidean_distances(x, y):
    """ Computes pairwise distances

        @param x: a [n x d] torch.FloatTensor of datapoints
        @param y: a [m x d] torch.FloatTensor of datapoints

        @return: a [n x m] torch.FloatTensor of pairwise distances
    """
    return torch.norm(x.unsqueeze(1) - y.unsqueeze(0), dim=2)


def gaussian_kernel(x, y, sigma):
    """ Computes pairwise Gaussian kernel (without normalizing constant)
        (note this is kernel as defined in non-parametric statistics, not a kernel as in RKHS)

        @param x: a [n x d] torch.FloatTensor of datapoints
        @param y: a [m x d] torch.FloatTensor of datapoints
        @param sigma: Gaussian kernel bandwith.
                      Either a scalar, or a [1 x m] torch.FloatTensor of datapoints

        @return: a [n x m] torch.FloatTensor of pairwise kernel computations,
                 without normalizing constant
    """
    return torch.exp(-.5 / (sigma**2) * euclidean_distances(x, y)**2)


class GaussianMeanShift(nn.Module):
    def __init__(self, num_seeds=100, max_iters=10, epsilon=0.05,
                 sigma=1.0, subsample_factor=1, batch_size=None):
        super().__init__()
        self.num_seeds = num_seeds
        self.max_iters = max_iters
        self.epsilon = epsilon
        self.h = sigma
        self.batch_size = batch_size
        self.subsample_factor = subsample_factor  # Must be int
        self.distance = euclidean_distances
        self.kernel = gaussian_kernel

    def forward(self, X):
        """ Run mean-shift
            @param X: a [n x d] torch.FloatTensor of d-dim unit vectors
            @param Z: a [m x d] torch.FloatTensor of seeds to run mean shift from
        """
        Z = self.select_smart_seeds(X)
        Z = self.seed_hill_climbing(X, Z)

        # Connected components
        cluster_labels = self.connected_components(Z)  # n
        cc_embedding = torch.zeros((self.num_seeds, Z.shape[0], Z.shape[1]), device=cluster_labels.device)  # mxmxd
        cc_embedding[cluster_labels, torch.arange(start=0, end=Z.shape[0]), :] = 1
        # print(cc_embedding)
        cc_embedding_norm = cc_embedding.sum(dim=1) + 0.00000001
        cc_embedding = torch.mul(cc_embedding, Z).sum(dim=1)
        cc_embedding = torch.div(cc_embedding, cc_embedding_norm)
        return cc_embedding  # cluster_labels, Z

    def connected_components(self, Z):
        """ Compute simple connected components algorithm.
            @param Z: a [n x d] torch.FloatTensor of datapoints
            @return: a [n] torch.LongTensor of cluster labels
        """

        n, d = Z.shape
        K = 0

        # SAMPLING/GROUPING
        cluster_labels = torch.ones((n,), dtype=torch.long, device=Z.device) * -1
        for i in range(n):
            if cluster_labels[i] == -1:

                # Find all points close to it and label it the same
                distances = self.distance(Z, Z[i: i + 1])  # Shape: [n x 1]
                component_seeds = distances[:, 0] <= self.epsilon

                # If at least one component already has a label, then use the mode of the label
                if torch.unique(cluster_labels[component_seeds]).shape[0] > 1:
                    temp = cluster_labels[component_seeds]
                    temp = temp[temp != -1]
                    label = torch.mode(temp)[0]
                else:
                    label = torch.tensor(K)
                    K += 1  # Increment number of clusters
                cluster_labels[component_seeds] = label.to(Z.device)

        return cluster_labels
        # return torch.from_numpy(cluster_labels)

    def seed_hill_climbing(self, X, Z):
        """ Run mean shift hill climbing algorithm on the seeds.
            The seeds climb the distribution given by the KDE of X
            @param X: a [n x d] torch.FloatTensor of d-dim unit vectors
            @param Z: a [m x d] torch.FloatTensor of seeds to run mean shift from
        """

        n, d = X.shape
        m = Z.shape[0]

        for _iter in range(self.max_iters):

            # Create a new object for Z
            new_Z = Z.clone()

            # Compute the update in batches
            for i in range(0, m, self.batch_size):
                W = self.kernel(Z[i: i + self.batch_size], X, self.h)  # Shape: [batch_size x n]
                Q = W / W.sum(dim=1, keepdim=True)  # Shape: [batch_size x n]
                new_Z[i: i + self.batch_size] = torch.mm(Q, X)

            Z = new_Z

        return Z

    def select_smart_seeds(self, X):
        """ Randomly select seeds that far away
            @param X: a [n x d] torch.FloatTensor of d-dim unit vectors
            @return: a [num_seeds x d] matrix of seeds
        """
        n, d = X.shape

        selected_indices = -1 * torch.ones(self.num_seeds, dtype=torch.long)

        # Initialize seeds matrix
        seeds = torch.empty((self.num_seeds, d), device=X.device)
        num_chosen_seeds = 0

        # Keep track of distances
        distances = torch.empty((n, self.num_seeds), device=X.device)

        # Select first seed
        selected_seed_index = np.random.randint(0, n)
        selected_indices[0] = selected_seed_index
        selected_seed = X[selected_seed_index, :]
        seeds[0, :] = selected_seed
        #print(torch.min(X), torch.max(X))
        distances[:, 0] = self.distance(X, selected_seed.unsqueeze(0))[:, 0]
        num_chosen_seeds += 1

        # Select rest of seeds
        for i in range(num_chosen_seeds, min(self.num_seeds, n)):

            # Find the point that has the furthest distance from the nearest seed
            distance_to_nearest_seed = torch.min(distances[:, :i], dim=1)[0]  # Shape: [n]
            #print(torch.min(distance_to_nearest_seed))
            # selected_seed_index = torch.argmax(distance_to_nearest_seed)
            selected_seed_index = torch.multinomial(distance_to_nearest_seed, 1)
            selected_indices[i] = selected_seed_index
            selected_seed = torch.index_select(X, 0, selected_seed_index)[0, :]
            seeds[i, :] = selected_seed

            # Calculate distance to this selected seed
            distances[:, i] = self.distance(X, selected_seed.unsqueeze(0))[:, 0]

        return seeds

    def mean_shift_smart_init(self, X, sigmas=None):
        """ Run mean shift with carefully selected seeds
            @param X: a [n x d] torch.FloatTensor of d-dim unit vectors
            @param sigmas: a [n] torch.FLoatTensor of values for per-datapoint sigmas
                           If None, use pre-specified value of sigma for all datapoints
            @return: a [n] array of cluster labels
        """
        subsampled_X = X[::self.subsample_factor, ...]  # Shape: [n//subsample_factor x d]
        if sigmas is not None:
            subsampled_sigmas = sigmas[::self.subsample_factor]  # Shape: [n//subsample_factor]
            self.h = subsampled_sigmas.unsqueeze(0)  # Shape: [1 x n//subsample_factor]

        # Get the seeds and subsampled points
        seeds = self.select_smart_seeds(subsampled_X)

        # Run mean shift
        seed_cluster_labels, updated_seeds = self.mean_shift_with_seeds(subsampled_X, seeds)

        # Get distances to updated seeds
        distances = self.distance(X, updated_seeds)

        # Get clusters by assigning point to closest seed
        closest_seed_indices = torch.argmin(distances, dim=1)  # Shape: [n]
        cluster_labels = seed_cluster_labels[closest_seed_indices]

        # Save cluster centers and labels
        uniq_labels = torch.unique(seed_cluster_labels)
        uniq_cluster_centers = torch.zeros((uniq_labels.shape[0], updated_seeds.shape[1]), dtype=torch.float, device=updated_seeds.device)
        for i, label in enumerate(uniq_labels):
            uniq_cluster_centers[i, :] = updated_seeds[seed_cluster_labels == i, :].mean(dim=0)
        self.uniq_cluster_centers = uniq_cluster_centers
        self.uniq_labels = uniq_labels

        return cluster_labels.to(X.device)  # Put it back on the device


class SlotInitKmeans(nn.Module):
    def __init__(self, num_slots, num_cluster=10, slot_dim=64):
        super().__init__()
        self.num_slots = num_slots
        self.num_cluster = num_cluster
        self.slot_dim = slot_dim
        self.KMeans = KMeansPP(10, 100, 0.0001, False)
        self.mapping_to_slots = nn.Sequential(
            nn.Linear(self.num_cluster * slot_dim, (self.num_cluster + self.num_slots) // 2 * slot_dim),
            nn.ReLU(inplace=True),
            nn.Linear((self.num_cluster + self.num_slots) // 2 * slot_dim, self.num_slots * slot_dim),
        )

    def forward(self, feat):
        #print(feat.shape)
        cluster_centers = self.KMeans(feat)
        distance_refpoint = torch.ones((1, self.slot_dim), device=cluster_centers.device)
        distances = torch.einsum('cd,ad->c', cluster_centers, distance_refpoint)
        idx = torch.argsort(distances)
        cluster_centers = cluster_centers[idx]
        cluster_centers = cluster_centers.flatten()
        slots = self.mapping_to_slots(cluster_centers)
        slots = slots.view((self.num_slots, self.slot_dim))
        return slots


class SlotInitKmeansExp(nn.Module):
    def __init__(self, num_slots, num_cluster=10, slot_dim=64):
        super().__init__()
        self.num_slots = num_slots
        self.num_cluster = num_cluster
        self.slot_dim = slot_dim
        self.KMeans = KMeansPP(10, 100, 0.0001, False)
        self.mapping_to_slots = nn.ModuleList([nn.Sequential(
                                                    nn.Linear(self.num_cluster*slot_dim, self.num_cluster//2*slot_dim),
                                                    nn.ReLU(inplace=True),
                                                    nn.Linear(self.num_cluster//2*slot_dim, slot_dim))
                                               for _ in range(self.num_slots)])

    def forward(self, feat):
        cluster_centers = self.KMeans(feat)
        distance_refpoint = torch.ones((1, self.slot_dim), device=cluster_centers.device)
        distances = torch.einsum('cd,ad->c', cluster_centers, distance_refpoint)
        idx = torch.argsort(distances)
        cluster_centers = cluster_centers[idx]
        cluster_centers = cluster_centers.flatten()
        slots = torch.zeros((self.num_slots, self.slot_dim), device=feat.device)
        for i in range(len(self.mapping_to_slots)):
            slots[i] = self.mapping_to_slots[i](cluster_centers)
        slots = slots.view((self.num_slots, self.slot_dim))
        return slots


class SlotInitPseudoWeights(nn.Module):
    def __init__(self, num_slots, num_cluster=10, slot_dim=64, eps=1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.num_cluster = num_cluster
        self.slot_dim = slot_dim
        self.eps = eps
        self.scale = slot_dim ** -0.5
        self.KMeans = KMeansPP(10, 100, 0.0001, False)
        self.context_net = nn.Sequential(nn.Linear(2, self.slot_dim),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(self.slot_dim, self.slot_dim))

        self.norm = nn.LayerNorm(self.slot_dim)

        self.pseudo_weights = nn.Sequential(nn.Linear(self.slot_dim * 2, self.slot_dim * 2),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(self.slot_dim * 2, self.slot_dim * 2),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(self.slot_dim * 2, self.slot_dim),
                                            nn.Sigmoid())
        self.out_linear = nn.Sequential(nn.Linear(self.slot_dim, self.slot_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(self.slot_dim, self.slot_dim))

    def forward(self, feat):
        cluster_centers = self.KMeans(feat)  # 10x64
        cc_norm = self.norm(cluster_centers)  # 10x64

        context_v = torch.arange(start=0, end=self.num_slots, step=1, device=feat.device) * (
                    2 * (torch.acos(torch.zeros(1)).item() * 2) / self.num_slots)
        context_v = torch.stack((torch.sin(context_v), torch.cos(context_v)), dim=0).permute(1, 0)  # 5x2

        context = self.context_net(context_v)  # 5x64
        context = context[None, :, :].repeat(self.num_cluster, 1, 1)  # 10x5x64

        cc_norm = cc_norm[:, None, :].repeat(1, self.num_slots, 1)  # 10x5x64

        pw_input = torch.cat((context, cc_norm), dim=-1)  # 10x5x128
        pseudo_weights = self.pseudo_weights(pw_input)  # 10x5x64

        cc_weighted = torch.mul(pseudo_weights, cluster_centers[:, None, :])  # 10x5x64

        slots = cc_weighted.sum(dim=0)  # / self.num_cluster  # 5x64
        slots = self.out_linear(slots)
        return slots


class SlotInitPseudoWeightsOld(nn.Module):
    def __init__(self, num_slots, num_cluster=10, slot_dim=64, eps=1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.num_cluster = num_cluster
        self.slot_dim = slot_dim
        self.eps = eps
        self.scale = slot_dim ** -0.5
        self.KMeans = KMeansPP(10, 100, 0.0001, False)
        self.context_net = nn.Sequential(nn.Linear(2, self.slot_dim),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(self.slot_dim, self.slot_dim))

        self.norm = nn.LayerNorm(self.slot_dim)

        self.pseudo_weights = nn.Sequential(nn.Linear(self.slot_dim * 2, self.slot_dim * 2),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(self.slot_dim * 2, self.slot_dim * 2),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(self.slot_dim * 2, self.slot_dim),
                                            nn.Sigmoid())

    def forward(self, feat):
        cluster_centers = self.KMeans(feat)  # 10x64
        cc_norm = self.norm(cluster_centers)  # 10x64

        context_v = torch.arange(start=0, end=self.num_slots, step=1, device=feat.device) * (2 * (torch.acos(torch.zeros(1)).item() * 2) / self.num_slots)
        context_v = torch.stack((torch.sin(context_v), torch.cos(context_v)), dim=0).permute(1, 0)  # 5x2

        context = self.context_net(context_v)  # 5x64
        context = context[None, :, :].repeat(self.num_cluster, 1, 1)  # 10x5x64

        cc_norm = cc_norm[:, None, :].repeat(1, self.num_slots, 1)  # 10x5x64

        pw_input = torch.cat((context, cc_norm), dim=-1)  # 10x5x128
        pseudo_weights = self.pseudo_weights(pw_input)  # 10x5x64

        cc_weighted = torch.mul(pseudo_weights, cluster_centers[:, None, :])  # 10x5x64

        slots = cc_weighted.sum(dim=0)  # / self.num_cluster  # 5x64
        return slots


class SlotInitDirect(nn.Module):
    def __init__(self, num_slots, num_cluster=10, slot_dim=64, eps=1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.num_cluster = num_slots
        self.slot_dim = slot_dim
        self.eps = eps
        self.scale = slot_dim ** -0.5
        self.KMeans = KMeansPP(self.num_slots, 100, 0.0001, False)

    def forward(self, feat):
        cluster_centers = self.KMeans(feat)  # 10x64
        return cluster_centers


class SlotInitMlp(nn.Module):
    def __init__(self, num_slots, num_cluster=10, slot_dim=64, eps=1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.num_cluster = num_slots
        self.slot_dim = slot_dim
        self.eps = eps
        self.scale = slot_dim ** -0.5
        self.KMeans = KMeansPP(self.num_cluster, 100, 0.0001, False)
        self.mlp = nn.Sequential(nn.Linear(slot_dim, slot_dim),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(slot_dim, slot_dim))

    def forward(self, feat):
        cluster_centers = self.KMeans(feat)  # 10x64
        slots = self.mlp(cluster_centers)
        return slots


class SlotInitDirectMS(nn.Module):
    def __init__(self, num_slots, num_cluster=10, slot_dim=64, eps=1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.num_cluster = num_slots
        self.slot_dim = slot_dim
        self.eps = eps
        self.scale = slot_dim ** -0.5
        self.KMeans = GaussianMeanShift(num_seeds=10, max_iters=10, epsilon=0.01/2, sigma=0.01*66/2, batch_size=1)

    def forward(self, feat):
        cluster_centers = self.KMeans(feat)  # 10x64
        return cluster_centers


class SlotInitMlpMS(nn.Module):
    def __init__(self, num_slots, num_cluster=10, slot_dim=64, eps=1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.num_cluster = num_slots
        self.slot_dim = slot_dim
        self.eps = eps
        self.scale = slot_dim ** -0.5
        self.KMeans = GaussianMeanShift(num_seeds=7, max_iters=10, epsilon=0.01/2, sigma=0.01*66/2, batch_size=1)
        self.mlp = nn.Sequential(nn.Linear(slot_dim, slot_dim),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(slot_dim, slot_dim))

    def forward(self, feat):
        cluster_centers = self.KMeans(feat)  # 10x64
        slots = self.mlp(cluster_centers)
        return slots


class SlotInitPseudoWeightsOldMS(nn.Module):
    def __init__(self, num_slots, num_cluster=10, slot_dim=64, eps=1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.num_cluster = num_slots
        self.slot_dim = slot_dim
        self.eps = eps
        self.scale = slot_dim ** -0.5
        self.KMeans = GaussianMeanShift(num_seeds=self.num_slots, max_iters=10, epsilon=0.01/2, sigma=0.01*66/2, batch_size=1)
        self.context_net = nn.Sequential(nn.Linear(2, self.slot_dim),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(self.slot_dim, self.slot_dim))

        self.norm = nn.LayerNorm(self.slot_dim)

        self.pseudo_weights = nn.Sequential(nn.Linear(self.slot_dim * 2, self.slot_dim * 2),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(self.slot_dim * 2, self.slot_dim * 2),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(self.slot_dim * 2, self.slot_dim),
                                            nn.Sigmoid())

    def forward(self, feat):
        cluster_centers = self.KMeans(feat)  # 10x64
        cc_norm = self.norm(cluster_centers)  # 10x64

        context_v = torch.arange(start=0, end=self.num_slots, step=1, device=feat.device) * (2 * (torch.acos(torch.zeros(1)).item() * 2) / self.num_slots)
        context_v = torch.stack((torch.sin(context_v), torch.cos(context_v)), dim=0).permute(1, 0)  # 5x2

        context = self.context_net(context_v)  # 5x64
        context = context[None, :, :].repeat(self.num_cluster, 1, 1)  # 10x5x64

        cc_norm = cc_norm[:, None, :].repeat(1, self.num_slots, 1)  # 10x5x64
        pw_input = torch.cat((context, cc_norm), dim=-1)  # 10x5x128
        pseudo_weights = self.pseudo_weights(pw_input)  # 10x5x64

        cc_weighted = torch.mul(pseudo_weights, cluster_centers[:, None, :])  # 10x5x64

        slots = cc_weighted.sum(dim=0)  # / self.num_cluster  # 5x64
        return slots


def sin_emb(x, n_freq=5, keep_ori=True):
    """
    create sin embedding for 3d coordinates
    input:
        x: Px3
        n_freq: number of raised frequency
    """
    embedded = []
    if keep_ori:
        embedded.append(x)
    emb_fns = [torch.sin, torch.cos]
    freqs = 2. ** torch.linspace(0., n_freq - 1, steps=n_freq)
    for freq in freqs:
        for emb_fn in emb_fns:
            embedded.append(emb_fn(freq * x))
    embedded_ = torch.cat(embedded, dim=2)
    return embedded_


def raw2outputs(raw, z_vals, rays_d, render_mask=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray in cam coor.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda x, y: 1. - torch.exp(-x * y)
    device = raw.device

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.tensor([1e-2], device=device).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = raw[..., :3]
    alpha = raw2alpha(raw[..., 3], dists)  # [N_rays, N_samples]
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], alpha.shape[1], 1), device=device), 1. - alpha + 1e-10], -1), -1)[:, :, :-1]

    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]
    #print(rgb_map.sum(), weights.sum())
    weights_norm = weights.detach() + 1e-5
    weights_norm /= weights_norm.sum(dim=-1, keepdim=True)
    depth_map = torch.sum(weights_norm * z_vals, -1)
    if render_mask:
        density = raw[..., 3]  # [N_rays, N_samples]
        mask_map = torch.sum(weights * density, dim=2)  # [N_rays,]
        #print(weights.sum())
        return rgb_map, depth_map, weights_norm, mask_map

    return rgb_map, depth_map, weights_norm


def get_perceptual_net(layer=4):
    assert layer > 0
    idx_set = [None, 4, 9, 16, 23, 30]
    idx = idx_set[layer]
    vgg = vgg16(pretrained=True)
    loss_network = nn.Sequential(*list(vgg.features)[:idx]).eval()
    for param in loss_network.parameters():
        param.requires_grad = False
    return loss_network


def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)

def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean(), fake_loss.mean()

def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty

def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k

class EqualConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = conv2d_gradfix.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )


class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = F.leaky_relu(out, 0.2, inplace=True) * 1.4

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )



class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
        stride=1,
        padding=1
    ):
        layers = []

        if downsample:
            layers.append(nn.AvgPool2d(kernel_size=2, stride=2))

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3, stride=1, padding=1)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True, stride=1, padding=1)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=True, activate=False, bias=False, stride=1, padding=0
        )

    def forward(self, input):
        out = self.conv1(input) * 1.4
        out = self.conv2(out) * 1.4

        skip = self.skip(input) * 1.4
        out = (out + skip) / math.sqrt(2)

        return out


class Discriminator(nn.Module):
    def __init__(self, size, ndf, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        channels = {
            4: ndf*2,
            8: ndf*2,
            16: ndf,
            32: ndf,
            64: ndf//2,
            128: ndf//2
        }

        convs = [ConvLayer(3, channels[size], 1, stride=1, padding=1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3, stride=1, padding=1)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input):
        out = self.convs(input) * 1.4

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out) * 1.4

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out

