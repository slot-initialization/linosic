import math
from .op import conv2d_gradfix
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
from torchvision.models import vgg16
from torch import autograd


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
            #slot_map = torch.sub(1, slot_map)
            #slot_map = torch.mul(-5, slot_map)
            #slot_map = torch.exp(slot_map)
            slot_max, idx = torch.max(slot_map, dim=-1)
            print(idx)
            slot_map_idx = self.eye_slot[idx] + self.eps
            slot_map = torch.mul(slot_map, slot_map_idx)
            slot_map = torch.div(slot_map, slot_max[..., None])
            #print(slot_map)
            #print(slot_map.shape)
            attn = torch.einsum('nl,ls->sn', cc_labels, slot_map)  # 5x64*64
            attn_weights = torch.div(attn, attn.sum(dim=-1, keepdim=True))
            #print(attn.shape)
            feat_slots = torch.einsum('ndl,ls->nsd', feat_labels, slot_map)  # 64*64x5x64
            feat_slots = torch.einsum('nsd,sn->nsd', feat_slots, attn_weights)  # 64*64x5x64
            #print(feat_slots.shape)
            feat_slots = feat_slots.sum(dim=0)
            #print(feat_slots.shape)
            slot_map = torch.div(slot_map, slot_map.sum(dim=0))
            slots = torch.einsum('ld,ls->lsd', cluster_centers, slot_map)  # 10x5x64
            slots = slots.sum(dim=0)  # 5x64
            # slots = feat_slots[0]  # 5x64
            #print(slots.shape)
            #for j in range(self.num_slots):
                # update = torch.cat((slots[j], current_feat_slots[j]), dim=0)
                # update = self.res_net_mlps[j](update)
                #slots[j] = torch.add(slots[j], update)
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
            #print(slots.shape)
            #slots[0] = slots[0] + self.to_res_bg(slots[0])
            #slots[1:] = slots[1:] + self.to_res_fg(slots[1:])
            slots_b.append(slots)
            attn_b.append(attn)
        slots_b, attn_b = torch.stack(slots_b, dim=0), torch.stack(attn_b, dim=0)  # Bx5x64 Bx5x64*64
        #print(slots_b.shape)
        #print(attn_b.shape)
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
# SlotInitPseudoWeights
        self.norm_feat = nn.LayerNorm(in_dim)
        self.slot_dim = slot_dim
        if opt.slot_init == "kmeans":
            self.SlotInitModule = SlotInitKmeans(num_slots=self.num_slots, num_cluster=self.num_cluster, slot_dim=self.slot_dim)
        elif opt.slot_init == "pooling":
            self.SlotInitModule = SlotInitPooling(num_slots=self.num_slots, num_cluster=self.num_cluster, slot_dim=self.slot_dim, eps=self.eps)
        elif opt.slot_init == "pixelattn":
            self.SlotInitModule = SlotInitPixelAttention(num_slots=self.num_slots, num_cluster=self.num_cluster, slot_dim=self.slot_dim, eps=self.eps)
        elif opt.slot_init == "vectorattn":
            self.SlotInitModule = SlotInitVectorAttention(num_slots=self.num_slots, num_cluster=self.num_cluster, slot_dim=self.slot_dim, eps=self.eps)
        elif opt.slot_init == "kmeansexp":
            self.SlotInitModule = SlotInitKmeansExp(num_slots=self.num_slots, num_cluster=self.num_cluster, slot_dim=self.slot_dim)
        elif opt.slot_init == "hash":
            self.SlotInitModule = SlotInitHash(num_slots=self.num_slots, num_cluster=self.num_cluster, slot_dim=self.slot_dim)
        elif opt.slot_init == "HashMax":
            self.SlotInitModule = SlotInitHashMax(num_slots=self.num_slots, num_cluster=self.num_cluster, slot_dim=self.slot_dim)
        elif opt.slot_init == "HashSum":
            self.SlotInitModule = SlotInitHashSum(num_slots=self.num_slots, num_cluster=self.num_cluster, slot_dim=self.slot_dim)
        elif opt.slot_init == "SA5CC":
            self.SlotInitModule = SlotInitSA5CC(num_slots=self.num_slots, num_cluster=self.num_cluster, slot_dim=self.slot_dim)
        elif opt.slot_init == "SAMLPMax":
            self.SlotInitModule = SlotInitSAMLPMax(num_slots=self.num_slots, num_cluster=self.num_cluster, slot_dim=self.slot_dim)
        elif opt.slot_init == "SAMLPSum":
            self.SlotInitModule = SlotInitSAMLPSum(num_slots=self.num_slots, num_cluster=self.num_cluster, slot_dim=self.slot_dim)
        elif opt.slot_init == "SAMLPMaxRes":
            self.SlotInitModule = SlotInitSAMLPMaxRes(num_slots=self.num_slots, num_cluster=self.num_cluster, slot_dim=self.slot_dim)
        elif opt.slot_init == "SAMLPSumRes":
            self.SlotInitModule = SlotInitSAMLPSumRes(num_slots=self.num_slots, num_cluster=self.num_cluster, slot_dim=self.slot_dim)
        elif opt.slot_init == "SAPoolingMax":
            self.SlotInitModule = SlotInitSAPoolingMax(num_slots=self.num_slots, num_cluster=self.num_cluster, slot_dim=self.slot_dim)
        elif opt.slot_init == "SAPoolingSum":
            self.SlotInitModule = SlotInitSAPoolingSum(num_slots=self.num_slots, num_cluster=self.num_cluster, slot_dim=self.slot_dim)
        elif opt.slot_init == "SAHash":
            self.SlotInitModule = SlotInitSAHash(num_slots=self.num_slots, num_cluster=self.num_cluster, slot_dim=self.slot_dim)
        elif opt.slot_init == "SAHashMax":
            self.SlotInitModule = SlotInitSAHashMax(num_slots=self.num_slots, num_cluster=self.num_cluster, slot_dim=self.slot_dim)
        elif opt.slot_init == "SAHashMaxReLU":
            self.SlotInitModule = SlotInitSAHashMaxReLU(num_slots=self.num_slots, num_cluster=self.num_cluster, slot_dim=self.slot_dim)
        elif opt.slot_init == "SAHashSum":
            self.SlotInitModule = SlotInitSAHashSum(num_slots=self.num_slots, num_cluster=self.num_cluster, slot_dim=self.slot_dim)
        elif opt.slot_init == "PoolingResMaxPermi":
            self.SlotInitModule = SlotInitPoolingResMaxPermi(num_slots=self.num_slots, num_cluster=self.num_cluster, slot_dim=self.slot_dim)
        elif opt.slot_init == "Context":
            self.SlotInitModule = SlotInitContext(num_slots=self.num_slots, num_cluster=self.num_cluster, slot_dim=self.slot_dim)
        elif opt.slot_init == "PseudoWeights":
            self.SlotInitModule = SlotInitPseudoWeights(num_slots=self.num_slots, num_cluster=self.num_cluster, slot_dim=self.slot_dim)
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


class SlotInitHash(nn.Module):
    def __init__(self, num_slots, num_cluster=10, slot_dim=64, eps=1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.num_cluster = num_cluster
        self.slot_dim = slot_dim
        self.eps = eps
        self.KMeans = KMeansPP(10, 100, 0.0001, False)
        self.cc_to_hash = nn.Sequential(nn.Linear(self.slot_dim, self.slot_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(self.slot_dim, self.slot_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(self.slot_dim, self.num_slots))

    def forward(self, feat):
        cluster_centers = self.KMeans(feat)  # 10x64
        hash_arr = self.cc_to_hash(cluster_centers)  # 10x5
        hash_arr = hash_arr.softmax(dim=0)
        _, max_ind = torch.max(hash_arr, dim=-1)  # 1
        eye = torch.eye(self.num_slots, device=feat.device)  # 5x5
        max_mask = torch.add(eye[max_ind, :], self.eps)  # 10x5
        masked_hash_arr = torch.mul(hash_arr, max_mask)  # 10x5
        slots = torch.mul(cluster_centers[:, None, :], masked_hash_arr[..., None])  # 10x5x64
        slots = slots.sum(dim=0)  # 5x64
        return slots


class SlotInitHashMax(nn.Module):  # Slot init with hashing and maxpooling
    def __init__(self, num_slots, num_cluster=10, slot_dim=64, eps=1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.num_cluster = num_cluster
        self.slot_dim = slot_dim
        self.eps = eps
        self.KMeans = KMeansPP(10, 100, 0.0001, False)
        self.cc_to_hash = nn.Sequential(nn.Linear(self.slot_dim, self.slot_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(self.slot_dim, self.slot_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(self.slot_dim, self.num_slots),
                                        nn.Sigmoid())
        self.max_pool = nn.MaxPool1d(kernel_size=10, return_indices=True)

    def forward(self, feat):
        cluster_centers = self.KMeans(feat)  # 10x64
        hashed_cc = self.cc_to_hash(cluster_centers)  # 10x5
        # max_hcc, _ = torch.max(hashed_cc, dim=1)  # 10
        # hashed_cc = torch.div(hashed_cc, max_hcc[..., None])  # 10x5
        hashed_cc = torch.sub(1, hashed_cc)
        hashed_cc = torch.mul(-5, hashed_cc)
        hashed_cc = torch.exp(hashed_cc)
        cc_attn, idx = self.max_pool(hashed_cc.permute(1, 0))  # 5x1, 5x1
        chosen_cc = cluster_centers[idx.squeeze(-1), :]  # 5x64
        slots = torch.mul(chosen_cc, cc_attn)
        return slots


class SlotInitHashSum(nn.Module):  # Slot init with hashing and sum
    def __init__(self, num_slots, num_cluster=10, slot_dim=64, eps=1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.num_cluster = num_cluster
        self.slot_dim = slot_dim
        self.eps = eps
        self.KMeans = KMeansPP(10, 100, 0.0001, False)
        self.cc_to_hash = nn.Sequential(nn.Linear(self.slot_dim, self.slot_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(self.slot_dim, self.slot_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(self.slot_dim, self.num_slots),
                                        nn.Sigmoid())

    def forward(self, feat):
        cluster_centers = self.KMeans(feat)  # 10x64
        hashed_cc = self.cc_to_hash(cluster_centers)  # 10x5
        hashed_cc = torch.sub(1, hashed_cc)
        hashed_cc = torch.mul(-5, hashed_cc)
        hashed_cc = torch.exp(hashed_cc)
        sum_hcc = hashed_cc.sum(dim=0)[None, ...]  # 1x5
        hashed_cc = torch.div(hashed_cc, sum_hcc)  # 10x5
        cc_hash_tensor = torch.einsum('cd,cs->csd', cluster_centers, hashed_cc)  # 10x5x64
        slots = cc_hash_tensor.sum(dim=0)  # 5x64
        return slots


class SlotInitSA5CC(nn.Module):  # SlotInitSelfAttn with 5 CC only
    def __init__(self, num_slots, num_cluster=10, slot_dim=64, eps=1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.num_cluster = num_cluster
        self.slot_dim = slot_dim
        self.eps = eps
        self.iter = 3
        self.softmax_temp = 1 / math.sqrt(self.slot_dim)
        self.KMeans = KMeansPP(5, 100, 0.0001, False)
        self.hash_dim = 8
        self.to_k = nn.Linear(self.slot_dim, self.slot_dim)
        self.to_q = nn.Linear(self.slot_dim, self.slot_dim)
        self.to_v = nn.Linear(self.slot_dim, self.slot_dim)
        self.to_k_norm = nn.LayerNorm(64)
        self.cc_norm = nn.LayerNorm(64)
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(nn.LayerNorm(slot_dim),
                                 nn.Linear(slot_dim, 2*slot_dim),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(2*slot_dim, slot_dim))

    def forward(self, feat):
        cluster_centers = self.KMeans(feat)  # 10x64
        cc_to_kv = self.to_k_norm(cluster_centers)
        k = self.to_k(cc_to_kv)  # 10x64
        v = self.to_v(cc_to_kv)  # 10x64

        for _ in range(self.iter):
            cc_prev = cluster_centers
            cluster_centers = self.cc_norm(cluster_centers)
            attn = torch.einsum("nd,md->nm", k, self.to_q(cluster_centers))**2 * self.softmax_temp
            attn = attn.softmax(dim=0) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)
            updates = torch.einsum("na,ad->nd", attn, v)
            cluster_centers = self.gru(updates, cc_prev)
            cluster_centers = cluster_centers + self.mlp(cluster_centers)

        return cluster_centers


class SlotInitSAMLPMax(nn.Module):  # SlotInitSelfAttn with MLP and MaxPooling
    def __init__(self, num_slots, num_cluster=10, slot_dim=64, eps=1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.num_cluster = num_cluster
        self.slot_dim = slot_dim
        self.eps = eps
        self.iter = 3
        self.softmax_temp = 1 / math.sqrt(self.slot_dim)
        self.KMeans = KMeansPP(10, 100, 0.0001, False)
        self.hash_dim = 8
        self.to_k = nn.Linear(self.slot_dim, self.slot_dim)
        self.to_q = nn.Linear(self.slot_dim, self.slot_dim)
        self.to_v = nn.Linear(self.slot_dim, self.slot_dim)
        self.to_k_norm = nn.LayerNorm(64)
        self.cc_norm = nn.LayerNorm(64)
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(nn.LayerNorm(slot_dim),
                                 nn.Linear(slot_dim, 2 * slot_dim),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(2 * slot_dim, slot_dim))
        self.slot_mapping = nn.ModuleList([nn.Sequential(nn.Linear(self.slot_dim, self.slot_dim),
                                                         nn.ReLU(inplace=True),
                                                         nn.Linear(self.slot_dim, self.slot_dim))
                                           for _ in range(self.num_slots)])
        self.max_pool = nn.MaxPool1d(kernel_size=10)

    def forward(self, feat):
        cluster_centers = self.KMeans(feat)  # 10x64
        cc_to_kv = self.to_k_norm(cluster_centers)
        k = self.to_k(cc_to_kv)  # 10x64
        v = self.to_v(cc_to_kv)  # 10x64

        for _ in range(self.iter):
            cc_prev = cluster_centers
            cluster_centers = self.cc_norm(cluster_centers)
            attn = torch.einsum("nd,md->nm", k, self.to_q(cluster_centers))**2 * self.softmax_temp
            attn = attn.softmax(dim=0) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)
            updates = torch.einsum("na,ad->nd", attn, v)
            cluster_centers = self.gru(updates, cc_prev)
            cluster_centers = cluster_centers + self.mlp(cluster_centers)
        slots = []
        for i in range(self.num_slots):
            slot_raw = self.slot_mapping[i](cluster_centers).permute(1, 0)  # 64x10
            slot = self.max_pool(slot_raw).permute(1, 0)  # 1x64
            slots.append(slot)
        slots = torch.stack(slots, dim=0).squeeze(1)
        return slots


class SlotInitSAMLPMaxRes(nn.Module):  # SlotInitSelfAttn with MLP, Residual and MaxPooling
    def __init__(self, num_slots, num_cluster=10, slot_dim=64, eps=1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.num_cluster = num_cluster
        self.slot_dim = slot_dim
        self.eps = eps
        self.iter = 3
        self.softmax_temp = 1 / math.sqrt(self.slot_dim)
        self.KMeans = KMeansPP(10, 100, 0.0001, False)
        self.hash_dim = 8
        self.to_k = nn.Linear(self.slot_dim, self.slot_dim)
        self.to_q = nn.Linear(self.slot_dim, self.slot_dim)
        self.to_v = nn.Linear(self.slot_dim, self.slot_dim)
        self.to_k_norm = nn.LayerNorm(64)
        self.cc_norm = nn.LayerNorm(64)
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(nn.LayerNorm(slot_dim),
                                 nn.Linear(slot_dim, 2 * slot_dim),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(2 * slot_dim, slot_dim))
        self.slot_mapping = nn.ModuleList([nn.Sequential(nn.Linear(self.slot_dim, self.slot_dim),
                                                         nn.ReLU(inplace=True),
                                                         nn.Linear(self.slot_dim, self.slot_dim),
                                                         nn.Sigmoid())
                                           for _ in range(self.num_slots)])
        self.max_pool = nn.MaxPool1d(kernel_size=10)

    def forward(self, feat):
        cluster_centers = self.KMeans(feat)  # 10x64
        cc_to_kv = self.to_k_norm(cluster_centers)
        k = self.to_k(cc_to_kv)  # 10x64
        v = self.to_v(cc_to_kv)  # 10x64

        for _ in range(self.iter):
            cc_prev = cluster_centers
            cluster_centers = self.cc_norm(cluster_centers)
            attn = torch.einsum("nd,md->nm", k, self.to_q(cluster_centers))**2 * self.softmax_temp
            attn = attn.softmax(dim=0) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)
            updates = torch.einsum("na,ad->nd", attn, v)
            cluster_centers = self.gru(updates, cc_prev)
            cluster_centers = cluster_centers + self.mlp(cluster_centers)
        slots = []
        for i in range(self.num_slots):
            cc_res = self.slot_mapping[i](cluster_centers)  # 64x10
            slot_raw = torch.mul(cc_res, cluster_centers).permute(1, 0)  # 10x64
            slot = self.max_pool(slot_raw).permute(1, 0)  # 1x64
            slots.append(slot)
        slots = torch.stack(slots, dim=0).squeeze(1)
        return slots


class SlotInitSAMLPSum(nn.Module):  # SlotInitSelfAttn with MLP and Sum
    def __init__(self, num_slots, num_cluster=10, slot_dim=64, eps=1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.num_cluster = num_cluster
        self.slot_dim = slot_dim
        self.eps = eps
        self.iter = 3
        self.softmax_temp = 1 / math.sqrt(self.slot_dim)
        self.KMeans = KMeansPP(10, 100, 0.0001, False)
        self.hash_dim = 8
        self.to_k = nn.Linear(self.slot_dim, self.slot_dim)
        self.to_q = nn.Linear(self.slot_dim, self.slot_dim)
        self.to_v = nn.Linear(self.slot_dim, self.slot_dim)
        self.to_k_norm = nn.LayerNorm(64)
        self.cc_norm = nn.LayerNorm(64)
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(nn.LayerNorm(slot_dim),
                                 nn.Linear(slot_dim, 2 * slot_dim),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(2 * slot_dim, slot_dim))
        self.slot_mapping = nn.ModuleList([nn.Sequential(nn.Linear(self.slot_dim, self.slot_dim),
                                                         nn.ReLU(inplace=True),
                                                         nn.Linear(self.slot_dim, self.slot_dim))
                                           for _ in range(self.num_slots)])

    def forward(self, feat):
        cluster_centers = self.KMeans(feat)  # 10x64
        cc_to_kv = self.to_k_norm(cluster_centers)
        k = self.to_k(cc_to_kv)  # 10x64
        v = self.to_v(cc_to_kv)  # 10x64

        for _ in range(self.iter):
            cc_prev = cluster_centers
            cluster_centers = self.cc_norm(cluster_centers)
            attn = torch.einsum("nd,md->nm", k, self.to_q(cluster_centers))**2 * self.softmax_temp
            attn = attn.softmax(dim=0) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)
            updates = torch.einsum("na,ad->nd", attn, v)
            cluster_centers = self.gru(updates, cc_prev)
            cluster_centers = cluster_centers + self.mlp(cluster_centers)
        slots = []
        for i in range(self.num_slots):
            slot_raw = self.slot_mapping[i](cluster_centers)  # 10x64
            slot = slot_raw.sum(dim=0)  # 64
            slots.append(slot)
        slots = torch.stack(slots, dim=0)  # 5x64
        return slots


class SlotInitSAMLPSumRes(nn.Module):  # SlotInitSelfAttn with MLP, Residual and Sum
    def __init__(self, num_slots, num_cluster=10, slot_dim=64, eps=1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.num_cluster = num_cluster
        self.slot_dim = slot_dim
        self.eps = eps
        self.iter = 3
        self.softmax_temp = 1 / math.sqrt(self.slot_dim)
        self.KMeans = KMeansPP(10, 100, 0.0001, False)
        self.hash_dim = 8
        self.to_k = nn.Linear(self.slot_dim, self.slot_dim)
        self.to_q = nn.Linear(self.slot_dim, self.slot_dim)
        self.to_v = nn.Linear(self.slot_dim, self.slot_dim)
        self.to_k_norm = nn.LayerNorm(64)
        self.cc_norm = nn.LayerNorm(64)
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(nn.LayerNorm(slot_dim),
                                 nn.Linear(slot_dim, 2 * slot_dim),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(2 * slot_dim, slot_dim))
        self.slot_mapping = nn.ModuleList([nn.Sequential(nn.Linear(self.slot_dim, self.slot_dim),
                                                         nn.ReLU(inplace=True),
                                                         nn.Linear(self.slot_dim, self.slot_dim))
                                           for _ in range(self.num_slots)])

    def forward(self, feat):
        cluster_centers = self.KMeans(feat)  # 10x64
        cc_to_kv = self.to_k_norm(cluster_centers)
        k = self.to_k(cc_to_kv)  # 10x64
        v = self.to_v(cc_to_kv)  # 10x64

        for _ in range(self.iter):
            cc_prev = cluster_centers
            cluster_centers = self.cc_norm(cluster_centers)
            attn = torch.einsum("nd,md->nm", k, self.to_q(cluster_centers))**2 * self.softmax_temp
            attn = attn.softmax(dim=0) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)
            updates = torch.einsum("na,ad->nd", attn, v)
            cluster_centers = self.gru(updates, cc_prev)
            cluster_centers = cluster_centers + self.mlp(cluster_centers)
        slots = []
        for i in range(self.num_slots):
            cc_res = self.slot_mapping[i](cluster_centers)  # 10x64
            slot_raw = torch.mul(cc_res, cluster_centers)  # 10x64
            slot = slot_raw.sum(dim=0)  # 64
            slots.append(slot)
        slots = torch.stack(slots, dim=0)  # 5x64
        return slots


class SlotInitSAPoolingMax(nn.Module):  # SlotInitSelfAttn with Residual Convolution and MaxPooling
    def __init__(self, num_slots, num_cluster=10, slot_dim=64, eps=1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.num_cluster = num_cluster
        self.slot_dim = slot_dim
        self.eps = eps
        self.iter = 3
        self.softmax_temp = 1 / math.sqrt(self.slot_dim)
        self.KMeans = KMeansPP(10, 100, 0.0001, False)
        self.hash_dim = 8
        self.to_k = nn.Linear(self.slot_dim, self.slot_dim)
        self.to_q = nn.Linear(self.slot_dim, self.slot_dim)
        self.to_v = nn.Linear(self.slot_dim, self.slot_dim)
        self.to_k_norm = nn.LayerNorm(64)
        self.cc_norm = nn.LayerNorm(64)
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(nn.LayerNorm(slot_dim),
                                 nn.Linear(slot_dim, 2 * slot_dim),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(2 * slot_dim, slot_dim))
        self.cc_separation = nn.Sequential(
                                            nn.Conv2d(1, self.num_slots, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
                                            nn.ReLU(inplace=True),
                                            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),  # 32
                                            nn.Conv2d(self.num_slots, self.num_slots, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
                                            nn.ReLU(inplace=True),
                                            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),  # 16
                                            nn.Conv2d(self.num_slots, self.num_slots, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
                                            nn.ReLU(inplace=True),
                                            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),  # 8
                                            nn.Conv2d(self.num_slots, self.num_slots, kernel_size=(1, int(self.slot_dim / 8)), stride=(1, 1),
                                                      padding=(0, 0)),
                                            nn.ReLU(inplace=True),
                                            )
        self.max_pool = nn.MaxPool1d(kernel_size=10)

    def forward(self, feat):
        cluster_centers = self.KMeans(feat)  # 10x64
        cc_to_kv = self.to_k_norm(cluster_centers)
        k = self.to_k(cc_to_kv)  # 10x64
        v = self.to_v(cc_to_kv)  # 10x64
        for _ in range(self.iter):
            cc_prev = cluster_centers
            cluster_centers = self.cc_norm(cluster_centers)
            attn = torch.einsum("nd,md->nm", k, self.to_q(cluster_centers))**2 * self.softmax_temp
            attn = attn.softmax(dim=0) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)
            updates = torch.einsum("na,ad->nd", attn, v)
            cluster_centers = self.gru(updates, cc_prev)
            cluster_centers = cluster_centers + self.mlp(cluster_centers)
        sep_cc = self.cc_separation(cluster_centers[None, None, ...])
        sep_cc = sep_cc.softmax(dim=1) + self.eps
        importance_sep_cc = torch.einsum('cz,sc->scz', cluster_centers.squeeze(), sep_cc.squeeze()).permute(0, 2, 1)  # [5, 10, 64]  --> [5, 64, 10]
        slots = self.max_pool(importance_sep_cc).squeeze(-1)  # 5x64
        return slots


class SlotInitSAPoolingSum(nn.Module):  # SlotInitSelfAttn with Residual Convolution and Sum
    def __init__(self, num_slots, num_cluster=10, slot_dim=64, eps=1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.num_cluster = num_cluster
        self.slot_dim = slot_dim
        self.eps = eps
        self.iter = 3
        self.softmax_temp = 1 / math.sqrt(self.slot_dim)
        self.KMeans = KMeansPP(10, 100, 0.0001, False)
        self.hash_dim = 8
        self.to_k = nn.Linear(self.slot_dim, self.slot_dim)
        self.to_q = nn.Linear(self.slot_dim, self.slot_dim)
        self.to_v = nn.Linear(self.slot_dim, self.slot_dim)
        self.to_k_norm = nn.LayerNorm(64)
        self.cc_norm = nn.LayerNorm(64)
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(nn.LayerNorm(slot_dim),
                                 nn.Linear(slot_dim, 2 * slot_dim),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(2 * slot_dim, slot_dim))
        self.cc_separation = nn.Sequential(
                                            nn.Conv2d(1, self.num_slots, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
                                            nn.ReLU(inplace=True),
                                            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),  # 32
                                            nn.Conv2d(self.num_slots, self.num_slots, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
                                            nn.ReLU(inplace=True),
                                            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),  # 16
                                            nn.Conv2d(self.num_slots, self.num_slots, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
                                            nn.ReLU(inplace=True),
                                            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),  # 8
                                            nn.Conv2d(self.num_slots, self.num_slots, kernel_size=(1, int(self.slot_dim / 8)), stride=(1, 1),
                                                      padding=(0, 0)),
                                            nn.ReLU(inplace=True),
                                            )

    def forward(self, feat):
        cluster_centers = self.KMeans(feat)  # 10x64
        cc_to_kv = self.to_k_norm(cluster_centers)
        k = self.to_k(cc_to_kv)  # 10x64
        v = self.to_v(cc_to_kv)  # 10x64
        for _ in range(self.iter):
            cc_prev = cluster_centers
            cluster_centers = self.cc_norm(cluster_centers)
            attn = torch.einsum("nd,md->nm", k, self.to_q(cluster_centers))**2 * self.softmax_temp
            attn = attn.softmax(dim=0) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)
            updates = torch.einsum("na,ad->nd", attn, v)
            cluster_centers = self.gru(updates, cc_prev)
            cluster_centers = cluster_centers + self.mlp(cluster_centers)
        sep_cc = self.cc_separation(cluster_centers[None, None, ...])
        sep_cc = sep_cc.softmax(dim=1) + self.eps
        importance_sep_cc = torch.einsum('cz,sc->scz', cluster_centers.squeeze(), sep_cc.squeeze())  # [5, 10, 64]
        slots = importance_sep_cc.sum(dim=1)  # 5x64
        return slots


class SlotInitSAHash(nn.Module):  # SlotInitSelfAttn with hashing to 5 slots
    def __init__(self, num_slots, num_cluster=10, slot_dim=64, eps=1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.num_cluster = num_cluster
        self.slot_dim = slot_dim
        self.eps = eps
        self.iter = 3
        self.softmax_temp = 1 / math.sqrt(self.slot_dim)
        self.KMeans = KMeansPP(10, 100, 0.0001, False)
        self.hash_dim = 8
        self.to_k = nn.Linear(self.slot_dim, self.slot_dim)
        self.to_q = nn.Linear(self.slot_dim, self.slot_dim)
        self.to_v = nn.Linear(self.slot_dim, self.slot_dim)
        self.to_k_norm = nn.LayerNorm(64)
        self.cc_norm = nn.LayerNorm(64)
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(nn.LayerNorm(slot_dim),
                                 nn.Linear(slot_dim, 2 * slot_dim),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(2 * slot_dim, slot_dim))
        self.cc_to_hash = nn.Sequential(nn.Linear(self.slot_dim, self.slot_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(self.slot_dim, self.slot_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(self.slot_dim, self.num_slots))

    def forward(self, feat):
        cluster_centers = self.KMeans(feat)  # 10x64
        cc_to_kv = self.to_k_norm(cluster_centers)
        k = self.to_k(cc_to_kv)  # 10x64
        v = self.to_v(cc_to_kv)  # 10x64

        for _ in range(self.iter):
            cc_prev = cluster_centers
            cluster_centers = self.cc_norm(cluster_centers)
            attn = torch.einsum("nd,md->nm", k, self.to_q(cluster_centers))**2 * self.softmax_temp
            attn = attn.softmax(dim=0) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)
            updates = torch.einsum("na,ad->nd", attn, v)
            cluster_centers = self.gru(updates, cc_prev)
            cluster_centers = cluster_centers + self.mlp(cluster_centers)
        hash_arr = self.cc_to_hash(cluster_centers)  # 10x5
        hash_arr = hash_arr.softmax(dim=0)
        _, max_ind = torch.max(hash_arr, dim=-1)  # 1
        eye = torch.eye(self.num_slots, device=feat.device)  # 5x5
        max_mask = torch.add(eye[max_ind, :], self.eps)  # 10x5
        masked_hash_arr = torch.mul(hash_arr, max_mask)  # 10x5
        slots = torch.mul(cluster_centers[:, None, :], masked_hash_arr[..., None])  # 10x5x64
        slots = slots.sum(dim=0)  # 5x64
        return slots


class SlotInitSAHashMax(nn.Module):  # SlotInitSelfAttn with hashing, sigmoid and max pool to 5 slots
    def __init__(self, num_slots, num_cluster=10, slot_dim=64, eps=1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.num_cluster = num_cluster
        self.slot_dim = slot_dim
        self.eps = eps
        self.iter = 3
        self.softmax_temp = 1 / math.sqrt(self.slot_dim)
        self.KMeans = KMeansPP(10, 100, 0.0001, False)
        self.hash_dim = 8
        self.to_k = nn.Linear(self.slot_dim, self.slot_dim)
        self.to_q = nn.Linear(self.slot_dim, self.slot_dim)
        self.to_v = nn.Linear(self.slot_dim, self.slot_dim)
        self.to_k_norm = nn.LayerNorm(64)
        self.cc_norm = nn.LayerNorm(64)
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(nn.LayerNorm(slot_dim),
                                 nn.Linear(slot_dim, 2 * slot_dim),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(2 * slot_dim, slot_dim))
        self.cc_to_hash = nn.Sequential(nn.Linear(self.slot_dim, self.slot_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(self.slot_dim, self.slot_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(self.slot_dim, self.num_slots),
                                        nn.Sigmoid())
        self.max_pool = nn.MaxPool1d(kernel_size=10, return_indices=True)

    def forward(self, feat):
        cluster_centers = self.KMeans(feat)  # 10x64
        cc_to_kv = self.to_k_norm(cluster_centers)
        k = self.to_k(cc_to_kv)  # 10x64
        v = self.to_v(cc_to_kv)  # 10x64

        for _ in range(self.iter):
            cc_prev = cluster_centers
            cluster_centers = self.cc_norm(cluster_centers)
            attn = torch.einsum("nd,md->nm", k, self.to_q(cluster_centers))**2 * self.softmax_temp
            attn = attn.softmax(dim=0) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)
            updates = torch.einsum("na,ad->nd", attn, v)
            cluster_centers = self.gru(updates, cc_prev)
            cluster_centers = cluster_centers + self.mlp(cluster_centers)
        hashed_cc = self.cc_to_hash(cluster_centers)  # 10x5
        hashed_cc = torch.sub(1, hashed_cc)
        hashed_cc = torch.mul(-5, hashed_cc)
        hashed_cc = torch.exp(hashed_cc)
        cc_attn, idx = self.max_pool(hashed_cc.permute(1, 0))  # 5x1, 5x1
        chosen_cc = cluster_centers[idx.squeeze(-1), :]  # 5x64
        slots = torch.mul(chosen_cc, cc_attn)
        return slots


class SlotInitSAHashMaxReLU(nn.Module):  # SlotInitSelfAttn with hashing, ReLU and max pool to 5 slots
    def __init__(self, num_slots, num_cluster=10, slot_dim=64, eps=1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.num_cluster = num_cluster
        self.slot_dim = slot_dim
        self.eps = eps
        self.iter = 3
        self.softmax_temp = 1 / math.sqrt(self.slot_dim)
        self.KMeans = KMeansPP(10, 100, 0.0001, False)
        self.hash_dim = 8
        self.to_k = nn.Linear(self.slot_dim, self.slot_dim)
        self.to_q = nn.Linear(self.slot_dim, self.slot_dim)
        self.to_v = nn.Linear(self.slot_dim, self.slot_dim)
        self.to_k_norm = nn.LayerNorm(64)
        self.cc_norm = nn.LayerNorm(64)
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(nn.LayerNorm(slot_dim),
                                 nn.Linear(slot_dim, 2 * slot_dim),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(2 * slot_dim, slot_dim))
        self.cc_to_hash = nn.Sequential(nn.Linear(self.slot_dim, self.slot_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(self.slot_dim, self.slot_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(self.slot_dim, self.num_slots),
                                        nn.ReLU(inplace=True))
        self.max_pool = nn.MaxPool1d(kernel_size=10, return_indices=True)

    def forward(self, feat):
        cluster_centers = self.KMeans(feat)  # 10x64
        cc_to_kv = self.to_k_norm(cluster_centers)
        k = self.to_k(cc_to_kv)  # 10x64
        v = self.to_v(cc_to_kv)  # 10x64

        for _ in range(self.iter):
            cc_prev = cluster_centers
            cluster_centers = self.cc_norm(cluster_centers)
            attn = torch.einsum("nd,md->nm", k, self.to_q(cluster_centers))**2 * self.softmax_temp
            attn = attn.softmax(dim=0) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)
            updates = torch.einsum("na,ad->nd", attn, v)
            cluster_centers = self.gru(updates, cc_prev)
            cluster_centers = cluster_centers + self.mlp(cluster_centers)
        hashed_cc = self.cc_to_hash(cluster_centers)  # 10x5
        max_hcc, _ = torch.max(hashed_cc, dim=1)  # 10
        hashed_cc = torch.div(hashed_cc, max_hcc[..., None])  # 10x5
        hashed_cc = torch.sub(1, hashed_cc)
        hashed_cc = torch.mul(-5, hashed_cc)
        hashed_cc = torch.exp(hashed_cc)
        cc_attn, idx = self.max_pool(hashed_cc.permute(1, 0))  # 5x1, 5x1
        chosen_cc = cluster_centers[idx.squeeze(-1), :]  # 5x64
        slots = torch.mul(chosen_cc, cc_attn)
        return slots


class SlotInitSAHashSum(nn.Module):  # SlotInitSelfAttn with hashing and sum to 5 slots
    def __init__(self, num_slots, num_cluster=10, slot_dim=64, eps=1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.num_cluster = num_cluster
        self.slot_dim = slot_dim
        self.eps = eps
        self.iter = 3
        self.softmax_temp = 1 / math.sqrt(self.slot_dim)
        self.KMeans = KMeansPP(10, 100, 0.0001, False)
        self.hash_dim = 8
        self.to_k = nn.Linear(self.slot_dim, self.slot_dim)
        self.to_q = nn.Linear(self.slot_dim, self.slot_dim)
        self.to_v = nn.Linear(self.slot_dim, self.slot_dim)
        self.to_k_norm = nn.LayerNorm(64)
        self.cc_norm = nn.LayerNorm(64)
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(nn.LayerNorm(slot_dim),
                                 nn.Linear(slot_dim, 2 * slot_dim),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(2 * slot_dim, slot_dim))
        self.cc_to_hash = nn.Sequential(nn.Linear(self.slot_dim, self.slot_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(self.slot_dim, self.slot_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(self.slot_dim, self.num_slots),
                                        nn.Sigmoid())

    def forward(self, feat):
        cluster_centers = self.KMeans(feat)  # 10x64
        cc_to_kv = self.to_k_norm(cluster_centers)
        k = self.to_k(cc_to_kv)  # 10x64
        v = self.to_v(cc_to_kv)  # 10x64

        for _ in range(self.iter):
            cc_prev = cluster_centers
            cluster_centers = self.cc_norm(cluster_centers)
            attn = torch.einsum("nd,md->nm", k, self.to_q(cluster_centers))**2 * self.softmax_temp
            attn = attn.softmax(dim=0) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)
            updates = torch.einsum("na,ad->nd", attn, v)
            cluster_centers = self.gru(updates, cc_prev)
            cluster_centers = cluster_centers + self.mlp(cluster_centers)
        hashed_cc = self.cc_to_hash(cluster_centers)  # 10x5
        hashed_cc = torch.sub(1, hashed_cc)
        hashed_cc = torch.mul(-5, hashed_cc)
        hashed_cc = torch.exp(hashed_cc)
        sum_hcc = hashed_cc.sum(dim=0)[None, ...]  # 1x5
        hashed_cc = torch.div(hashed_cc, sum_hcc)  # 10x5
        cc_hash_tensor = torch.einsum('cd,cs->csd', cluster_centers, hashed_cc)  # 10x5x64
        slots = cc_hash_tensor.sum(dim=0)  # 5x64
        return slots


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


class SlotInitPooling(nn.Module):
    def __init__(self, num_slots, num_cluster=10, slot_dim=64, eps=1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.num_cluster = num_cluster
        self.slot_dim = slot_dim
        self.eps = eps
        self.KMeans = KMeansPP(10, 100, 0.0001, False)
        self.cc_separation = nn.Sequential(nn.Conv2d(1, self.num_slots, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
                                           nn.ReLU(inplace=True),
                                           nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),  # 32
                                           nn.Conv2d(self.num_slots, self.num_slots, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
                                           nn.ReLU(inplace=True),
                                           nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),  # 16
                                           nn.Conv2d(self.num_slots, self.num_slots, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
                                           nn.ReLU(inplace=True),
                                           nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),  # 8
                                           nn.Conv2d(self.num_slots, self.num_slots, kernel_size=(1, int(self.slot_dim/8)), stride=(1, 1), padding=(0, 0)),
                                           nn.ReLU(inplace=True),
                                          )
        self.lin_lat = 4
        self.sep_reduction = nn.ModuleList([nn.Sequential(nn.Linear(self.num_cluster * self.slot_dim, self.lin_lat * self.slot_dim),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(self.lin_lat * self.slot_dim, self.slot_dim)) for _ in range(self.num_slots)])
        #self.slot_decision = nn.MaxPool2d((1, 2), 1)

    def forward(self, feat):
        cluster_centers = self.KMeans(feat)[None, None, ...]
        #print(cluster_centers.shape)
        sep_cc = self.cc_separation(cluster_centers)
        sep_cc = sep_cc.softmax(dim=1) + self.eps
        importance_sep_cc = torch.einsum('cz,sc->scz', cluster_centers.squeeze(), sep_cc.squeeze()).flatten(start_dim=1)  # [5, 10, 64] --> [5, 10*64]
        slots = torch.zeros((5, self.slot_dim), device=sep_cc.device)
        for i in range(len(self.sep_reduction)):
            slots[i, :] = self.sep_reduction[i](importance_sep_cc[i, ...])
        return slots


class SlotInitPoolingRes(nn.Module):
    def __init__(self, num_slots, num_cluster=10, slot_dim=64, eps=1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.num_cluster = num_cluster
        self.slot_dim = slot_dim
        self.eps = eps
        self.KMeans = KMeansPP(10, 100, 0.0001, False)
        self.cc_separation = nn.Sequential(nn.Conv2d(1, self.num_slots, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
                                           nn.ReLU(inplace=True),
                                           nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),  # 32
                                           nn.Conv2d(self.num_slots, self.num_slots, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
                                           nn.ReLU(inplace=True),
                                           nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),  # 16
                                           nn.Conv2d(self.num_slots, self.num_slots, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
                                           nn.ReLU(inplace=True),
                                           nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),  # 8
                                           nn.Conv2d(self.num_slots, self.num_slots, kernel_size=(1, int(self.slot_dim/8)), stride=(1, 1), padding=(0, 0)),
                                           nn.ReLU(inplace=True),
                                           )
        self.lin_lat = 4
        self.sep_reduction = nn.ModuleList([nn.Linear(2*self.slot_dim, slot_dim) for _ in range(self.num_slots)])

    def forward(self, feat):
        cluster_centers = self.KMeans(feat)[None, None, ...]
        # print(cluster_centers.shape)
        sep_cc = self.cc_separation(cluster_centers)
        sep_cc = sep_cc.softmax(dim=1) + self.eps
        importance_sep_cc = torch.einsum('cz,sc->scz', cluster_centers.squeeze(), sep_cc.squeeze())  # [5, 10, 64]
        # slots = torch.zeros((5, self.slot_dim), device=sep_cc.device)
        slots = importance_sep_cc[:, 0, :].squeeze(dim=1)  # 5x64
        for j in range(self.num_cluster-1):
            cc_part2 = importance_sep_cc[:, j+1, :].squeeze(dim=1)  # 5x64
            slots = torch.cat((slots, cc_part2), dim=-1)  # 5x2*64
            for i in range(self.num_slots):
                slots[i, :] = self.sep_reduction[i](slots[i, ...])
        return slots


class SlotInitPoolingResMaxPermi(nn.Module):
    def __init__(self, num_slots, num_cluster=10, slot_dim=64, eps=1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.num_cluster = num_cluster
        self.slot_dim = slot_dim
        self.eps = eps
        self.KMeans = KMeansPP(10, 100, 0.0001, False)
        self.cc_separation = nn.Sequential(nn.Conv2d(1, self.num_slots, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
                                           nn.ReLU(inplace=True),
                                           nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),  # 32
                                           nn.Conv2d(self.num_slots, self.num_slots, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
                                           nn.ReLU(inplace=True),
                                           nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),  # 16
                                           nn.Conv2d(self.num_slots, self.num_slots, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
                                           nn.ReLU(inplace=True),
                                           nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),  # 8
                                           nn.Conv2d(self.num_slots, self.num_slots, kernel_size=(1, int(self.slot_dim/8)), stride=(1, 1), padding=(0, 0)),
                                           nn.ReLU(inplace=True),
                                           )
        self.lin_lat = 4
        self.sep_reduction = nn.ModuleList([nn.Sequential(nn.Linear(self.slot_dim, self.slot_dim),
                                                          nn.ReLU(inplace=True),
                                                          nn.Linear(self.slot_dim, self.slot_dim))
                                            for _ in range(self.num_slots)])

    def forward(self, feat):
        cluster_centers = self.KMeans(feat)
        sep_cc = self.cc_separation(cluster_centers[None, None, ...]).squeeze()
        sep_cc = sep_cc.softmax(dim=1) + self.eps
        importance_sep_cc = torch.einsum('cz,sc->scz', cluster_centers, sep_cc)  # [5, 10, 64]
        importance_sep_cc = torch.div(importance_sep_cc, sep_cc.sum(dim=1)[:, None, None])
        # slots = torch.zeros(importance_sep_cc.shape, device=feat.device)
        slots = torch.zeros((self.num_slots, self.num_cluster, self.slot_dim), device=feat.device)  # 5x10x64
        for i in range(self.num_slots):
            slots[i, ...] = self.sep_reduction[i](importance_sep_cc[i, ...])  # 5x10x64
        slots = slots.sum(dim=1)
        return slots


class SlotInitVectorAttention(nn.Module):
    def __init__(self, num_slots, num_cluster=10, slot_dim=64, eps=1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.num_cluster = num_cluster
        self.slot_dim = slot_dim
        self.eps = eps
        self.scale = self.slot_dim ** -0.5
        self.KMeans = KMeansPP(10, 100, 0.0001, False)
        self.slot_bg = nn.Sequential(nn.Linear(1, self.slot_dim//2),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(self.slot_dim//2, self.slot_dim))
        self.slot_fg = nn.Sequential(nn.Linear(1, self.slot_dim//2),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(self.slot_dim//2, self.slot_dim))
        self.sep_reduction = nn.ModuleList([nn.Sequential(
                                               nn.Linear(self.slot_dim, self.slot_dim),
                                               nn.ReLU(inplace=True),
                                               nn.Linear(self.slot_dim, self.slot_dim))
                                            for _ in range(self.num_slots)])
        self.iter = 3

    def forward(self, feat):
        start = torch.ones(1, 1).to(device=feat.device)
        slot_bg = self.slot_bg(start)
        slot_fg = self.slot_fg(start).expand(self.num_slots-1, -1)
        slots = torch.cat((slot_bg, slot_fg), dim=0)
        cluster_centers = self.KMeans(feat)
        for _ in range(self.iter):
            updates = slots.clone()  # needed because otherwise backpropagation will not work
            dot_prod = torch.einsum('ac,bc->ab', updates, cluster_centers) * self.scale
            attn = dot_prod.softmax(dim=0) + self.eps
            attn_weights = attn / attn.sum(dim=-1, keepdim=True)
            updates = torch.einsum('ab,bc->ac', attn_weights, cluster_centers)
            #print(update.shape)
            for i in range(len(self.sep_reduction)):
                slots[i] = self.sep_reduction[i](updates[i])
        # slots = slots[None, ...]
        # slots = updates
        return slots


class SlotInitPixelAttention(nn.Module):
    def __init__(self, num_slots, num_cluster=10, slot_dim=64, eps=1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.num_cluster = num_cluster
        self.slot_dim = slot_dim
        self.eps = eps
        self.att_iter = 3
        self.lin_lat = 4
        self.KMeans = KMeansPP(10, 100, 0.0001, False)
        self.attn_cc_enc = nn.Sequential(nn.Linear(2, 5),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(5, 1))
        self.sep_reduction = nn.ModuleList([nn.Sequential(
                                                nn.Linear(self.num_cluster*self.slot_dim, self.lin_lat*self.slot_dim),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(self.lin_lat*self.slot_dim, self.slot_dim))
                                            for _ in range(self.num_slots)])

    def forward(self, feat):
        slots = torch.zeros((self.num_slots, self.slot_dim), device=feat.device).normal_(mean=0.5, std=1)[:, None, ...]
        cluster_centers = self.KMeans(feat)[None, ...]

        for _ in range(self.att_iter):
            distances = torch.subtract(cluster_centers, slots)
            abs_dis = torch.abs(distances)
            max_d = torch.max(abs_dis)
            raw_attn = 1 - torch.divide(abs_dis, max_d)
            attn = raw_attn.softmax(dim=0)
            cc_flat = cluster_centers.flatten(start_dim=1)
            attn_flat = attn.flatten(start_dim=1)[:, :, None]
            cc_exp = cc_flat.repeat(5, 1)[:, :, None]
            cat_soft_cc = torch.cat((cc_exp, attn_flat), dim=-1)
            attn_cc_enc = self.attn_cc_enc(cat_soft_cc).squeeze()
            for i in range(len(self.sep_reduction)):
                slots[i] = self.sep_reduction[i](attn_cc_enc[i])
        return slots.permute(1, 0, 2).squeeze(0)


class SlotInitContext(nn.Module):
    def __init__(self, num_slots, num_cluster=10, slot_dim=64, eps=1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.num_cluster = num_cluster
        self.slot_dim = slot_dim
        self.eps = eps
        self.scale = slot_dim ** -0.5
        self.KMeans = KMeansPP(10, 100, 0.0001, False)

        self.context_v = torch.arange(start=0, end=self.num_slots, step=1, device='cuda') * (2*torch.pi/self.num_slots)
        self.context_v = torch.stack((torch.sin(self.context_v), torch.cos(self.context_v)), dim=0).permute(1, 0)  # 5x2
        self.context_net = nn.Sequential(nn.Linear(2, self.slot_dim),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(self.slot_dim, self.slot_dim))

        self.norm = nn.LayerNorm(self.slot_dim)

        self.cc_to_slots = nn.Sequential(nn.Linear(self.slot_dim, self.slot_dim),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(self.slot_dim, self.num_slots))

        self.pseudo_weights = nn.Sequential(nn.Linear(self.slot_dim*2, self.slot_dim*2),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(self.slot_dim*2, self.slot_dim*2),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(self.slot_dim*2, self.slot_dim),
                                            nn.Sigmoid())

    def forward(self, feat):
        cluster_centers = self.KMeans(feat)  # 10x64
        cc_norm = self.norm(cluster_centers)  # 10x64

        slot_buckets = self.cc_to_slots(cc_norm)**2 * self.scale  # 10x5
        slot_buckets = slot_buckets.softmax(dim=-1)  # + self.eps  # 10x5

        context = self.context_net(self.context_v)  # 5x64
        context = context[None, :, :].repeat(self.num_cluster, 1, 1)  # 10x5x64

        cc_norm = cc_norm[:, None, :].repeat(1, self.num_slots, 1)  # 10x5x64
        pw_input = torch.cat((context, cc_norm), dim=-1)  # 10x5x128
        pseudo_weights = self.pseudo_weights(pw_input)  # 10x5x64

        cc_weighted = torch.mul(pseudo_weights, cluster_centers[:, None, :])  # 10x5x64

        slot_buckets = torch.div(slot_buckets, slot_buckets.sum(dim=0, keepdim=True) + self.eps)  # 10x5
        slot_buckets = torch.mul(cc_weighted, slot_buckets[:, :, None])  # 10x5x64
        slots = slot_buckets.sum(dim=0)  # 5x64
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
        self.laten_dim = 16
        self.context_net = nn.Sequential(nn.Linear(2, self.laten_dim),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(self.laten_dim, self.laten_dim))

        self.norm = nn.LayerNorm(self.slot_dim)
        self.cc_squeeze = nn.Sequential(nn.Linear(self.slot_dim, self.slot_dim//2),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(self.slot_dim//2, self.laten_dim))
        self.pseudo_weights = nn.Sequential(nn.Linear(self.laten_dim * 2, self.slot_dim//2),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(self.slot_dim//2, self.slot_dim),
                                            nn.Sigmoid()
                                            )

    def forward(self, feat):
        cluster_centers = self.KMeans(feat)  # 10x64
        cc_norm = self.norm(cluster_centers)  # 10x64
        #cc_norm = cluster_centers
        context_v = torch.arange(start=0, end=self.num_slots, step=1, device=feat.device) * (2 * (torch.acos(torch.zeros(1)).item() * 2) / self.num_slots)
        context_v = torch.stack((torch.sin(context_v), torch.cos(context_v)), dim=0).permute(1, 0)  # 5x2

        context = self.context_net(context_v)  # 5x4
        context = context[None, :, :].repeat(self.num_cluster, 1, 1)  # 10x5x4

        cc_norm = self.cc_squeeze(cc_norm)  # 10x4
        cc_norm = cc_norm[:, None, :].repeat(1, self.num_slots, 1)  # 10x5x4

        pw_input = torch.cat((context, cc_norm), dim=-1)  # 10x5x8
        pseudo_weights = self.pseudo_weights(pw_input)  # 10x5x64

        # cc_weighted = torch.mul(pseudo_weights, cluster_centers[:, None, :])  # 10x5x64
        cc_weighted = torch.einsum('csd,csd->csd', pseudo_weights, cluster_centers[:, None, :])

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

