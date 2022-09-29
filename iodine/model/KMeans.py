from torch import nn
import torch
from math import pi
import numpy as np

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


class SlotInitKmeans(nn.Module):
    def __init__(self, num_slots, num_cluster=10, slot_dim=64):
        super().__init__()
        self.num_slots = num_slots
        self.num_cluster = num_cluster
        self.slot_dim = slot_dim
        self.mapping_to_slots = nn.Sequential(
            nn.Linear(self.num_cluster * slot_dim, (self.num_cluster + self.num_slots) // 2 * slot_dim),
            nn.ReLU(inplace=True),
            nn.Linear((self.num_cluster + self.num_slots) // 2 * slot_dim, self.num_slots * slot_dim),
        )

    def forward(self, cluster_centers):
        cluster_centers = cluster_centers.flatten(start_dim=1)
        slots = self.mapping_to_slots(cluster_centers)
        slots = slots.view((cluster_centers.shape[0], self.num_slots, self.slot_dim))
        return slots


class SlotInitPooling(nn.Module):
    def __init__(self, num_slots, num_cluster=10, slot_dim=64, eps=1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.num_cluster = num_cluster
        self.slot_dim = slot_dim
        self.eps = eps
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
                                           nn.ReLU(inplace=True))
        self.lin_lat = 4
        self.sep_reduction = nn.ModuleList([nn.Sequential(nn.Linear(self.num_cluster * self.slot_dim, self.lin_lat * self.slot_dim),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(self.lin_lat * self.slot_dim, self.slot_dim)) for _ in range(self.num_slots)])

    def forward(self, cluster_centers):
        sep_cc = self.cc_separation(cluster_centers[:, None, ...])
        sep_cc = sep_cc.softmax(dim=2) + self.eps
        importance_sep_cc = torch.einsum('bcz,bsc->bscz', cluster_centers.squeeze(), sep_cc.squeeze()).flatten(start_dim=2)  # [B, 5, 10, 64] --> [B, 5, 10*64]
        slots = torch.zeros((self.num_slots, self.slot_dim), device=sep_cc.device)
        for i in range(len(self.sep_reduction)):
            slots[:, i, :] = self.sep_reduction[i](importance_sep_cc[:, i, ...])
        return slots


class SlotInitSAMLPMax(nn.Module):  # SlotInitSelfAttn with MLP and MaxPooling
    def __init__(self, num_slots, num_cluster=10, slot_dim=64, eps=1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.num_cluster = num_cluster
        self.slot_dim = slot_dim
        self.eps = eps
        self.iter = 3
        self.softmax_temp = self.slot_dim ** -0.5
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

    def forward(self, cluster_centers):
        k = self.to_k(cluster_centers)  # Bx10x64
        v = self.to_v(cluster_centers)  # Bx10x64

        for _ in range(self.iter):
            cc_prev = cluster_centers
            cluster_centers = self.cc_norm(cluster_centers)
            attn = torch.einsum("bnd,bmd->bnm", k, self.to_q(cluster_centers)) * self.softmax_temp
            attn = attn.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)
            updates = torch.einsum("bna,bad->bnd", attn, v)
            cluster_centers = self.gru(updates, cc_prev)
            cluster_centers = cluster_centers + self.mlp(cluster_centers)
        slots = []
        for i in range(self.num_slots):
            slot_raw = self.slot_mapping[i](cluster_centers).permute(0, 2, 1)  # Bx64x10
            slot = self.max_pool(slot_raw).permute(0, 2, 1)  # Bx1x64
            slots.append(slot)
        slots = torch.stack(slots, dim=1).squeeze(2)
        return slots


class SlotInitContext(nn.Module):
    def __init__(self, num_slots, num_cluster=10, slot_dim=64, eps=1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.num_cluster = num_cluster
        self.slot_dim = slot_dim
        self.eps = eps
        self.scale = slot_dim ** -0.5

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

    def forward(self, cluster_centers):
        cc_norm = self.norm(cluster_centers)  # Bx10x64

        slot_buckets = self.cc_to_slots(cc_norm)**2 * self.scale  # Bx10x5
        slot_buckets = slot_buckets.softmax(dim=-1)  # + self.eps  # Bx10x5

        context = self.context_net(self.context_v)  # Bx5x64
        context = context[:, None, :, :].repeat(1, self.num_cluster, 1, 1)  # Bx10x5x64

        cc_norm = cc_norm[:, :, None, :].repeat(1, 1, self.num_slots, 1)  # Bx10x5x64
        pw_input = torch.cat((context, cc_norm), dim=-1)  # Bx10x5x128
        pseudo_weights = self.pseudo_weights(pw_input)  # Bx10x5x64

        cc_weighted = torch.mul(pseudo_weights, cluster_centers[:, :, None, :])  # Bx10x5x64

        slot_buckets = torch.div(slot_buckets, slot_buckets.sum(dim=1, keepdim=True) + self.eps)  # Bx10x5
        slot_buckets = torch.mul(cc_weighted, slot_buckets[:, :, :, None])  # Bx10x5x64
        slots = slot_buckets.sum(dim=1)  # Bx5x64
        return slots


class SlotInitPseudoWeights(nn.Module):
    def __init__(self, opt, num_slots, num_cluster=10, slot_dim=64, eps=1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.num_cluster = num_cluster
        self.slot_dim = slot_dim
        self.eps = eps

        # Position encoding for slots in the context_net
        pos_enc_len = 32
        pos_enc_sin = lambda x, y: torch.sin(x * (pi / pos_enc_len) * y)
        pos_enc_cos = lambda x, y: torch.cos(x * (pi / pos_enc_len) * y)
        pos_code_arange = torch.arange(0, pos_enc_len)[None, :].repeat(num_slots, 1)
        pos_code_sin = torch.zeros(pos_code_arange.shape)
        pos_code_cos = torch.zeros(pos_code_arange.shape)
        for i in range(0, num_slots):
            pos_code_sin[i] = pos_enc_sin(pos_code_arange[i].clone(), i + 1)
            pos_code_cos[i] = pos_enc_cos(pos_code_arange[i].clone(), num_slots - i)
        self.context_v = torch.cat([pos_code_sin, pos_code_cos], dim=-1)  # .to(opt.device_type)  # 5x64

        self.context_net = nn.Sequential(nn.Linear(64, 64),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(64, 64))

        self.norm = nn.LayerNorm(self.slot_dim)

        self.pseudo_weights = nn.Sequential(nn.Linear(self.slot_dim + 64, self.slot_dim + 64),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(self.slot_dim + 64, self.slot_dim + 64),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(self.slot_dim + 64, self.slot_dim),
                                            nn.Sigmoid())

    def forward(self, cluster_centers):
        cc_norm = self.norm(cluster_centers)  # Bx10x64
        context = self.context_net(self.context_v.to(device=cluster_centers.device, copy=True))  # 5x64
        context = context[None, None, :, :].repeat(cluster_centers.shape[0], self.num_cluster, 1, 1)  # Bx10x5x64

        cc_norm = cc_norm[:, :, None, :].repeat(1, 1, self.num_slots, 1)  # Bx10x5x64
        pw_input = torch.cat((context, cc_norm), dim=-1)  # Bx10x5x128
        pseudo_weights = self.pseudo_weights(pw_input)  # Bx10x5x64

        cc_weighted = torch.mul(pseudo_weights, cluster_centers[:, :, None, :])  # Bx10x5x64
        slots = cc_weighted.sum(dim=1)  # / self.num_cluster  # Bx5x64
        return slots


class SlotInitPseudoWeightsMlp(nn.Module):
    def __init__(self, num_slots, num_cluster=10, slot_dim=64, eps=1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.num_cluster = num_cluster
        self.slot_dim = slot_dim
        self.eps = eps

        self.context_v = torch.arange(start=0, end=self.num_slots, step=1, device='cuda') * (2*torch.pi/self.num_slots)
        self.context_v = torch.stack((torch.sin(self.context_v), torch.cos(self.context_v)), dim=0).permute(1, 0)  # 5x2
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

    def forward(self, cluster_centers):
        cc_norm = self.norm(cluster_centers)  # Bx10x64

        context = self.context_net(self.context_v)  # 5x64
        context = context[None, None, :, :].repeat(cluster_centers.shape[0], self.num_cluster, 1, 1)  # Bx10x5x64

        cc_norm = cc_norm[:, :, None, :].repeat(1, 1, self.num_slots, 1)  # Bx10x5x64
        pw_input = torch.cat((context, cc_norm), dim=-1)  # Bx10x5x128
        pseudo_weights = self.pseudo_weights(pw_input)  # Bx10x5x64

        cc_weighted = torch.mul(pseudo_weights, cluster_centers[:, :, None, :])  # Bx10x5x64
        slots = cc_weighted.sum(dim=1)  # / self.num_cluster  # Bx5x64
        slots = self.out_linear(slots)
        return slots


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
        # print(cluster_labels)
        cc_embedding = torch.zeros((20, Z.shape[0], Z.shape[1]), device=cluster_labels.device)  # mxmxd
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
