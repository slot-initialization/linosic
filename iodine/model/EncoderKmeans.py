import numpy as np
from torch import nn
import torch
import torch.nn.functional as F


def build_grid(resolution, device):
    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1)).to(device)


class SoftPositionEmbed(nn.Module):
    def __init__(self, hidden_size):
        """Builds the soft position embedding layer.
        Args:
        hidden_size: Size of input feature dimension.
        resolution: Tuple of integers specifying width and height of grid.
        """
        super().__init__()
        self.embedding = nn.Linear(7, hidden_size, bias=True)

    def forward(self, inputs):
        grid = build_grid(inputs.shape[2:], inputs.device)
        inputs = self.embedding(torch.cat((grid.repeat((inputs.shape[0], 1, 1, 1)), inputs.permute(0, 2, 3, 1)), dim=-1)).permute(0, 3, 1, 2)
        return inputs


class Encoder(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.encoder_pos = SoftPositionEmbed(hid_dim)
        self.conv1 = nn.Conv2d(hid_dim, hid_dim, 5, padding=2)
        self.conv2 = nn.Conv2d(hid_dim, hid_dim, 5, padding=2)
        self.conv3 = nn.Conv2d(hid_dim, hid_dim, 5, padding=2)
        self.conv4 = nn.Conv2d(hid_dim, hid_dim, 5, padding=2)

    def forward(self, x):
        x = self.encoder_pos(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        return x