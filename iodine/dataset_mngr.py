import os
import torchvision
from torchvision.io import ImageReadMode as irm
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode as Ipm
from torchvision import transforms
from matplotlib import pyplot as plt

class DataSetCreator(torch.utils.data.Dataset):
    def __init__(self, options, max_num_slots=5, split_root='./datasets/MDS/train/', mask_root='./datasets/MDS/test_mask/', size=-1, resolution=(128, 128), mask=False):
        super(DataSetCreator, self).__init__()
        self.opt = options
        self.max_num_slots = max_num_slots
        self.split_root = split_root
        self.mask = mask
        if self.mask:
            self.mask_root = mask_root
        """Build dataset."""
        scenes = sorted(os.listdir(self.split_root))
        if size >= 0:
            scenes = list(scenes)[:size]
        self.scenes = scenes
        self.transform = self.img_transform()  # torchvision.transforms.Resize(resolution, interpolation=Ipm.BILINEAR)
        self.resize = torchvision.transforms.Resize(resolution, interpolation=Ipm.NEAREST)
        self.len = len(self.scenes)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        scene_name = self.scenes[idx]
        ds_image = self.transform((torchvision.io.read_image(self.split_root + scene_name, mode=irm.RGB)).float())
        #plt.imshow(ds_image.permute(1, 2, 0).cpu().numpy())
        #plt.show()
        sample = {'image': ds_image}

        if self.mask:
            mask_f = self.resize(torchvision.io.read_image(self.mask_root + scene_name, mode=irm.RGB)).float()
            u_vals = torch.unique(mask_f.flatten(start_dim=1), dim=1)
            u_val_background_idx = (u_vals == torch.tensor([64, 64, 64]).view(3, 1).repeat(1, len(u_vals[0]))).sum(dim=0) == 3
            not_u_val_background_idx = ~ u_val_background_idx
            u_vals = torch.cat((u_vals[:, not_u_val_background_idx], u_vals[:, u_val_background_idx]), dim=1)
            ds_mask = torch.zeros(self.max_num_slots, mask_f.shape[1], mask_f.shape[2])
            for z in range(min(u_vals.shape[1], self.max_num_slots)):
                u_v = u_vals[:, z][:, None, None].repeat(1, mask_f.shape[1], mask_f.shape[2])
                ds_mask[z][torch.prod([mask_f == u_v][0], dim=0).bool()] = 1
            sample['mask'] = ds_mask
        return sample

    def img_transform(self):
        ds_version = self.opt.dataset_version
        if ds_version == 'MDS':
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((128, 128)),
                transforms.ToTensor()
            ])
        elif ds_version in ['CLEVR4', 'CLEVR6', 'CLEVR8', 'CLEVR10']:
            transform = transforms.Compose([
                transforms.ToPILImage(),
                # transforms.CenterCrop(192),
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
            ])
        elif ds_version in ['CHAIRS_DIVERSE', 'CHAIRS_EASY']:
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
            ])
        return transform



