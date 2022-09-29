import os
import json
import torchvision
from torchvision.io import ImageReadMode as irm
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import InterpolationMode as Ipm
from matplotlib import pyplot as plt


class DsCreator(torch.utils.data.Dataset):
    def __init__(self, split='train', max_num_slots=5, ds_root='./datasets/CLEVR4/', size=-1, resolution=(128, 128), mask=False):
        super(DsCreator, self).__init__()
        self.split = split
        self.max_num_slots = max_num_slots
        self.ds_root = ds_root
        self.split_root = ds_root + split + '/'
        self.mask = mask
        if self.mask:
            self.mask_root = self.ds_root + split + '_mask/'
        """Build dataset."""
        print(self.split)
        scenes = sorted(os.listdir(self.split_root))
        if size >= 0:
            scenes = list(scenes)[:size]  #@TODO: CHANGE the number to "400:" from ":100" to get the last 100 pictures with the high object count
        resize = torchvision.transforms.Resize(resolution, interpolation=Ipm.BILINEAR)
        img_f = [resize(torchvision.io.read_image(self.split_root + scene, mode=irm.RGB)).float() for scene in scenes]
        #img_f = []
        #for scene in scenes:
        #    try:
        #        img_f.append(resize(torchvision.io.read_image(self.split_root + scene, mode=irm.RGB)).float())
        #    except:
        #        print('Delete this scene from the dataset, it has errors:')
        #        print(scene)
        self.ds_images = [((i_f / 255.0) - 0.5) * 2.0 for i_f in img_f]

        if self.mask:
            self.ds_masks = []
            resize = torchvision.transforms.Resize(resolution, interpolation=Ipm.NEAREST)
            mask_f = [resize(torchvision.io.read_image(self.mask_root + mask, mode=irm.RGB)).float() for mask in scenes]
            for m_f in mask_f:
                u_vals = torch.unique(m_f.flatten(start_dim=1), dim=1)
                u_val_background_idx = (u_vals == torch.tensor([64, 64, 64]).view(3, 1).repeat(1, len(u_vals[0]))).sum(dim=0) == 3
                not_u_val_background_idx = ~ u_val_background_idx
                u_vals = torch.cat((u_vals[:, not_u_val_background_idx], u_vals[:, u_val_background_idx]), dim=1)
                ds_mask = torch.zeros(min(u_vals.shape[1], self.max_num_slots), m_f.shape[1], m_f.shape[2])
                for z in range(min(u_vals.shape[1], self.max_num_slots)):
                    u_v = u_vals[:, z][:, None, None].repeat(1, m_f.shape[1], m_f.shape[2])
                    ds_mask[z][torch.prod([m_f == u_v][0], dim=0).bool()] = 1
                self.ds_masks.append(ds_mask)
        print('Created Dataset with masks:', self.split, len(self.ds_images), len(self.ds_masks)) if self.mask \
            else print('Created Dataset without masks:', self.split, len(self.ds_images))

    def __len__(self):
        return len(self.ds_images)

    def __getitem__(self, idx):
        if self.mask:
            mask = self.ds_masks[idx]
            #print('mask', mask.shape)
            img = self.ds_images[idx]
            #print('img', img.shape)
            sample = {'image': img,
                      'mask': mask}
            return sample
        else:
            img = self.ds_images[idx]
            sample = {'image': img}
            return sample



