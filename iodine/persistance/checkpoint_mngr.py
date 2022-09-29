import os
import torch


class CheckpointManagement:
    def __init__(self, options, ckpt_root, update_ckpt=False):
        self.opt = options.opt
        self.update_ckpt = update_ckpt
        self.ckpt_root = ckpt_root

    def save_dict(self, save_dict, identifier):
        if not self.update_ckpt:
            path_to_ckpt = self.ckpt_root+identifier+'.ckpt'
            with open(path_to_ckpt, 'w') as ckpt_f:
                torch.save(save_dict, path_to_ckpt)
                ckpt_f.close()
        else:
            list_ckpt_root = os.listdir(self.ckpt_root)
            if not len(list_ckpt_root) == 0:
                os.remove(self.ckpt_root + list_ckpt_root[0])
            path_to_ckpt = self.ckpt_root + identifier + '.ckpt'
            with open(path_to_ckpt, 'w') as ckpt_f:
                torch.save(save_dict, path_to_ckpt)
                ckpt_f.close()



