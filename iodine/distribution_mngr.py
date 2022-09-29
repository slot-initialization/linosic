import torch.distributed as dist
import torch
import os


class DistributionManager:
    def __init__(self, options):
        self.opt = options.opt
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            self.rank = int(os.environ["RANK"])
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.gpu = int(os.environ['LOCAL_RANK'])
            self.distributed = True
        else:
            print('Not using distributed mode')
            self.distributed = False
        torch.cuda.set_device(self.gpu)
        self.dist_backend = 'nccl'
        torch.distributed.init_process_group(backend=self.dist_backend,
                                             world_size=self.world_size,
                                             rank=self.rank)
        torch.distributed.barrier()











