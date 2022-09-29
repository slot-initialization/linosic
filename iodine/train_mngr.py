from arg_parser import Options
import torch.distributed as dist
import torch
import time
import datetime
from tqdm import tqdm
from model_mngr import ModelManagement
from test_mngr import TestManagement
from persistance.tensorboard_mngr import TensorboardLogger
from persistance.checkpoint_mngr import CheckpointManagement
from path_mngr import PathManagement
from utils.parallel_utils import *
from distribution_mngr import DistributionManager


class TrainManagement:
    def __init__(self):
        self.options = Options()
        self.opt = self.options.opt
        self.mode = self.opt.mode
        print(self.mode)
        print(self.options.opt)
        torch.manual_seed(self.opt.seed)
        self.distr_management = DistributionManager(options=self.options)
        self.path_management = PathManagement(options=self.options)
        self.options.save_options(self.path_management.options_root)
        self.model_management = ModelManagement(options=self.options,
                                                train_root=self.path_management.train_root,
                                                test_eval_root=self.path_management.test_root,
                                                mask_root=self.path_management.test_mask_root,
                                                resume_checkpoint_path=self.path_management.resume_checkpoints_path)
        self.test_management = TestManagement(options=self.options,
                                              data_loader=self.model_management.test_data_loader)
        self.tb_management = TensorboardLogger(options=self.options,
                                               log_path=self.path_management.logs_root)
        self.ckpt_management = CheckpointManagement(options=self.options,
                                                    ckpt_root=self.path_management.checkpoints_root,
                                                    update_ckpt=False)
        self.best_ckpt_management = CheckpointManagement(options=self.options,
                                                         ckpt_root=self.path_management.best_checkpoints_root,
                                                         update_ckpt=True)

    def train(self):
        device = self.opt.device_type
        best_test_loss = None
        model = self.model_management.model
        model.to(device)
        if self.opt.parallel:
            model = torch.nn.parallel.DistributedDataParallel(module=model,
                                                              device_ids=[self.distr_management.gpu])
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)
        len_tdl = len(self.model_management.train_data_loader)
        start_time = time.time()
        for epoch in range(self.opt.resume_epoch, self.opt.max_epoch):
            torch.distributed.barrier()
            model.train()
            epoch_loss = torch.zeros(size=(1,), device=self.opt.device_type)
            for sample in tqdm(self.model_management.train_data_loader):
                if self.opt.parallel:
                    self.model_management.sampler.set_epoch(epoch)
                model.train()
                image = sample['image'].to(device)
                loss = model(image)
                loss = loss.mean()
                model.zero_grad()
                loss.backward()
                self.model_management.optimizer.step()
                epoch_loss += loss
            epoch_loss = epoch_loss / len_tdl
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(epoch_loss.clone().detach())
            if is_main_process():
                epoch_loss = epoch_loss.item()
                print('\n', epoch, epoch_loss)
                write_mode = 'w' if not epoch else 'a'
                with open(self.path_management.train_loss_root + 'train_loss.txt', write_mode) as train_loss_f:
                    train_loss_f.write('Epoch: ' + str(epoch) +
                                       ' train_loss: ' + str(epoch_loss) +
                                       ' time: ' + str(datetime.timedelta(seconds=time.time() - start_time)) +
                                       '\n')
                    train_loss_f.close()
                self.tb_management.write(result_dict={'train_loss': epoch_loss}, epoch=epoch)
                if not epoch % self.opt.save_freq:
                    self.ckpt_management.save_dict(save_dict=self.model_management.get_state(), identifier=str(epoch))
            if not epoch % self.opt.test_freq:
                torch.distributed.barrier()
                metric_dict, image_dict = self.test_management.test_model(model=self.model_management.model)
                if is_main_process():
                    self.tb_management.write(result_dict=metric_dict, epoch=epoch)
                    self.tb_management.display(image_dict=image_dict, epoch=epoch)
                    test_loss = metric_dict['test_loss']
                    if best_test_loss is None or test_loss < best_test_loss:
                        self.best_ckpt_management.save_dict(save_dict=self.model_management.get_state(), identifier=str(epoch))
                        best_test_loss = test_loss

        self.tb_management.close_writer()


if __name__ == '__main__':
    train_management = TrainManagement()
    train_management.train()
















