import os
from utils.parallel_utils import *


class PathManagement:
    def __init__(self, options):
        self.opt = options.opt
        self.train_root = self.create_path([self.opt.datasets_root, self.opt.dataset_version, 'train']) + '/'
        self.test_root = self.create_path([self.opt.datasets_root, self.opt.dataset_version, 'test']) + '/'
        self.test_mask_root = self.create_path([self.opt.datasets_root, self.opt.dataset_version, 'test_mask']) + '/'
        self.val_root = self.create_path([self.opt.datasets_root, self.opt.dataset_version, 'val']) + '/'
        self.val_mask_root = self.create_path([self.opt.datasets_root, self.opt.dataset_version, 'val_mask']) + '/'
        if self.opt.experiment_name:
            final_name = self.opt.model_version+'_'+self.opt.experiment_name
        else:
            final_name = self.opt.model_version
        self.checkpoints_root = self.create_path([self.opt.checkpoints_root, self.opt.dataset_version, final_name]) + '/'
        self.best_checkpoints_root = self.create_path([self.opt.checkpoints_root, self.opt.dataset_version, final_name, 'best']) + '/'
        self.resume_checkpoints_path = self.checkpoints_root + '/' + str(self.opt.resume_epoch) + '.ckpt'

        self.logs_root = self.create_path([self.opt.logs_root, self.opt.dataset_version, final_name]) + '/'
        self.options_root = self.create_path([self.opt.logs_root, self.opt.dataset_version, final_name, 'options']) + '/'
        self.train_loss_root = self.create_path([self.opt.logs_root, self.opt.dataset_version, final_name, 'train_loss']) + '/'

        # Only for Evaluation of a model.
        if self.opt.mode == 'eval':
            if self.opt.eval_best_checkpoint:
                print(self.best_checkpoints_root)
                ckpt_f = os.listdir(self.best_checkpoints_root)[0]
                self.checkpoint_path = self.best_checkpoints_root + ckpt_f
                self.eval_result_path = self.create_path([self.checkpoints_root, 'evaluation', 'best']) + '/result.txt'
            else:
                self.checkpoint_path = self.checkpoints_root + str(self.opt.eval_checkpoint_number) + '.ckpt'
                self.eval_result_path = self.create_path([self.checkpoints_root, 'evaluation']) + '/result.txt'

    def create_path(self, directory_list):
        current_path = ''
        for f_name in directory_list:
            current_path = current_path + f_name + '/'
            if is_main_process():
                if not os.path.exists(current_path):
                    os.makedirs(current_path, exist_ok=True)
        return current_path[:-1]  # delete last "/"








