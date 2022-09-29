from arg_parser import Options
import torch
import json
from model_mngr import ModelManagement
from test_mngr import TestManagement
from path_mngr import PathManagement
from persistance.tensorboard_mngr import TensorboardLogger


class EvalManagement:
    def __init__(self):
        self.options = Options()
        self.opt = self.options.opt
        self.mode = self.opt.mode
        print(self.mode)
        print(self.options.opt)
        torch.manual_seed(self.opt.seed)
        self.path_management = PathManagement(options=self.options)
        self.model_management = ModelManagement(options=self.options,
                                                train_root=None,
                                                test_eval_root=self.path_management.val_root,
                                                mask_root=self.path_management.val_mask_root,
                                                eval_checkpoint_path=self.path_management.checkpoint_path)
        self.test_management = TestManagement(options=self.options,
                                              data_loader=self.model_management.test_data_loader)
        self.tb_management = TensorboardLogger(options=self.options,
                                               log_path=self.path_management.logs_root)

    def evaluate(self):
        model = self.model_management.model
        evaluation_metric_dict, evaluation_img_dict = self.test_management.test_model(model=model)
        print(evaluation_metric_dict)
        with open(self.path_management.eval_result_path, 'a') as eval_f:
            json.dump(evaluation_metric_dict, eval_f, indent=2)
            eval_f.close()
        self.tb_management.display(image_dict=evaluation_img_dict, epoch=0)
        self.tb_management.close_writer()


if __name__ == '__main__':
    eval_management = EvalManagement()
    eval_management.evaluate()









