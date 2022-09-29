from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid


class TensorboardLogger:
    def __init__(self, options, log_path):
        self.opt = options.opt
        self.tb_writer = SummaryWriter(log_dir=log_path)

    def write(self, result_dict, epoch):
        for key in result_dict.keys():
            self.tb_writer.add_scalar(tag=key, scalar_value=result_dict[key], global_step=epoch)
        self.tb_writer.flush()

    def display(self, image_dict, epoch):
        for key in image_dict:
            img_list_length = len(image_dict[key])
            for i in range(img_list_length):
                img = image_dict[key][i]  # .detach()  # .cpu().numpy()
                img_grid = make_grid(img)
                self.tb_writer.add_image(str(key)+'_'+str(i), img_grid, global_step=epoch)

    def close_writer(self):
        self.tb_writer.close()









