import argparse
import json
import os


class Options:
    def __init__(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--mode', default='train', type=str, help='What do you want to do: "train" model or "eval" model?')  # train
        parser.add_argument('--load_options_path', default='', type=str, help='Path to options.txt file, from wich you want to load arguments.')
        parser.add_argument('--device_type', default='cuda', type=str, help='The device type on which the model will run')
        parser.add_argument('--seed', default=0, type=int, help='The seed that initializes the torch-randomness.')

        # Generel arguments for training
        parser.add_argument('--parallel', action='store_true', help='Run model on multiple devices in parallel batch mode. This argument has to be the same for training and evaluating the same model.')
        parser.add_argument('--resume_training', action='store_true', help="WHeter to resume training from given epoch number and checkpoint.")
        parser.add_argument('--resume_epoch', default=0, type=int, help='From which epoch to resume training.')
        parser.add_argument('--max_epoch', default=1000, type=int, help='After this epoch count the training stops.')
        parser.add_argument('--base_learning_rate', default=3e-4, type=float, help='The amount how much weights can change at maximum during one backward pass.')
        parser.add_argument('--weight_decay', default=0.0000, type=float, help='')
        parser.add_argument('--device_ids', default='0', type=str, help='The devices on which the model will run in (not)parallel batch mode: single gpu --> "0", two gpus -->"0,1"')
        parser.add_argument('--test_freq', default=20, type=int, help='The epoch frequency, after which you test the model during training.')  # 20
        parser.add_argument('--save_freq', default=20, type=int, help='The epoch frequency, after which yousave the model during training.')  # 20
        parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
        parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

        # General arguments for evaluation
        parser.add_argument('--eval_best_checkpoint', action='store_true', help='Wether you want to evaluate the best model for one experiment.')
        parser.add_argument('--eval_checkpoint_number', default=0, type=int, help='The checkpoint number in the according experiment folder.')
        parser.add_argument('--save_generated_images', default=0, type=int, help='Wether you want to store a fraction of the reconstructions in the evaluation.')
        parser.add_argument('--sort_slots', action='store_true', help='Sort slots')

        # Dataset specific
        parser.add_argument('--dataset_version', default='CLEVR4', type=str, help='Name of the dataset in the datasets_root')
        parser.add_argument('--datasets_root', default='./datasets', help='Root where all datasets are stored (not just one specific)')
        parser.add_argument('--train_dataset_size', default=-1, type=int, help='Size of training dataset')  # -1
        parser.add_argument('--test_dataset_size', default=-1, type=int, help='Size of test dataset, which is used after test_freq during training.')  # -1
        parser.add_argument('--train_batch_size', default=16, type=int, help='Train dataset get split up in mini batches of this size.')
        parser.add_argument('--num_workers', default=4, type=int, help='Amount of worker threads for dataset management.')

        # Persistance
        parser.add_argument('--experiment_name', default='', type=str, help='Experiment name to further help distinguishing different runs.')
        parser.add_argument('--checkpoints_root', default='./checkpoints', help='Where the checkpoints of all different models are saved after save_freq during training. The final checkpoint directory is ./checkpoints/dataset_version/model_version+_+experiment_name/.')
        parser.add_argument('--logs_root', default='./logs', help='Where tensorboard saves its logs after test_freq. The final los directory is ./logs/dataset_version/model_version+_+experiment_name/.')

        # Iodine specific
        parser.add_argument('--model_version', default='pseudoweights', type=str, help='Name of the model version: original, kmeans, pseudoweights, context, selfattention, transformer')
        parser.add_argument('--iters', default=5, type=int, help='How often iodine iterates on one image.')
        parser.add_argument('--slots', default=5, type=int, help='How much slots the latent model space will have.')
        parser.add_argument('--sigma', default=0.1, type=float, help='')
        parser.add_argument('--dim_latent', default=64, type=int, help='The dimension of the model latent space.')
        parser.add_argument('--resolution', default=128, type=int, help='Resolution of the images and the model latent space.')
        parser.add_argument('--img_channels', default=3, type=int, help='')
        parser.add_argument('--layernorm', action='store_true', help='Wether the model uses layer normalization or not')
        parser.set_defaults(layernorm=True)
        parser.add_argument('--stop_gradient', action='store_true', help='Wether the model uses layer normalization or not')
        parser.set_defaults(stop_gradient=True)
        parser.add_argument('--img_encoder', action='store_true', help='Wether the model_versions use an encoder before kmeans algorithm.')
        parser.add_argument('--cluster_centers', default=10, type=int, help='Amount of cluster centers in the the kmeans algorithm.')
        parser.add_argument('--kmeans_iteration', default=100, type=int, help='Maximum iterations of the kmeans algorithm.')
        parser.add_argument('--ref_conv_channels', default=64, type=int, help='Latent size of the convolution network in the refinement network.')
        parser.add_argument('--ref_conv_layers', default=4, type=int, help='Amount of convolution layers of the convolution network in the refinement network.')
        parser.add_argument('--ref_mlp_units', default=256, type=int, help='Size of mlp in refinement network.')
        parser.add_argument('--ref_kernel_size', default=3, type=int, help='Kernel window size of the convolution network in the refinement network.')
        parser.add_argument('--ref_stride_size', default=2, type=int, help='Stride size of the kernel of the convolution network in the refinement network.')
        parser.add_argument('--dec_conv_channels', default=64, type=int, help='Latent size of the convolution network in the decoder network.')
        parser.add_argument('--dec_conv_layers', default=4, type=int, help='Amount of convolution layers of the convolution network in the decoder network.')
        parser.add_argument('--dec_kernel_size', default=3, type=int, help='Kernel window size of the convolution network in the decoder network.')
        parser.add_argument('--encoding', default='posterior,grad_post,image,means,mask,mask_logits,mask_posterior,grad_means,grad_mask,likelihood,leave_one_out_likelihood,coordinate', type=str, help='Comma separated encodings.')

        self.opt = parser.parse_args()
        self.opt.device_ids = [int(gpu_id) for gpu_id in self.opt.device_ids.split(',')]
        self.opt.encoding = [enc for enc in self.opt.encoding.split(',')]
        assert self.opt.mode in ['train', 'eval']
        assert self.opt.model_version in ['pseudoweights', 'kmeans', 'original', 'trafo', 'direct', 'direct_ms', 'pseudoweights_ms', 'mlp_ms']
        if self.opt.load_options_path:
            self.opt = self.load_options(self.opt.load_options_path)
        self.mode = self.opt.mode

    def save_options(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        with open(save_path+'options.txt', 'w') as f:
            json.dump(self.opt.__dict__, f, indent=2)
            f.close()

    def load_options(self, load_path):
        prs = argparse.ArgumentParser()
        opt = prs.parse_args()
        with open(load_path, 'r') as f:
            opt.__dict__ = json.load(f)
        return opt

