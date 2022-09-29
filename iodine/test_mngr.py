import torch
import torchmetrics as tom
from ari_metric import ARI
from torch.nn import DataParallel
from tqdm import tqdm
from utils.parallel_utils import *
from utils.test_utils import *


class TestManagement:
    def __init__(self, options, data_loader):
        self.opt = options.opt
        self.save_generated_images = self.opt.save_generated_images
        # Used Metrics
        self.ssim = tom.StructuralSimilarityIndexMeasure().to(self.opt.device_type)
        self.psnr = tom.PeakSignalNoiseRatio().to(self.opt.device_type)
        self.lpips_vgg = tom.image.lpip.LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(self.opt.device_type)
        self.lpips_alex = tom.image.lpip.LearnedPerceptualImagePatchSimilarity(net_type='alex').to(self.opt.device_type)
        self.ari = ARI()

        self.data_loader = data_loader
        self.length_dl = len(self.data_loader)

        self.device = self.opt.device_type

    def test_model(self, model):
        model.eval()
        # testing_model = model.module if isinstance(model, DataParallel) else model
        metric_dict = {'ssim': 0.,
                       'psnr': 0.,
                       'lpips_vgg': 0.,
                       'lpips_alex': 0.,
                       'ari': 0.,
                       'test_loss': 0.}
        save_amount = self.save_generated_images
        img_dict = {'pred_img': [],
                    'pred_mask': [],
                    'pred_slots': [],
                    'truth_img': [],
                    'truth_mask': []}
        for sample in tqdm(self.data_loader):
            test_image = sample['image'].to(self.device)
            mask_image_t = sample['mask'].to(self.device)

            pred, pred_mask, mean = model.reconstruct(test_image)
            pred, pred_mask, mean = pred.detach(), pred_mask.detach(), mean.detach()
            metric_dict['ssim'] += self.ssim(preds=pred, target=test_image)
            metric_dict['psnr'] += self.psnr(preds=pred, target=test_image)
            metric_dict['lpips_vgg'] += self.lpips_vgg(pred, test_image)
            metric_dict['lpips_alex'] += self.lpips_alex(pred, test_image)
            metric_dict['ari'] += self.ari.ari_calc(mask_image_t.clone(), pred_mask.squeeze(2))
            metric_dict['test_loss'] += model(test_image).detach()
            model.zero_grad(set_to_none=True)

            if save_amount:
                img_dict['pred_img'].append(pred[:1])
                img_dict['pred_mask'].append(pred_mask[0])
                img_dict['pred_slots'].append(mean[0])
                img_dict['truth_img'].append(test_image[:1])
                img_dict['truth_mask'].append(mask_image_t[0][:, None, ...])
                save_amount -= 1
        self.ssim.reset()  # Needed, so that ssim does not litter memory

        for key in metric_dict.keys():
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(metric_dict[key])
            metric_dict[key] = metric_dict[key].item() / (self.length_dl * len(self.opt.device_ids))
        if self.save_generated_images:
            img_dict_length = len(img_dict['pred_img'])
            for i in range(img_dict_length):
                if self.opt.sort_slots:
                    img_dict['pred_slots'][i], \
                    img_dict['pred_mask'][i] = stickbreaking_process(recons=img_dict['pred_slots'][i],
                                                                     pred_mask=img_dict['pred_mask'][i],
                                                                     truth_mask=img_dict['truth_mask'][i])

                img_dict['pred_img'][i] = plot_images(image=img_dict['pred_img'][i], gradient=False, invert=True)
                img_dict['pred_mask'][i] = plot_images(image=img_dict['pred_mask'][i], gradient=True, invert=False)
                img_dict['pred_slots'][i] = plot_images(image=img_dict['pred_slots'][i], gradient=False, invert=True)
                img_dict['truth_img'][i] = plot_images(image=img_dict['truth_img'][i], gradient=False, invert=True)
                img_dict['truth_mask'][i] = plot_images(image=img_dict['truth_mask'][i], gradient=True, invert=False)
        model.train()
        return metric_dict, img_dict










