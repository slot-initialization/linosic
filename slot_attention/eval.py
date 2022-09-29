from dataset import *
from model import *
import torch
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torchmetrics as tom
import argparse
from tqdm import tqdm
import ari
from PIL import Image
parser = argparse.ArgumentParser()


parser.add_argument('--version', default='pseudoweights', type=str, help='Valid Versions: original, kmeans, pooling, context, pseudoweights, selfattention, trafo, mlp, direct')
parser.add_argument('--seed', default=4, type=int, help='random seed')
parser.add_argument('--batch_size', default=1, type=int, help='Batchsize of the test_dataset_loader')
parser.add_argument('--size', default=500, type=int, help='Dataset size for validation')  # 100
parser.add_argument('--num_slots', default=5, type=int, help='Number of slots in Slot Attention.')
parser.add_argument('--num_clusters', default=10, type=int, help='Number of clusters in KmeansPP.')
parser.add_argument('--num_iterations', default=3, type=int, help='Number of attention iterations.')
parser.add_argument('--hid_dim', default=64, type=int, help='hidden dimension size')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers for loading data')
parser.add_argument('--ds_root', default='./datasets/CLEVR/', type=str, help='dataset root, CHAIRS, CLEVR, MultiDsprites')
parser.add_argument('--data_set', default='CLEVR6', type=str, help='dataset version: CHAIRS, CLEVR, MDS')
parser.add_argument('--resolution', default=128, type=int, help='Resolution of latent space')
parser.add_argument('--checkpoint', default='.ckpt', type=str, help='Path to checkpoints file')
parser.add_argument('--parallel', default=1, type=int, help='If model WAS TRAINED in parallel batches')
parser.add_argument('--plot_save_root', default='./', type=str, help='dataset root, CHAIRS, CLEVR, MultiDsprites')
parser.add_argument('--sort_slots', default=0, type=int, help='sort slots')


def stack_images(img_stack: torch.tensor, fill=1, pad=1):
    N, C, H, W = img_stack.shape
    # img_stack = torch.nn.functional.pad(img_stack, pad=(pad, pad, pad, pad), mode='constant', value=fill)
    flat_img = torch.full((C, H+2*pad, (W+2*pad)*N), fill_value=fill).to(torch.float)
    for i in range(N):
        part_img = img_stack[i]
        flat_img[:, 1:H+1, 1 + i*(W+1): (i+1)*(W+1)] = part_img
    return flat_img


def image_to_gradient(image):
    N, C, H, W = image.shape
    image = image
    image = image - torch.min(image)
    image = image / torch.max(image)
    image = image.repeat(1, 3, 1, 1)
    high_value = torch.tensor([1., 1., 0.], device=image.device)[None, :, None, None].repeat(1, 1, H, W)
    low_value = torch.tensor([0., 0., 0.5], device=image.device)[None, :, None, None].repeat(1, 1, H, W)
    high_image = image * high_value
    low_image = (1-image) * low_value
    image = high_image + low_image
    #image = image - torch.min(image)
    # image = image / torch.max(image)

    return image


def plot_images(image: torch.tensor, save_path: str, gradient=False):
    assert len(image.shape) == 4
    if gradient:
        image = image_to_gradient((image + 1) / 2)
    else:
        image = (image + 1) / 2
    image = stack_images(image, fill=1, pad=1)

    transform = T.ToPILImage()
    image = transform(image)
    # torchvision.utils.save_image(image, opt.plot_save_root + save_name + str(j) + '.png')
    image.save(save_path)


def stickbreaking_process(recons, pred_mask, truth_mask):
    B, K, H, W, CM = pred_mask.size()
    _, _, _, _, CR = recons.size()
    _, TK, _, _ = truth_mask.size()
    truth_mask = truth_mask[..., None]
    truth_mask_sum = truth_mask.sum(dim=2).sum(dim=2).squeeze(0).repeat(1, K) + 1e-12
    overlap = torch.einsum('bthwc,bphwc->tp', truth_mask, pred_mask)
    # print(overlap)
    overlap = overlap / truth_mask_sum
    # print(overlap)
    max_overlap = torch.max(overlap, dim=0)[1]

    pred_mask_embed = torch.zeros((B, TK, K, H, W, CM), device=truth_mask.device)
    for i in range(pred_mask.shape[1]):
        pred_mask_embed[:, max_overlap[i], i, :, :, :] = pred_mask[:, i]  # * truth_mask[:, max_overlap[i]]
    recons_embed = recons[:, None, :, :, :, :].repeat(1, TK, 1, 1, 1, 1)
    # pred_mask_embed_rgb = pred_mask_embed.repeat(1, 1, 1, 1, 1, CR)
    #print(pred_mask_embed.shape)
    #print(torch.max(pred_mask_embed), torch.min(pred_mask_embed))
    recons_embed = pred_mask_embed * recons_embed
    pred_mask = pred_mask_embed.sum(dim=2)
    #print(pred_mask.shape)
    #print(torch.max(pred_mask), torch.min(pred_mask))
    recons = recons_embed.sum(dim=2)
    #print(recons.shape)
    #print(torch.max(recons), torch.min(recons))
    #print(torch.max(recons), torch.min(recons))
    #recons = recons.sum(dim=1)
    #recons_show = ((recons[0]+1) / 2).cpu().numpy()
    #plt.imshow(recons_show)
    #plt.show()
    #for i in range(TK):
    #    pred_mask_show = pred_mask[0, i, :, :].cpu().numpy()
    #    #print(truth_mask_fg_sum[0, i, :, :].sum())
    #    plt.imshow(pred_mask_show)
    #    plt.show()
    #    recons_show = ((recons[0, i]+1)/2).cpu().numpy()
    #    plt.imshow(recons_show)
    #    plt.show()
    return recons, pred_mask

device = torch.device('cuda')
opt = parser.parse_args()
print(opt)

# Hyperparameters.
torch.manual_seed(opt.seed)
batch_size = opt.batch_size
num_slots = opt.num_slots
num_iterations = opt.num_iterations
resolution = (opt.resolution, opt.resolution)
size = opt.size
# test_set = CLEVR('val')
val_set = DsCreator(split='val',
                     max_num_slots=opt.num_slots,
                     ds_root=opt.ds_root,
                     size=opt.size,
                     resolution=resolution,
                     mask=True)

val_dataloader = torch.utils.data.DataLoader(val_set,
                                             batch_size=opt.batch_size,
                                             shuffle=False,
                                             num_workers=opt.num_workers)
model = SlotAttentionAutoEncoder(resolution,
                                 opt.num_slots,
                                 opt.num_clusters,
                                 opt.num_iterations,
                                 opt.hid_dim,
                                 opt.version).to(device)
state_dict = torch.load(opt.checkpoint)
if opt.parallel == 1:
    model = torch.nn.DataParallel(model)
model.load_state_dict(state_dict=state_dict['model_state_dict'])
model.to(device)
model.eval()
#ckpt_fs = [f for f in os.listdir('./tmp/' + opt.dataset + '/' + opt.version + '/') if f.endswith('.ckpt')]
ssim = tom.StructuralSimilarityIndexMeasure().to('cuda')
psnr = tom.PeakSignalNoiseRatio().to('cuda')
lpips_vgg = tom.image.lpip.LearnedPerceptualImagePatchSimilarity(net_type='vgg').to('cuda')
lpips_alex = tom.image.lpip.LearnedPerceptualImagePatchSimilarity(net_type='alex').to('cuda')
results = []
len_val_dataloader = len(val_dataloader)
with torch.no_grad():
    # model = SlotAttentionAutoEncoder(resolution, num_slots, num_iterations, 64)
    ssim_v, psnr_v, lpips_vgg_v, lpips_alex_v, ari_v = 0, 0, 0, 0, 0
    j = 0
    for sample in tqdm(val_dataloader):
        image = sample['image'].to(device)
        mask_image_t = sample['mask'].to(device)
        recon_combined, recons, masks, slots = model(image)

        #if opt.sort_slots == 1:
        #    recons, masks = stickbreaking_process(recons, masks, mask_image_t)
        """Outcomment the needed part"""
        """If you want to get the high object count images go to dataset.py and search for the comment"""
        """To save the different images"""
        plot_images(recon_combined, opt.plot_save_root + 'recon_combined' + str(j) + '.png', gradient=False)
        plot_images(recons.permute(0, 1, 4, 2, 3).squeeze(0), opt.plot_save_root + 'recons' + str(j) + '.png', gradient=False)
        plot_images(masks.permute(0, 1, 4, 2, 3).squeeze(0), opt.plot_save_root + 'masks' + str(j) + '.png', gradient=True)
        """To save the Groundtruth image"""
        transform = T.ToPILImage()
        image = transform((image[0]+1)/2)
        image.save(opt.plot_save_root + 'ground_truth' + str(j) + '.png')
        j = j+1
        #print(recon_combined.shape, recons.shape, masks.shape)
        # Metrics
        """Evaluations"""
        ssim_v += ssim(preds=recon_combined, target=image).item()
        psnr_v += psnr(preds=recon_combined, target=image).item()
        lpips_vgg_v += lpips_vgg(recon_combined, image).item()
        lpips_alex_v += lpips_alex(recon_combined, image).item()
        ari_v += ari.ari_calc(mask_image_t, masks.squeeze(-1))
    results.append([ssim_v / len_val_dataloader, psnr_v / len_val_dataloader,
                    lpips_vgg_v / len_val_dataloader, lpips_alex_v / len_val_dataloader, ari_v / len_val_dataloader])
    #print(ssim_v / len_val_dataloader, psnr_v / len_val_dataloader,
    #                lpips_vgg_v / len_val_dataloader, lpips_alex_v / len_val_dataloader, ari_v / len_val_dataloader)
    print('------------------------------------------------------')
    print(opt.data_set, opt.version)
    print(results)

#with open('./tmp/' + opt.data_set + '/' + opt.version + '/' + 'eval_results.txt', 'w') as fp:
#    for item in results:
#        # write each item on a new line
#        fp.write("%s\n" % item)
#    print('Done')


