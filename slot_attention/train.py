import argparse
import sys

from dataset import *
from model import *
from tqdm import tqdm
import time
import datetime
import torch.optim as optim
import torch
import torchmetrics as tom
from torch.utils.tensorboard import SummaryWriter
import ari

parser = argparse.ArgumentParser()
parser.add_argument('--resume', default=0, type=int, help='If you want to resume training')
parser.add_argument('--resume_path', default='./', type=str, help='The path to the checkpoint from which you want to resume')
parser.add_argument('--start_epoch', default=0, type=int, help='From which epoch you want to start')
parser.add_argument('--model_dir_root', default='./tmp/', type=str,
                    help='where to save models')
parser.add_argument('--data_set', default='CLEVR6', type=str,
                    help='All Datasets in the "datasets" folder.')
parser.add_argument('--version', default='mlp', type=str,
                    help='Valid Versions: original, kmeans, pooling, context, pseudoweights, selfattention, trafo, mlp, direct')
parser.add_argument('--seed', default=0, type=int,
                    help='random seed')
parser.add_argument('--resolution', default=128, type=int,
                    help='resolution of the images and the latent model.')
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--test_batch_size', default=100, type=int)
parser.add_argument('--train_set_size', default=100, type=int)
parser.add_argument('--test_set_size', default=100, type=int)
parser.add_argument('--num_slots', default=5, type=int,
                    help='Number of slots in Slot Attention.')
parser.add_argument('--num_clusters', default=10, type=int,
                    help='Number of clusters in KmeansPP.')
parser.add_argument('--num_iterations', default=3, type=int,
                    help='Number of attention iterations.')
parser.add_argument('--hid_dim', default=64, type=int,
                    help='hidden dimension size')
parser.add_argument('--learning_rate', default=0.0004, type=float)
parser.add_argument('--warmup_steps', default=10000, type=int,
                    help='Number of warmup steps for the learning rate.')
parser.add_argument('--decay_rate', default=0.5, type=float,
                    help='Rate for the learning rate decay.')
parser.add_argument('--decay_steps', default=100000, type=int,
                    help='Number of steps for the learning rate decay.')
parser.add_argument('--num_workers', default=4, type=int,
                    help='number of workers for loading data')
parser.add_argument('--num_epochs', default=1000, type=int,
                    help='number of epochs')
parser.add_argument('--parallel', default=1, type=int,
                    help='Run model in parallel batches on multiple gpus.')


opt = parser.parse_args()
print(opt)
resolution = (opt.resolution, opt.resolution)
torch.manual_seed(opt.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ds_root = './datasets/'+opt.data_set+'/'
path_to_test = opt.model_dir_root + opt.data_set + '/' + opt.version + '/'
path_to_best = opt.model_dir_root + opt.data_set + '/' + opt.version + '/best/'
if not os.path.isdir(path_to_test):
    os.makedirs(path_to_test, exist_ok=True)
if not os.path.isdir(path_to_best):
    os.makedirs(path_to_best, exist_ok=True)


ssim = tom.StructuralSimilarityIndexMeasure().to('cuda')
psnr = tom.PeakSignalNoiseRatio().to('cuda')
lpips_vgg = tom.image.lpip.LearnedPerceptualImagePatchSimilarity(net_type='vgg').to('cuda')
lpips_alex = tom.image.lpip.LearnedPerceptualImagePatchSimilarity(net_type='alex').to('cuda')
tb_writer = SummaryWriter(log_dir=path_to_test)


test_set = DsCreator(split='test',
                     max_num_slots=opt.num_slots,
                     ds_root=ds_root,
                     size=opt.test_set_size,
                     resolution=resolution,
                     mask=True)
train_set = DsCreator(split='train',
                      max_num_slots=opt.num_slots,
                      ds_root=ds_root,
                      size=opt.train_set_size,
                      resolution=resolution,
                      mask=False)
train_dataloader = torch.utils.data.DataLoader(train_set,
                                               batch_size=opt.batch_size,
                                               shuffle=True,
                                               num_workers=opt.num_workers)
test_dataloader = torch.utils.data.DataLoader(test_set,
                                              batch_size=opt.test_batch_size,
                                              shuffle=False,
                                              num_workers=opt.num_workers)


model = SlotAttentionAutoEncoder(resolution,
                                 opt.num_slots,
                                 opt.num_clusters,
                                 opt.num_iterations,
                                 opt.hid_dim,
                                 opt.version).to(device)
if opt.resume == 1:
    state_dict = torch.load(opt.resume_path)
    print(state_dict.keys())
    model.load_state_dict(state_dict=state_dict['model_state_dict'])
if opt.parallel == 1:
    model = torch.nn.DataParallel(model)
model.to(device)
# model.load_state_dict(torch.load('./tmp/model6.ckpt')['model_state_dict'])
criterion = nn.MSELoss()
params = [{'params': model.parameters()}]
optimizer = optim.Adam(params, lr=opt.learning_rate)
if opt.resume == 1:
    optimizer.load_state_dict(state_dict=state_dict['optimizer_state_dict'])

start = time.time()
i = 0
best_test_loss = None
for epoch in range(opt.start_epoch, opt.num_epochs):
    model.train()
    total_loss = 0
    for sample in tqdm(train_dataloader):
        i += 1
        if i < opt.warmup_steps:
            learning_rate = opt.learning_rate * (i / opt.warmup_steps)
        else:
            learning_rate = opt.learning_rate
        learning_rate = learning_rate * (opt.decay_rate ** (i / opt.decay_steps))
        optimizer.param_groups[0]['lr'] = learning_rate
        image = sample['image'].to(device)
        recon_combined, recons, masks, slots = model(image)
        loss = criterion(recon_combined, image)
        total_loss += loss.item()
        del recons, masks, slots
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    total_loss /= len(train_dataloader)
    print("Epoch: {}, Loss: {}, Time: {}".format(epoch, total_loss, datetime.timedelta(seconds=time.time() - start)))

    if not epoch % 20:
        # Save network
        model.eval()
        path_to_ckpt = opt.model_dir_root + opt.data_set + '/' + opt.version + '/' + str(epoch) + '.ckpt'
        with open(path_to_ckpt, 'w') as ckpt_f:
            torch.save({'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, path_to_ckpt)
            ckpt_f.close()
        # Evaluate network
        with torch.no_grad():
            ssim_v, psnr_v, lpips_vgg_v, lpips_alex_v, ari_v, test_loss = 0, 0, 0, 0, 0, 0
            for sample in tqdm(test_dataloader):
                test_image = sample['image'].to(device)
                mask_image_t = sample['mask'].to(device)
                recon_combined, _, mask_image_p, _ = model(test_image)
                ssim_v += ssim(preds=recon_combined, target=test_image).item()
                psnr_v += psnr(preds=recon_combined, target=test_image).item()
                lpips_vgg_v += lpips_vgg(recon_combined, test_image).item()
                lpips_alex_v += lpips_alex(recon_combined, test_image).item()
                ari_v += ari.ari_calc(mask_image_t, mask_image_p.squeeze(-1))
                test_loss = criterion(recon_combined, test_image).item()
            ssim_v, psnr_v, lpips_vgg_v, lpips_alex_v, ari_v, test_loss = ssim_v / len(test_dataloader), \
                                                                          psnr_v / len(test_dataloader), \
                                                                          lpips_vgg_v / len(test_dataloader), \
                                                                          lpips_alex_v / len(test_dataloader), \
                                                                          ari_v / len(test_dataloader), \
                                                                          test_loss
            tb_writer.add_scalar(tag='test_loss', scalar_value=test_loss, global_step=epoch)
            tb_writer.add_scalar(tag='ari', scalar_value=ari_v, global_step=epoch)
            tb_writer.add_scalar(tag='ssim', scalar_value=ssim_v, global_step=epoch)
            tb_writer.add_scalar(tag='psnr', scalar_value=psnr_v, global_step=epoch)
            tb_writer.add_scalar(tag='lpips_vgg', scalar_value=lpips_vgg_v, global_step=epoch)
            tb_writer.add_scalar(tag='lpips_alex', scalar_value=lpips_alex_v, global_step=epoch)
            tb_writer.flush()
            if (best_test_loss is None) or (test_loss < best_test_loss):
                try:
                    os.remove(path_to_best+os.listdir(path_to_best)[0])
                except:
                    pass
                path_to_best_ckpt = path_to_best + str(epoch) + '.ckpt'
                with open(path_to_best_ckpt, 'w') as best_f:
                    torch.save({'model_state_dict': model.module.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict()}, path_to_best_ckpt)
                    best_f.close()
                best_test_loss = test_loss
tb_writer.close()
