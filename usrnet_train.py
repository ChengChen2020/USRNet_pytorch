import os
import time
import argparse
import functools
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.parallel import DataParallel, DistributedDataParallel

import utils_image as util
from usrnet_model import USRNet
from usrnet_data import DatasetUSRNet

CUDA_VISIBLE_DEVICES = 3


def get_bare_model(network):
    """Get bare model, especially under wrapping with
    DistributedDataParallel or DataParallel.
    """
    if isinstance(network, (DataParallel, DistributedDataParallel)):
        network = network.module
    return network


def save_network(save_dir, network, network_label, iter_label):
    save_filename = '{}_{}.pth'.format(iter_label, network_label)
    save_path = os.path.join(save_dir, save_filename)
    network = get_bare_model(network)
    state_dict = network.state_dict()
    for key, param in state_dict.items():
        state_dict[key] = param.cpu()
    torch.save(state_dict, save_path)


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=0.2)
        if m.bias is not None:
            m.bias.data.zero_()


def main():

    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    path_task = os.path.join('train_log', time.strftime("%Y%m%d-%H%M%S"))
    path_model = os.path.join(path_task, 'models')
    path_image = os.path.join(path_task, 'images')

    util.mkdirs(['model_zoo', path_task, path_model, path_image])

    torch.cuda.set_device(1)
    print('Current device:', torch.cuda.current_device())
    device = torch.device('cuda')

    device_ids = [1, 2]
    model = nn.DataParallel(USRNet(), device_ids=device_ids)
    model = model.to(device)
    model.apply(init_weights)

    optimizer = Adam(model.parameters(), lr=1e-4, betas=[0.9, 0.999])
    scheduler = MultiStepLR(optimizer,
                            milestones=[10000, 20000, 30000, 40000],
                            gamma=0.5
                            )
    criterion = nn.L1Loss().to(device)
    train_set = DatasetUSRNet('train', opt.train_path, opt.batch_size, opt.patch_size)
    train_loader = DataLoader(train_set,
                              batch_size=opt.batch_size,
                              shuffle=True,
                              num_workers=2,
                              drop_last=True,
                              pin_memory=True)
    valid_set = DatasetUSRNet('test', opt.valid_path, 1)
    valid_loader = DataLoader(valid_set, batch_size=1,
                              shuffle=False, num_workers=1,
                              drop_last=False, pin_memory=True)
    print("Train Dataset size:", len(train_loader.dataset))
    print("Train Loader size:", len(train_loader))
    print("Valid Dataset size:", len(valid_loader.dataset))
    print("Valid Loader size:", len(valid_loader))
    print("Data Loader successful!")

    current_step = 0

    for epoch in tqdm(range(1000)):

        for i, train_data in enumerate(train_loader):

            model.train()
            current_step += 1

            L = train_data['L'].to(device)
            H = train_data['H'].to(device)
            k = train_data['k'].to(device)
            sf = int(train_data['sf'][0, ...].squeeze().cpu().numpy())
            sigma = train_data['sigma'].to(device)

            optimizer.zero_grad()
            out = model(L, k, sf, sigma)
            loss = criterion(out, H)
            loss.backward()

            optimizer.step()
            scheduler.step()

            if current_step % opt.checkpoint_print == 0:
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step,
                                                                          scheduler.get_last_lr()[0])
                message += '{:s}: {:.3e} '.format('loss', loss.item())
                print(message)

            if current_step % opt.checkpoint_save == 0:
                save_network(path_model, model, 'USRNet', current_step)

            if current_step % opt.checkpoint_test == 0:
                model.eval()
                avg_psnr = 0.0
                idx = 0

                for valid_data in valid_loader:
                    idx += 1
                    image_name_ext = os.path.basename(valid_data['path'][0])
                    img_name, _ = os.path.splitext(image_name_ext)

                    img_dir = os.path.join(path_image, img_name)
                    util.mkdir(img_dir)

                    L = valid_data['L'].to(device)
                    H = valid_data['H'].to(device)
                    k = valid_data['k'].to(device)
                    sf = int(valid_data['sf'][0, ...].squeeze().cpu().numpy())
                    sigma = valid_data['sigma'].to(device)

                    with torch.no_grad():
                        E = model(L, k, sf, sigma)

                    E_img = util.tensor2uint(E.detach()[0].float().cpu())
                    H_img = util.tensor2uint(H.detach()[0].float().cpu())

                    # Save estimated image E
                    save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, current_step))
                    util.imsave(E_img, save_img_path)

                    # PSNR
                    current_psnr = util.calculate_psnr(E_img, H_img, border=sf ** 2)
                    avg_psnr += current_psnr

                avg_psnr = avg_psnr / idx
                print('<epoch:{:3d}, iter:{:8,d}, Average PSNR : {:<.2f}dB\n'.format(epoch, current_step, avg_psnr))

    print('Saving the final model.')
    save_network('model_zoo', model, 'USRNet', 'latest')
    print('End of training.')


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--patch_size', type=int, default=96)
parser.add_argument('--train_path', type=str, default='trainsets/train_combined')
parser.add_argument('--valid_path', type=str, default='testsets/Set5')
parser.add_argument('--checkpoint_print', type=int, default=50)
parser.add_argument('--checkpoint_save', type=int, default=500)
parser.add_argument('--checkpoint_test', type=int, default=500)
opt = parser.parse_args()


if __name__ == '__main__':
    main()
