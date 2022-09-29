import os
# import math
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.parallel import DataParallel, DistributedDataParallel

import utils_parameter
import utils_image as util
from usrnet_model import USRNet
from usrnet_data import DatasetUSRNet


def get_bare_model(network):
    """Get bare model, especially under wrapping with
    DistributedDataParallel or DataParallel.
    """
    if isinstance(network, (DataParallel, DistributedDataParallel)):
        network = network.module
    return network


def main(json_path='config.json'):

    opt = utils_parameter.parse(json_path, is_train=True)
    util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
    device = torch.device('cpu')

    model = USRNet()
    optimizer = Adam(model.parameters(), lr=1e-4,
                     betas=opt['train']['optimizer_betas'],
                     weight_decay=opt['train']['optimizer_wd'])
    scheduler = MultiStepLR(optimizer,
                            opt['train']['scheduler_milestones'],
                            opt['train']['scheduler_gamma']
                            )
    criterion = nn.L1Loss().to(device)

    # for epoch in range(1000000):
    #     pass

    train_loader = None
    test_loader = None
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = DatasetUSRNet(dataset_opt)
            # train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            # logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            train_loader = DataLoader(train_set,
                                      batch_size=dataset_opt['dataloader_batch_size'],
                                      shuffle=dataset_opt['dataloader_shuffle'],
                                      num_workers=2,
                                      drop_last=True,
                                      pin_memory=True)
        elif phase == 'test':
            test_set = DatasetUSRNet(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)

        else:
            assert False

    assert train_loader is not None and test_loader is not None

    current_step = 0
    border = opt['scale']

    for epoch in range(1):  # keep running

        model.train()

        for i, train_data in enumerate(train_loader):

            current_step += 1

            L = train_data['L'].to(device)  # low-quality image
            H = train_data['H'].to(device)
            k = train_data['k'].to(device)  # blur kernel
            sf = np.int(train_data['sf'][0, ...].squeeze().cpu().numpy())  # scale factor
            sigma = train_data['sigma'].to(device)  # noise level

            print(L.shape, H.shape, k.shape, sf, sigma.shape)
            optimizer.zero_grad()
            out = model(L, k, sf, sigma)
            loss = criterion(out, H)
            loss.backward()
            optimizer.step()
            print(out.shape)
            scheduler.step()
            break

        if current_step % opt['train']['checkpoint_test'] == 1:

            model.eval()

            avg_psnr = 0.0
            idx = 0

            for test_data in test_loader:
                idx += 1
                image_name_ext = os.path.basename(test_data['L_path'][0])
                img_name, ext = os.path.splitext(image_name_ext)

                img_dir = os.path.join(opt['path']['images'], img_name)
                util.mkdir(img_dir)

                # model.feed_data(test_data)

                L = test_data['L'].to(device)  # low-quality image
                H = test_data['H'].to(device)
                k = test_data['k'].to(device)  # blur kernel
                sf = np.int(test_data['sf'][0, ...].squeeze().cpu().numpy())  # scale factor
                sigma = test_data['sigma'].to(device)  # noise level

                print(L.shape, H.shape, k.shape, sf, sigma.shape)

                with torch.no_grad():
                    E = model(L, k, sf, sigma)

                E_img = util.tensor2uint(E.detach()[0].float().cpu())
                H_img = util.tensor2uint(H.detach()[0].float().cpu())

                # -----------------------
                # save estimated image E
                # -----------------------
                save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, current_step))
                util.imsave(E_img, save_img_path)

                # -----------------------
                # calculate PSNR
                # -----------------------
                current_psnr = util.calculate_psnr(E_img, H_img, border=border)

                # logger.info('{:->4d}--> {:>10s} | {:<4.2f}dB'.format(idx, image_name_ext, current_psnr))

                avg_psnr += current_psnr

            avg_psnr = avg_psnr / idx

            print(avg_psnr)


if __name__ == '__main__':
    main()
