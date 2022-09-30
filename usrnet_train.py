import os
import functools
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import init
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


def save_network(save_dir, network, network_label, iter_label):
    save_filename = '{}_{}.pth'.format(iter_label, network_label)
    save_path = os.path.join(save_dir, save_filename)
    network = get_bare_model(network)
    state_dict = network.state_dict()
    for key, param in state_dict.items():
        state_dict[key] = param.cpu()
    torch.save(state_dict, save_path)


def init_weights(net, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
    """
    # Kai Zhang, https://github.com/cszn/KAIR
    #
    # Args:
    #   init_type:
    #       default, none: pass init_weights
    #       normal; normal; xavier_normal; xavier_uniform;
    #       kaiming_normal; kaiming_uniform; orthogonal
    #   init_bn_type:
    #       uniform; constant
    #   gain:
    #       0.2
    """
    def init_fn(m, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
        classname = m.__class__.__name__

        if classname.find('Conv') != -1 or classname.find('Linear') != -1:

            if init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)

            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_type))

            if m.bias is not None:
                m.bias.data.zero_()

        elif classname.find('BatchNorm2d') != -1:

            if init_bn_type == 'uniform':  # preferred
                if m.affine:
                    init.uniform_(m.weight.data, 0.1, 1.0)
                    init.constant_(m.bias.data, 0.0)
            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_bn_type))

    print('Initialization method [{:s} + {:s}], gain is [{:.2f}]'.format(init_type, init_bn_type, gain))
    fn = functools.partial(init_fn, init_type=init_type, init_bn_type=init_bn_type, gain=gain)
    net.apply(fn)


def main(json_path='config.json'):

    opt = utils_parameter.parse(json_path, is_train=True)
    util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
    device = torch.device('cpu')

    # if 'tiny' in model_name:
    #     model = net(n_iter=6, h_nc=32, in_nc=4, out_nc=3, nc=[16, 32, 64, 64],
    #                 nb=2, act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose")
    # else:
    #     model = net(n_iter=8, h_nc=64, in_nc=4, out_nc=3, nc=[64, 128, 256, 512],
    #                 nb=2, act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose")

    model = USRNet().to(device)
    opt_net = opt['net']
    opt_train = opt['train']
    init_weights(model,
                 init_type=opt_net['init_type'],
                 init_bn_type=opt_net['init_bn_type'],
                 gain=opt_net['init_gain'])

    optimizer = Adam(model.parameters(), lr=1e-4,
                     betas=opt_train['optimizer_betas'],
                     weight_decay=opt_train['optimizer_wd'])
    scheduler = MultiStepLR(optimizer,
                            opt_train['scheduler_milestones'],
                            opt_train['scheduler_gamma']
                            )
    criterion = nn.L1Loss().to(device)

    train_loader = test_loader = None
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

    log_dict = OrderedDict()
    for epoch in range(1):  # keep running

        model.train()

        for i, train_data in enumerate(train_loader):

            current_step += 1

            L = train_data['L'].to(device)  # low-quality image
            H = train_data['H'].to(device)
            k = train_data['k'].to(device)  # blur kernel
            sf = np.int(train_data['sf'][0, ...].squeeze().cpu().numpy())  # scale factor
            sigma = train_data['sigma'].to(device)  # noise level

            optimizer.zero_grad()
            out = model(L, k, sf, sigma)
            loss = criterion(out, H)
            loss.backward()

            log_dict['loss'] = loss.item()

            optimizer.step()
            scheduler.step()

        if current_step % opt['train']['checkpoint_print'] == 0:
            message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step,
                                                                      scheduler.get_lr())
            for k, v in log_dict.items():  # merge log information into message
                message += '{:s}: {:.3e} '.format(k, v)
            print(message)

        if current_step % opt_train['checkpoint_save'] == 0:
            model.save(current_step)

        if current_step % opt_train['checkpoint_test'] == 1:

            model.eval()

            avg_psnr = 0.0
            idx = 0

            for test_data in test_loader:
                idx += 1
                image_name_ext = os.path.basename(test_data['L_path'][0])
                img_name, ext = os.path.splitext(image_name_ext)

                img_dir = os.path.join(opt['log_path']['images'], img_name)
                util.mkdir(img_dir)

                L = test_data['L'].to(device)  # low-quality image
                H = test_data['H'].to(device)
                k = test_data['k'].to(device)  # blur kernel
                sf = np.int(test_data['sf'][0, ...].squeeze().cpu().numpy())  # scale factor
                sigma = test_data['sigma'].to(device)  # noise level

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

    model.save('latest')


if __name__ == '__main__':
    main()
