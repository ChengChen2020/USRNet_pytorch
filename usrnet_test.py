import os
import cv2
import argparse
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import convolve
from collections import OrderedDict

import torch

import utils_image as util
from usrnet_model import USRNet


def main(model_name, testset_name):
    device = torch.device('cuda')

    model = USRNet().to(device)
    model_path = os.path.join('model_zoo', model_name + '.pth')
    kernels = loadmat(os.path.join('kernels', 'kernels_12.mat'))['kernels']
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()

    result_name = testset_name + '_' + model_name

    L_path = os.path.join('testsets', testset_name)  # L_path and H_path
    E_path = os.path.join('test_log', result_name)
    if opt.save_LEH:
        util.mkdir(E_path)

    L_paths = util.get_image_paths(L_path)

    for (noise_level, sf) in [(0.0, 2), (0.0, 3), (0.01, 3), (0.03, 3), (0.0, 4)]:

        for k_index in range(kernels.shape[1]):
            test_results = OrderedDict()
            test_results['psnr'] = []
            kernel = kernels[0, k_index].astype(np.float64)

            # util.surf(kernel)  # Visualize kernel in 3D

            for img in L_paths:

                np.random.seed(0)

                # (1) Degradation
                img_name, _ = os.path.splitext(os.path.basename(img))

                img_H = util.imread_uint(img, n_channels=3)
                img_H = util.modcrop(img_H, np.lcm(sf, 8))

                img_L = convolve(img_H, kernel[..., np.newaxis], mode='wrap')  # Blur
                img_L = img_L[0::sf, 0::sf, ...]  # Downsample
                img_L = util.uint2single(img_L) + np.random.normal(0, noise_level, img_L.shape)  # AWGN

                x = util.single2tensor4(img_L)
                k = util.single2tensor4(kernel[..., np.newaxis])
                sigma = torch.tensor(noise_level).float().view([1, 1, 1, 1])
                [x, k, sigma] = [ele.to(device) for ele in [x, k, sigma]]

                # (2) Inference
                x = model(x, k, sf, sigma)
                img_E = util.tensor2uint(x)

                # (3) PSNR
                psnr = util.calculate_psnr(img_E, img_H, border=sf ** 2)
                test_results['psnr'].append(psnr)

                # (4) Save
                img_L = util.single2uint(img_L)
                if opt.save_LEH:
                    k_v = kernel / np.max(kernel) * 1.2
                    k_v = util.single2uint(np.tile(k_v[..., np.newaxis], [1, 1, 3]))
                    k_v = cv2.resize(k_v, (3 * k_v.shape[1], 3 * k_v.shape[0]), interpolation=cv2.INTER_NEAREST)
                    img_I = cv2.resize(img_L, (sf * img_L.shape[1], sf * img_L.shape[0]), interpolation=cv2.INTER_NEAREST)
                    img_I[:k_v.shape[0], -k_v.shape[1]:, :] = k_v  # Kernel at top right corner
                    img_I[:img_L.shape[0], :img_L.shape[1], :] = img_L
                    util.imsave(np.concatenate([img_I, img_E, img_H], axis=1),
                                os.path.join(E_path, img_name + '_x' + str(sf) + '_k' + str(k_index + 1) + '_n' + str(noise_level) + '_LEH.png'))

            ave_psnr_k = sum(test_results['psnr']) / len(test_results['psnr'])
            print('------> Average PSNR(RGB) of ({}) scale factor: ({}), kernel: ({}) sigma: ({}): {:.2f} dB'.format(
                testset_name, sf, k_index + 1, noise_level, ave_psnr_k))


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='latest')
parser.add_argument('--testset_name', type=str, default='BSD68')
parser.add_argument('--save_LEH', action='store_true')
opt = parser.parse_args()

if __name__ == '__main__':
    main(opt.model_name, opt.testset_name)
