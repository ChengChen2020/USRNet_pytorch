import os
import argparse
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import convolve
from collections import OrderedDict

import torch

import utils_image as util
from usrnet_model import USRNet


def downsample_np(x, sf=3, center=False):
    st = (sf - 1) // 2 if center else 0
    return x[st::sf, st::sf, ...]


def main(model_name, testset_name):
    device = torch.device('cuda')

    model = USRNet().to(device)
    model_path = os.path.join('model_zoo', model_name + '.pth')
    kernels = loadmat(os.path.join('kernels', 'kernels_12.mat'))['kernels']
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()

    result_name = testset_name + '_' + model_name

    L_path = os.path.join('testsets', testset_name)  # L_path and H_path, fixed, for Low-quality images
    E_path = os.path.join('results', result_name)  # E_path, fixed, for Estimated images
    util.mkdir(E_path)

    L_paths = util.get_image_paths(L_path)

    noise_level_img = 0
    noise_level_model = noise_level_img

    save_L = save_E = False

    for sf in [2, 3, 4]:

        for k_index in range(kernels.shape[1]):
            test_results = OrderedDict()
            test_results['psnr'] = []
            kernel = kernels[0, k_index].astype(np.float64)

            # util.surf(kernel)

            idx = 0

            for img in L_paths:

                # --------------------------------
                # (1) classical degradation, img_L
                # --------------------------------
                idx += 1
                img_name, ext = os.path.splitext(os.path.basename(img))
                img_H = util.imread_uint(img, n_channels=3)  # HR image, int8
                img_H = util.modcrop(img_H, np.lcm(sf, 8))  # modcrop

                # generate degraded LR image
                img_L = convolve(img_H, kernel[..., np.newaxis], mode='wrap')  # blur
                img_L = downsample_np(img_L, sf, center=False)  # downsample, standard s-fold downsampler

                img_L = util.uint2single(img_L)  # uint2single

                np.random.seed(0)  # for reproducibility
                img_L += np.random.normal(0, noise_level_img, img_L.shape)  # add AWGN

                x = util.single2tensor4(img_L)

                k = util.single2tensor4(kernel[..., np.newaxis])
                sigma = torch.tensor(noise_level_model).float().view([1, 1, 1, 1])
                [x, k, sigma] = [el.to(device) for el in [x, k, sigma]]

                # --------------------------------
                # (2) inference
                # --------------------------------
                x = model(x, k, sf, sigma)

                # --------------------------------
                # (3) img_E
                # --------------------------------
                img_E = util.tensor2uint(x)

                if save_E:
                    util.imsave(img_E, os.path.join(E_path, img_name + '_x' + str(sf) + '_k' + str(
                        k_index + 1) + '_' + model_name + '.png'))

                # --------------------------------
                # (4) img_LEH
                # --------------------------------
                img_L = util.single2uint(img_L)
                # if save_LEH:
                #     k_v = kernel / np.max(kernel) * 1.2
                #     k_v = util.single2uint(np.tile(k_v[..., np.newaxis], [1, 1, 3]))
                #     k_v = cv2.resize(k_v, (3 * k_v.shape[1], 3 * k_v.shape[0]), interpolation=cv2.INTER_NEAREST)
                #     img_I = cv2.resize(img_L, (sf * img_L.shape[1], sf * img_L.shape[0]), interpolation=cv2.INTER_NEAREST)
                #     img_I[:k_v.shape[0], -k_v.shape[1]:, :] = k_v
                #     img_I[:img_L.shape[0], :img_L.shape[1], :] = img_L
                #     util.imshow(np.concatenate([img_I, img_E, img_H], axis=1),
                #                 title='LR / Recovered / Ground-truth') if show_img else None
                #     util.imsave(np.concatenate([img_I, img_E, img_H], axis=1),
                #                 os.path.join(E_path, img_name + '_x' + str(sf) + '_k' + str(k_index + 1) + '_LEH.png'))

                if save_L:
                    util.imsave(img_L,
                                os.path.join(E_path, img_name + '_x' + str(sf) + '_k' + str(k_index + 1) + '_LR.png'))

                psnr = util.calculate_psnr(img_E, img_H, border=sf ** 2)  # change with your own border
                test_results['psnr'].append(psnr)
                # logger.info(
                #     '{:->4d}--> {:>10s} -- x{:>2d} --k{:>2d} PSNR: {:.2f}dB'.format(idx, img_name + ext, sf, k_index, psnr))

            ave_psnr_k = sum(test_results['psnr']) / len(test_results['psnr'])
            print('------> Average PSNR(RGB) of ({}) scale factor: ({}), kernel: ({}) sigma: ({}): {:.2f} dB'.format(
                testset_name, sf, k_index + 1, noise_level_model, ave_psnr_k))


parser = argparse.ArgumentParser()
parser.add_argument('--testset_name', type=str, default='BSD68')
parser.add_argument('--model_name', type=str, default='latest')
opt = parser.parse_args()

if __name__ == '__main__':
    main(opt.model_name, opt.testset_name)
