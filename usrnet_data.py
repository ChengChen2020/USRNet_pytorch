import os
import argparse
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import convolve

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader

import utils_image as util

from utils_kernel import motionblurkernel_synthesis
from utils_kernel import gaussianblurkernel_synthesis


class DatasetUSRNet(data.Dataset):
    """
    # -----------------------------------------
    # Get L/k(kernel)/sf(scale_factor)/sigma for USRNet.
    # Only "paths_H" and kernel is needed, synthesize L on-the-fly.
    # -----------------------------------------
    """
    def __init__(self, phase, path, batch_size=48, patch_size=96):
        super(DatasetUSRNet, self).__init__()
        self.phase = phase
        self.n_channels = 3
        self.sigma_max = 25  # Max noise level
        self.scales = [1, 2, 3, 4]
        self.sf_validation = 3  # Validation scale factor
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.kernels = loadmat(os.path.join('kernels', 'kernels_12.mat'))['kernels']  # For validation

        self.paths_H = util.get_image_paths(path)
        self.count = 0

    def __getitem__(self, index):

        H_path = self.paths_H[index]
        img_H = util.imread_uint(H_path, self.n_channels)

        if self.phase == 'train':

            # (1) Scale factor, ensure each batch only involves one scale factor
            if self.count % self.batch_size == 0:
                self.sf = np.random.choice(self.scales)
                self.count = 0
            self.count += 1
            H, W, _ = img_H.shape

            # (2) Patch crop
            rnd_h = np.random.randint(0, max(0, H - self.patch_size))
            rnd_w = np.random.randint(0, max(0, W - self.patch_size))
            patch_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            # (3) Augmentation
            mode = np.random.randint(0, 8)
            patch_H = util.augment_img(patch_H, mode=mode)

            # (4) Synthesis kernel
            r_value = np.random.randint(0, 8)
            if r_value > 4:
                # Motion blur
                k = motionblurkernel_synthesis(k_size=25)
            else:
                # Gaussian blur
                sf_k = np.random.choice(self.scales)
                k = gaussianblurkernel_synthesis(scale_factor=np.array([sf_k, sf_k]))
                mode_k = np.random.randint(0, 8)
                k = util.augment_img(k, mode=mode_k)

            noise_level = np.random.randint(0, self.sigma_max) / 255.0

            # (5) Degradation
            img_L = convolve(patch_H, np.expand_dims(k, axis=2), mode='wrap')  # Blur
            img_L = img_L[0::self.sf, 0::self.sf, ...]  # Downsample
            img_L = util.uint2single(img_L) + np.random.normal(0, noise_level, img_L.shape)  # AWGN
            img_H = patch_H

        else:
            # Validation
            k = self.kernels[0, 0].astype(np.float64)
            k /= np.sum(k)
            noise_level = 0.

            img_H = util.modcrop(img_H, self.sf_validation)

            # Degradation
            img_L = convolve(img_H, np.expand_dims(k, axis=2), mode='wrap')  # Blur
            img_L = img_L[0::self.sf_validation, 0::self.sf_validation, ...]  # Downsample
            img_L = util.uint2single(img_L) + np.random.normal(0, noise_level, img_L.shape)  # AWGN
            self.sf = self.sf_validation

        k = util.single2tensor3(np.expand_dims(np.float32(k), axis=2))
        img_H, img_L = util.uint2tensor3(img_H), util.single2tensor3(img_L)
        noise_level = torch.FloatTensor([noise_level]).view([1, 1, 1])

        return {'L': img_L, 'H': img_H, 'k': k, 'sigma': noise_level, 'sf': self.sf, 'path': H_path}

    def __len__(self):
        return len(self.paths_H)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--patch_size', type=int, default=96)
    opt = parser.parse_args()

    train_set = DatasetUSRNet('train', 'trainsets/train_combined', opt.batch_size, opt.patch_size)
    train_loader = DataLoader(train_set,
                              batch_size=opt.batch_size,
                              shuffle=True,
                              num_workers=2,
                              drop_last=True,
                              pin_memory=True)

    test_set = DatasetUSRNet('test', 'testsets/Set5', 1)
    test_loader = DataLoader(test_set, batch_size=1,
                             shuffle=False, num_workers=1,
                             drop_last=False, pin_memory=True)

    print(len(train_loader))
    print(len(test_loader))
    train_data = next(iter(train_loader))
    print(train_data['L'].shape)
    print(train_data['H'].shape)
    print(train_data['k'].shape)
    print(train_data['sigma'].shape)
    print(train_data['sf'][0])

    test_data = next(iter(test_loader))
    print(test_data['L'].shape)
    print(test_data['H'].shape)
    print(test_data['k'].shape)
    print(test_data['sigma'].shape)
    print(test_data['sf'][0])
    print(test_data['path'])
    print(os.path.basename(test_data['path'][0]))
