import os
import cv2
import numpy as np
from scipy.io import loadmat

import utils_image as util

kernels = loadmat(os.path.join('kernels', 'kernels_12.mat'))['kernels']

np_kernels = []

for k_index in range(kernels.shape[1]):

    kernel = kernels[0, k_index].astype(np.float64)
    k_v = kernel / np.max(kernel) * 1.2
    k_v = util.single2uint(np.tile(k_v[..., np.newaxis], [1, 1, 3]))
    k_v = cv2.resize(k_v, (3 * k_v.shape[1], 3 * k_v.shape[0]), interpolation=cv2.INTER_NEAREST)

    np_kernels.append(k_v)

util.imsave(np.concatenate(np_kernels, axis=1), 'kernels/kernels.png')