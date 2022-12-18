import cv2
import numpy as np
from scipy.signal import convolve2d


# https://github.com/tkkcc/prior/blob/879a0b6c117c810776d8cc6b63720bf29f7d0cc4/util/gen_kernel.py
def motionblurkernel_synthesis(k_size=25):
    kdims = [k_size, k_size]
    x = randomTrajectory(250)
    k = None
    while k is None:
        k = kernelFromTrajectory(x)

    # center pad to kdims
    pad_width = ((kdims[0] - k.shape[0]) // 2, (kdims[1] - k.shape[1]) // 2)
    pad_width = [(pad_width[0],), (pad_width[1],)]

    if pad_width[0][0] < 0 or pad_width[1][0] < 0:
        k = k[0:k_size, 0:k_size]
    else:
        k = np.pad(k, pad_width, "constant")

    x1, x2 = k.shape
    if np.random.randint(0, 4) == 1:
        k = cv2.resize(k, (np.random.randint(x1, 5 * x1), np.random.randint(x2, 5 * x2)), interpolation=cv2.INTER_LINEAR)
        y1, y2 = k.shape
        k = k[(y1 - x1) // 2: (y1 - x1) // 2 + x1, (y2 - x2) // 2: (y2 - x2) // 2 + x2]

    if np.sum(k) < 0.1:
        k = fspecial_gauss(k_size, 0.1 + 6 * np.random.rand(1))

    k = k / np.sum(k)

    return k


def fspecial_gauss(size, sigma):
    x, y = np.mgrid[-size // 2 + 1: size // 2 + 1, -size // 2 + 1: size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()


def kernelFromTrajectory(x):
    h = 5 - np.log(np.random.rand()) / 0.15
    h = np.round(min([h, 27])).astype(int)
    h = h + 1 - h % 2
    w = h
    k = np.zeros((h, w))

    xmin = np.min(x[0])
    xmax = np.max(x[0])
    ymin = np.min(x[1])
    ymax = np.max(x[1])
    xthr = np.arange(xmin, xmax, (xmax - xmin) / w)
    ythr = np.arange(ymin, ymax, (ymax - ymin) / h)

    for i in range(1, xthr.size):
        for j in range(1, ythr.size):
            idx = (
                (x[0, :] >= xthr[i - 1])
                & (x[0, :] < xthr[i])
                & (x[1, :] >= ythr[j - 1])
                & (x[1, :] < ythr[j])
            )
            k[i - 1, j - 1] = np.sum(idx)
    if np.sum(k) == 0:
        return
    k = k / np.sum(k)
    k = convolve2d(k, fspecial_gauss(3, 1), "same")
    k = k / np.sum(k)
    return k


def randomTrajectory(t):
    x = np.zeros((3, t))
    v = np.random.randn(3, t)
    r = np.zeros((3, t))
    trv = 1
    trr = 2 * np.pi / t
    for t in range(1, t):
        F_rot = np.random.randn(3) / (t + 1) + r[:, t - 1]
        F_trans = np.random.randn(3) / (t + 1)
        r[:, t] = r[:, t - 1] + trr * F_rot
        v[:, t] = v[:, t - 1] + trv * F_trans
        st = v[:, t]
        st = rot3D(st, r[:, t])
        x[:, t] = x[:, t - 1] + st
    return x


def rot3D(x, r):
    Rx = np.array([[1, 0, 0], [0, np.cos(r[0]), -np.sin(r[0])], [0, np.sin(r[0]), np.cos(r[0])]])
    Ry = np.array([[np.cos(r[1]), 0, np.sin(r[1])], [0, 1, 0], [-np.sin(r[1]), 0, np.cos(r[1])]])
    Rz = np.array([[np.cos(r[2]), -np.sin(r[2]), 0], [np.sin(r[2]), np.cos(r[2]), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    x = R @ x
    return x


# https://github.com/assafshocher/BlindSR_dataset_generator
def gaussianblurkernel_synthesis(k_size=np.array([25, 25]), scale_factor=np.array([4, 4]), min_var=0.6, max_var=12., noise_level=0):
    # Set random eigen-vals (lambdas) and angle (theta) for COV matrix
    lambda_1 = min_var + np.random.rand() * (max_var - min_var)
    lambda_2 = min_var + np.random.rand() * (max_var - min_var)
    theta = np.random.rand() * np.pi  # random theta
    noise = 0 - noise_level + np.random.rand(*k_size) * noise_level * 2

    # Set COV matrix using Lambdas and Theta
    LAMBDA = np.diag([lambda_1, lambda_2])
    Q = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    SIGMA = Q @ LAMBDA @ Q.T
    INV_SIGMA = np.linalg.inv(SIGMA)[None, None, :, :]

    # Set expectation position (shifting kernel for aligned image)
    MU = k_size // 2 - 0.5 * (scale_factor - 1)
    MU = MU[None, None, :, None]

    # Create meshgrid for Gaussian
    [X, Y] = np.meshgrid(range(k_size[0]), range(k_size[1]))
    Z = np.stack([X, Y], 2)[:, :, :, None]

    # Calcualte Gaussian for every pixel of the kernel
    ZZ = Z - MU
    ZZ_t = ZZ.transpose(0, 1, 3, 2)
    raw_kernel = np.exp(-0.5 * np.squeeze(ZZ_t @ INV_SIGMA @ ZZ)) * (1 + noise)

    # shift the kernel so it will be centered ?

    # Normalize the kernel and return
    kernel = raw_kernel / np.sum(raw_kernel)

    return kernel


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    for i in range(1):
        # k = gaussianblurkernel_synthesis(noise_level=np.random.randint(0, 25))
        k = motionblurkernel_synthesis()
        # k = fspecial_gauss(25, 2)
        print(k.shape)
        plt.imshow(k, interpolation="nearest", cmap="gray")
        plt.show()
    # for i in range(10):
    #     x = randomTrajectory(250)
    #     # print(x.shape)
    #     # plt.plot(x[0])
    #     # plt.plot(x[1])
    #     # plt.plot(x[2])
    #     # plt.show()
    #     k = kernelFromTrajectory(x)
    #     print(k.shape)
