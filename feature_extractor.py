# coding=utf-8
import skimage
from cv2 import cv2
from skimage.feature import local_binary_pattern, hog
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


def create_surf_feature(files, hessian_threshold=50):
    arraies = []
    surf = cv2.xfeatures2d.SURF_create(hessian_threshold)
    for f in tqdm(files):
        rgb, nir = (f[..., 0:3] * 255).astype(np.uint8), (
                    f[..., -1] * 255).astype(np.uint8)
        kp, des = surf.detectAndCompute(rgb, None)
        print(kp)
        print(len(kp))


def create_lbp_texture_feature(files):
    arraies = []
    radius = 1
    n_point = radius * 8
    for f in tqdm(files, desc='lbp feature:'):
        f = (f * 255).astype(np.uint8)
        features = []
        for c in range(f.shape[-1]):
            lbp = local_binary_pattern(f[..., c], n_point, radius, 'default')
            max_bins = int(lbp.max() + 1)
            lbp_feature, _ = np.histogram(lbp, normed=True, bins=max_bins,
                                          range=(0, max_bins))
            features.extend(lbp_feature)
        arraies.append(features)
    return np.stack(arraies, axis = 0)

def create_color_feature(files):
    arraies = []
    for f in tqdm(files, desc='color feature:'):
        f = (f * 255).astype(np.uint8)
        features = []
        for c in range(f.shape[-1]):
            hist, _ = np.histogram(f[..., c], normed=True, bins=256,
                                range=(0, 256))
            features.extend(hist)
        arraies.append(features)
    return np.stack(arraies, axis = 0)

def create_hog_feature(files, orientations=8, pixels_per_cell=(16, 16),
                       cells_per_block=(1, 1)):
    arraies = []
    for f in tqdm(files, desc='hog feature:'):
        f = (f * 255).astype(np.uint8)
        features = []
        for c in range(f.shape[-1]):
            _, hog_image = hog(f[..., c], orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block,
                                               visualise=True,
                                               transform_sqrt=True)
            hist, _ = np.histogram(hog_image, normed=True, bins=256,
                                   range=(0, 256))
            features.extend(hist)
        arraies.append(features)
    return np.stack(arraies, axis = 0)

if __name__ == '__main__':
    data = np.load("data224.npz")
    x_train, y_train, x_test, y_test = data['x_train'], data['y_train'], data[
        'x_test'], data['y_test']

    # print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    # create_surf_feature(x_test[:5])
    # print(create_lbp_texture_feature(x_test[:5]).shape)
    # print(create_color_feature(x_test[:5]).shape)
    # print(create_hog_feature(x_train[:5]).shape)

    img = x_train[1]
    # plt.figure()
    # plt.imshow(img[..., 0:3][..., :: -1])
    # plt.show()
    #
    # plt.figure()
    # plt.imshow(img[..., -1], cmap='gray')
    # plt.show()

    for colour_channel in (0, 1, 2, 3):
        _, img[:, :, colour_channel] = hog(img[:, :, colour_channel], orientations=8, pixels_per_cell=(16, 16),
                                        cells_per_block=(1, 1),
                                        visualise=True,
                                        transform_sqrt=True)

    plt.figure()
    plt.imshow(img[:, :, 0:3])
    plt.show()

    plt.figure()
    plt.imshow(img[:, :, -1], cmap='gray')
    plt.show()

