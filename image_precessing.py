# coding=utf-8
from pathlib import Path
import os
import cv2
from tqdm import tqdm


def normalize(x):
    '''
    normalize the image from 0~255 to 0 ~ 1
    :param x:
    :return:
    '''
    return x/255.0

def ndvi(rgb, nir):
    return (nir + rgb[:, :, 0]) / (nir - rgb[:, :, 0])


def read_image_path(rgb_path, nir_path):
    '''
    read rgb, nir image path
    :param rgb_path:
    :param nir_path:
    :return:
    '''
    rgb_paths = set([os.path.split(str(p))[-1]for p in Path(rgb_path).glob(
        '*.png')])
    nir_paths = set([os.path.split(str(p))[-1]for p in Path(nir_path).glob(
        '*.png')])
    intersection = rgb_paths.intersection(nir_paths)

    paths = [(os.path.join(rgb_path, i), os.path.join(nir_path, i)) for i in
              intersection]
    return paths

def write2ndvi(paths):
    '''
    read the channel rgb, nir and convert to ndvi image
    :param paths:
    :return:
    '''

    ndvi_path = './data/ndvi'
    os.makedirs(ndvi_path, exist_ok=True)
    for rgb, nir in tqdm(paths):
        rgb_image = normalize(cv2.imread(rgb)[..., :: -1])
        nir_image = normalize(cv2.imread(nir, cv2.IMREAD_GRAYSCALE))

        ndvi_image = ndvi(rgb_image, nir_image)
        name = os.path.split(rgb)[-1]

        cv2.imwrite(os.path.join(ndvi_path, name), ndvi_image)




if __name__ == '__main__':
    paths = read_image_path('./data/rgb', './data/nir')
    write2ndvi(paths)
