# coding=utf-8
from pathlib import Path
import os
from skimage import morphology
from lxml import etree
import cv2
from skimage.morphology import binary_dilation
from tqdm import tqdm
import numpy as np
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt

def normalize(x):
    '''
    normalize the image from 0~255 to 0 ~ 1
    :param x:
    :return:
    '''
    return x / 255.0


def ndvi(rgb, nir):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1] , rgb[:, :,
                                                          2]
    r, g, b = r / (r + b + g), g / (r + b + g), b / (r + b + g)

    exg = np.clip(2 * g - r - b, 0, 1) * 255.0
    return exg


def read_image_path(rgb_path, nir_path):
    '''
    read rgb, nir image path
    :param rgb_path:
    :param nir_path:
    :return:
    '''
    rgb_paths = set(
        [os.path.split(str(p))[-1] for p in sorted(Path(rgb_path).glob(
            '*.png'))])
    nir_paths = set(
        [os.path.split(str(p))[-1] for p in sorted(Path(nir_path).glob(
            '*.png'))])
    intersection = sorted(rgb_paths.intersection(nir_paths))

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

        # using morphology transformation to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        ndvi_image = cv2.morphologyEx(ndvi_image, cv2.MORPH_OPEN, kernel,
                                      iterations= 2)
        ndvi_image = cv2.morphologyEx(ndvi_image, cv2.MORPH_CLOSE, kernel,
                                      iterations = 2)

        ndvi_image = ndvi_image.astype(np.uint8)

        # 0 black, 255 white
        _, thre = cv2.threshold(ndvi_image, 13, 255, cv2.THRESH_BINARY)
        thre = thre > thre.mean()

        # remove small object
        thre = morphology.remove_small_objects(thre,min_size=200,
                                               connectivity=1)


        # thre = cv2.subtract(255, thre)
        # thre = cv2.adaptiveThreshold(ndvi_image, 255,
        #                          cv2.ADAPTIVE_THRESH_MEAN_C,
        #                       cv2.THRESH_BINARY, 11, 2)

        cv2.imwrite(os.path.join(ndvi_path, name), thre * 255)


def segment2boundingbox(mask_path, gt_path):
    filename = os.path.split(mask_path)[-1]
    annotation = etree.Element('annotation')
    etree.SubElement(annotation, 'filename').text = filename


    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255
    mask = cv2.dilate(mask, np.ones((3,3),np.uint8),iterations = 2)


    size = etree.SubElement(annotation, 'size')
    etree.SubElement(size, 'width').text = mask.shape[0]
    etree.SubElement(size, 'height').text = mask.shape[1]

    # annotation the different object with different color
    mask_label = label(mask)

    # bounding box
    props = regionprops(mask_label)

    # ground truth image
    gt_image = cv2.cvtColor(cv2.imread(gt_path), cv2.COLOR_BGR2GRAY)

    # differnt color correspond to differnt label
    print(gt_image, np.unique(gt_image))

    for prop in props:
        print('Found bbox', prop.bbox)
        ymin, xmin, ymax, xmax = prop.bbox

        gt_crop = gt_image[xmin:ymin, xmax:ymax]
        print(gt_crop)

        object = etree.SubElement(annotation, 'object')
        etree.SubElement(object, 'name').text = 'weed or crop'
        bndbox = etree.SubElement(object, 'bndbox')
        etree.SubElement(bndbox, 'xmin').text = xmin
        etree.SubElement(bndbox, 'xmax').text = xmax
        etree.SubElement(bndbox, 'ymin').text = ymin
        etree.SubElement(bndbox, 'ymax').text = ymax

#         corresponding to different label



# def segment2boundingbox(mask_path):
#     fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (15, 5))
#     img_0 = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255
#     img_0 = cv2.dilate(img_0, np.ones((3,3),np.uint8),iterations = 2)
#
#     #
#     lbl_0 = label(img_0)
#     props = regionprops(lbl_0)
#     boxs = img_0.copy() * 255.0
#     print(len(props))
#
#
#     for prop in props:
#         print('Found bbox', prop.bbox)
#         minr, minc, maxr, maxc = prop.bbox
#
#         cv2.rectangle(boxs, (minc, minr), (maxc,maxr),
#                       (255, 0, 0), 5)
#
#
#     ax1.imshow(img_0)
#     ax1.set_title('Image')
#
#     ax2.imshow(lbl_0)
#     ax2.set_title('Labeling')
#
#     ax3.imshow(boxs)
#     ax3.set_title('Image with derived bounding box')
#     plt.show()


if __name__ == '__main__':
    # paths = read_image_path('./data/rgb', './data/nir')
    # write2ndvi(paths)
    segment2boundingbox('./data/ndvi/masks_150528_frame1004.png')
