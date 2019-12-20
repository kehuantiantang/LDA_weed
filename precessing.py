# coding=utf-8
from pathlib import Path
import os
from skimage import morphology
from lxml import etree
import cv2
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
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
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
    rgb_paths = set([os.path.split(str(p))[-1] for p in
                     sorted(Path(rgb_path).glob('*.png'))])
    nir_paths = set([os.path.split(str(p))[-1] for p in
                     sorted(Path(nir_path).glob('*.png'))])
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
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        ndvi_image = cv2.morphologyEx(ndvi_image, cv2.MORPH_OPEN, kernel,
                                      iterations=2)
        ndvi_image = cv2.morphologyEx(ndvi_image, cv2.MORPH_CLOSE, kernel,
                                      iterations=2)

        ndvi_image = ndvi_image.astype(np.uint8)

        # 0 black, 255 white
        _, thre = cv2.threshold(ndvi_image, 13, 255, cv2.THRESH_BINARY)
        thre = thre > thre.mean()

        # remove small object
        thre = morphology.remove_small_objects(thre, min_size=200,
                                               connectivity=1)

        # thre = cv2.subtract(255, thre)
        # thre = cv2.adaptiveThreshold(ndvi_image, 255,
        #                          cv2.ADAPTIVE_THRESH_MEAN_C,
        #                       cv2.THRESH_BINARY, 11, 2)

        cv2.imwrite(os.path.join(ndvi_path, name), thre * 255)


def segment2boundingbox(mask_path, gt_path):
    '''
    read the ground truth file and locate the object which it corresponding
    label
    :param mask_path:
    :param gt_path:
    :return:
    '''

    filename = os.path.split(mask_path)[-1]
    xml_path = './data/xml'
    os.makedirs(xml_path, exist_ok=True)

    annotation = etree.Element('annotation')
    etree.SubElement(annotation, 'filename').text = str(filename)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)

    size = etree.SubElement(annotation, 'size')
    etree.SubElement(size, 'width').text = str(mask.shape[0])
    etree.SubElement(size, 'height').text = str(mask.shape[1])

    # annotation the different object with different color
    mask_label = label(mask)

    # bounding box
    props = regionprops(mask_label)

    # ground truth image
    gt_image = cv2.cvtColor(cv2.imread(gt_path), cv2.COLOR_BGR2GRAY)

    label_name = {'crop': '0', 'weed': '1'}

    # different color correspond to different label
    # print(gt_image, np.unique(gt_image))

    for prop in props:
        # print('Found bbox', prop.bbox)
        xmin, ymin, xmax, ymax = prop.bbox

        #  corresponding to different label
        gt_crop = gt_image[xmin:xmax, ymin:ymax]
        # print(np.unique(gt_crop))
        weed_num = gt_crop.reshape(-1, ).tolist().count(76)
        crop_num = gt_crop.reshape(-1, ).tolist().count(150)
        gt_label = 'weed' if weed_num > crop_num else 'crop'

        # write to xml object
        object = etree.SubElement(annotation, 'object')
        etree.SubElement(object, 'name').text = gt_label
        etree.SubElement(object, 'label').text = label_name[gt_label]
        bndbox = etree.SubElement(object, 'bndbox')
        etree.SubElement(bndbox, 'xmin').text = str(xmin)
        etree.SubElement(bndbox, 'xmax').text = str(xmax)
        etree.SubElement(bndbox, 'ymin').text = str(ymin)
        etree.SubElement(bndbox, 'ymax').text = str(ymax)

        # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (15, 5))  # ax1.imshow(mask_label)  # ax2.imshow(gt_image)  # ax3.imshow(gt_crop)  # plt.show()

    tree = etree.ElementTree(annotation)
    tree.write(os.path.join(xml_path, filename.split('.')[0] + '.xml'),
               pretty_print=True, xml_declaration=True, encoding='utf-8')


def object2patch(xml_path, img_shape):
    patch = './data/patch/'
    os.makedirs(patch, exist_ok=True)
    record_file = './data/patch/label.txt'

    records = []
    counter = 0
    for path in tqdm(sorted(Path(xml_path).glob('*.xml')), desc='object2patch'):

        xml = etree.parse(str(path))
        root = xml.getroot()

        filename = root.find('filename').text
        rgb_image = cv2.imread('./data/rgb/' + filename)
        nir_image = cv2.imread('./data/nir/' + filename, cv2.COLOR_BGR2GRAY)
        mask_image = cv2.imread('./data/ndvi/' + filename,
                                cv2.COLOR_BGR2GRAY) / 255

        for boxes in root.iter('object'):
            ymin, xmin, ymax, xmax = None, None, None, None

            for box in boxes.findall("bndbox"):
                ymin = int(box.find("ymin").text)
                xmin = int(box.find("xmin").text)
                ymax = int(box.find("ymax").text)
                xmax = int(box.find("xmax").text)

            label = boxes.find('label').text
            name = boxes.find('name').text

            # rgb, nir, mask
            save_filename = patch + '%08d_%s.png'
            counter += 1

            rgb_crop = cv2.resize(rgb_image[xmin:xmax, ymin:ymax, :] * np.stack(
                (mask_image[xmin:xmax, ymin:ymax],) * 3, axis=-1), img_shape)

            nir_crop = cv2.resize(
                nir_image[xmin:xmax, ymin:ymax] * mask_image[xmin:xmax,
                                                  ymin:ymax], img_shape)

            img_crop = np.concatenate([rgb_crop, np.expand_dims(nir_crop,
                                                                axis=-1)],
                                      axis = -1)

            # save file to rgb, nir, npy file
            img_filename = save_filename.replace('png', 'npy') % (counter,
                                                                  'all')
            rgb_filename = save_filename % (counter, 'rgb')
            nir_filename = save_filename %(counter, 'nir')

            np.save(img_filename, img_crop)
            cv2.imwrite(rgb_filename, rgb_crop)
            cv2.imwrite(nir_filename, nir_crop)
            line = (filename, img_filename, rgb_filename, nir_filename, name,
                    label)

            records.append(line)

    # write to label file
    with open(record_file, 'w') as f:
        for line in records:
            line_string = ','.join(line) + '\n'
            f.write(line_string)


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
    #
    # for path in tqdm(Path('./data/ndvi').glob('*.png')):
    #         segment2boundingbox(str(path),  str(path).replace('ndvi', 'gt'))

    object2patch('./data/xml', (224, 224))
