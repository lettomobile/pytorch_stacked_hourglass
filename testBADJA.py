import json
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
import numpy as np

import utils.img
from utils.group import HeatmapParser
import BADJA.ref as ds

parser = HeatmapParser()


def post_process(det, mat_, trainval, c=None, s=None, resolution=None):
    mat = np.linalg.pinv(np.array(mat_).tolist() + [[0, 0, 1]])[:2]
    res = det.shape[1:3]
    cropped_preds = parser.parse(np.float32([det]))[0]

    if len(cropped_preds) > 0:
        cropped_preds[:, :, :2] = utils.img.kpt_affine(cropped_preds[:, :, :2] * 4, mat)  # size 1x16x3

    preds = np.copy(cropped_preds)
    ##for inverting predictions from input res on cropped to original image
    if trainval != 'cropped':
        for j in range(preds.shape[1]):
            preds[0, j, :2] = utils.img.transform(preds[0, j, :2], c, s, resolution, invert=1)
    return preds


def inference(img, func, config):
    """
    forward pass at test time
    calls post_process to post process results
    """

    height, width = img.shape[0:2]
    center = (width / 2, height / 2)
    scale = max(height, width) / 200
    res = (config['train']['input_res'], config['train']['input_res'])

    mat_ = utils.img.get_transform(center, scale, res)[:2]
    inp = img / 255

    def array2dict(tmp):
        return {
            'det': tmp[0][:, :, :16],
        }

    tmp1 = array2dict(func([inp]))
    tmp2 = array2dict(func([inp[:, ::-1]]))

    tmp = {}
    for ii in tmp1:
        tmp[ii] = np.concatenate((tmp1[ii], tmp2[ii]), axis=0)

    det = tmp['det'][0, -1] + tmp['det'][1, -1, :, :, ::-1][ds.flipped_parts['mpii']]
    if det is None:
        return [], []
    det = det / 2

    det = np.minimum(det, 1)

    return post_process(det, mat_, 'valid', res)


def get_img():
    f = open('BADJA\joint_annotations\dog.json', 'r')
    # Return the json file as a dictionary
    train_file = json.load(f)
    for i in train_file:
        image_path = 'BADJA\%s' % (i['segmentation_path'])
        # print(image_path)
        # image = mpimg.imread(image_path)
        # image_plot = plt.imshow(image)
        # plt.show()

        ## img
        orig_img = cv2.imread(image_path)[:, :, ::-1]

        ## kp
        key_points = i['joints']
        visibility = i['visibility']

        yield key_points, orig_img


def main():
    from train import init
    func, config = init()

    def runner(imgs):
        return func(0, config, 'inference', imgs=torch.Tensor(np.float32(imgs)))['preds']

    def do(img):
        ans = inference(img, runner, config)
        # ...

    gts = []
    preds = []
    normalizing = []

    for anns, img in get_img():
        gts.append(anns)
        preds = do(img)


if __name__ == '__main__':
    main()
