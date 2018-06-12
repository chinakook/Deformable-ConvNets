# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Haozhi Qi
# --------------------------------------------------------														  
import _init_paths
from utils import image
import cv2
import argparse
import pprint
import os
import sys
from config.config import config, update_config
from bbox.bbox_transform import bbox_pred, clip_boxes

import time
import cv2
import numpy as np
import mxnet as mx
from nms.nms import py_nms_wrapper, py_softnms_wrapper
import logging
import shutil


from symbols import *
from core.loader import PyramidAnchorIterator
from core import callback, metric
from core.module import MutableModule
from utils.create_logger import create_logger
from utils.load_data import load_gt_roidb, merge_roidb, filter_roidb
from utils.load_model import load_param
from utils.PrefetchingIter import PrefetchingIter
from utils.lr_scheduler import WarmupMultiFactorScheduler
IMG_H = 1024
IMG_W = 1024
im_shape = [IMG_H,IMG_W] 
def parse_args():
    parser = argparse.ArgumentParser(description='Train Faster-RCNN network')
    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--frequent', help='frequency of logging', default=config.default.frequent, type=int)
    args = parser.parse_args()
    return args

args = parse_args()
if __name__ == '__main__':
    #testdir = '/mnt/6B133E147DED759E/2016_01_18_07_01_01'
    testdir = '/home/caizhendong/git/Deformable-ConvNets/data/plates/train/JPEGImages'
    files = [i for i in os.listdir(testdir) if i.endswith('.jpg')]
    sym_instance = eval(config.symbol + '.' + config.symbol)()
    sym = sym_instance.get_symbol(config, is_train=False)
    save_dict = mx.nd.load('%s-%04d.params' % ("/home/caizhendong/git/Deformable-ConvNets/output/fpn/coco/traffic_sign/train/fpn_traffic_sign", 7))
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    mod = mx.mod.Module(symbol=sym, context=mx.gpu(0), data_names=['data', 'im_info'], label_names=None)
    mod.bind(for_training=False, data_shapes=[('data', (1, 3, IMG_H, IMG_W)), ('im_info', (1, 3))], label_shapes=None, force_rebind=False)
    mod.set_params(arg_params=arg_params, aux_params=aux_params, force_init=False)
    rimg = cv2.imread("/home/caizhendong/git/Deformable-ConvNets/data/traffic_sign/train/JPEGImages/000000-1-0311170000-0000011.jpg")
    img = cv2.resize(rimg, (IMG_H,IMG_W))
    im_tensor = image.transform(img, [103.53, 116.28, 123.675], 0.017)
    im_info = np.array([[  IMG_H,   IMG_W,   4.18300658e-01]])
    batch = mx.io.DataBatch([mx.nd.array(im_tensor), mx.nd.array(im_info)])
    mod.forward(batch)
    output_names = mod.output_names
    output_tensor = mod.get_outputs()
    mod.get_outputs()[0].wait_to_read()   

    output = dict(zip(output_names ,output_tensor))

    rois = output['rois_output'].asnumpy()[:, 1:]
    scores = output['cls_prob_reshape_output'].asnumpy()[0]
    bbox_deltas = output['bbox_pred_reshape_output'].asnumpy()[0]
    np.savetxt("scores.txt",scores)
    pred_boxes = bbox_pred(rois, bbox_deltas)
    pred_boxes = clip_boxes(pred_boxes, im_shape[-2:])

    num_classes = 2

    all_cls_dets = [[] for _ in range(num_classes)]
    
    for j in range(1, num_classes):
        indexes = np.where(scores[:, j] > 0.1)[0]
        print(indexes)
        cls_scores = scores[indexes, j, np.newaxis]
        cls_boxes = pred_boxes[indexes, j * 4:(j + 1) * 4]
        cls_dets = np.hstack((cls_boxes, cls_scores)).copy()
        all_cls_dets[j] = cls_dets

    for idx_class in range(1, num_classes):
        nms = py_nms_wrapper(0.3)
        keep = nms(all_cls_dets[idx_class])
        all_cls_dets[idx_class] = all_cls_dets[idx_class][keep, :]

    for i in range(all_cls_dets[1].shape[0]):
        cv2.rectangle(rimg, (int(all_cls_dets[1][i][0]), int(all_cls_dets[1][i][1]))
        ,(int(all_cls_dets[1][i][2]), int(all_cls_dets[1][i][3])),(0,0,255),1)
        print((int(all_cls_dets[1][i][0]), int(all_cls_dets[1][i][1]))
        ,(int(all_cls_dets[1][i][2]), int(all_cls_dets[1][i][3])))
    
    cv2.imshow("w", rimg)
    cv2.waitKey()