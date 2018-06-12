import _init_paths

import os
from utils import image
from symbols import *
from bbox.bbox_transform import bbox_pred, clip_boxes

import time
import cv2
import numpy as np
import mxnet as mx
from nms.nms import py_nms_wrapper, py_softnms_wrapper

IMG_H = 1024
IMG_W = 1024

def det(mod, fn):
    
    raw_img = cv2.imread(fn)

    if raw_img.shape[0] < raw_img.shape[1]:
        raw_img = cv2.copyMakeBorder(raw_img,0
        ,raw_img.shape[1]-raw_img.shape[0], 0, 0, cv2.BORDER_CONSTANT)

    im_shape = [IMG_H,IMG_W] # reverse order
    img = cv2.resize(raw_img, (IMG_H,IMG_W))
    raw_h = img.shape[0]
    raw_w = img.shape[1]

    #im_tensor = image.transform(img, [104,117,124], 0.0167)
    im_tensor = image.transform(img, [103.53, 116.28, 123.675], 0.017)

    im_info = np.array([[  IMG_H,   IMG_W,   4.18300658e-01]])

    batch = mx.io.DataBatch([mx.nd.array(im_tensor), mx.nd.array(im_info)])

    start = time.time()
    mod.forward(batch)

    output_names = mod.output_names
    output_tensor = mod.get_outputs()
    mod.get_outputs()[0].wait_to_read()
    print ("time", time.time()-start, "secs.")

    output = dict(zip(output_names ,output_tensor))

    rois = output['rois_output'].asnumpy()[:, 1:]
    scores = output['cls_prob_reshape_output'].asnumpy()[0]
    bbox_deltas = output['bbox_pred_reshape_output'].asnumpy()[0]

    pred_boxes = bbox_pred(rois, bbox_deltas)
    pred_boxes = clip_boxes(pred_boxes, im_shape[-2:])

    num_classes = 2

    all_cls_dets = [[] for _ in range(num_classes)]

    for j in range(1, num_classes):
        indexes = np.where(scores[:, j] > 0.1)[0]
        cls_scores = scores[indexes, j, np.newaxis]
        cls_boxes = pred_boxes[indexes, j * 4:(j + 1) * 4]
        cls_dets = np.hstack((cls_boxes, cls_scores)).copy()
        all_cls_dets[j] = cls_dets

    for idx_class in range(1, num_classes):
        nms = py_nms_wrapper(0.3)
        keep = nms(all_cls_dets[idx_class])
        all_cls_dets[idx_class] = all_cls_dets[idx_class][keep, :]

    for i in range(all_cls_dets[1].shape[0]):
        cv2.rectangle(img, (int(all_cls_dets[1][i][0]), int(all_cls_dets[1][i][1]))
        ,(int(all_cls_dets[1][i][2]), int(all_cls_dets[1][i][3])),(0,0,255),1)

    
    cv2.imshow("w", img)
    cv2.waitKey()

if __name__ == '__main__':
    sym, arg_params, aux_params = mx.model.load_checkpoint('/home/caizhendong/git/Deformable-ConvNets/output/fpn/coco/traffic_sign/train/fpn_traffic_sign',1)


    mod = mx.mod.Module(symbol=sym, context=mx.gpu(0), data_names=['data', 'im_info'], label_names=None)
    mod.bind(for_training=False, data_shapes=[('data', (1, 3, IMG_H, IMG_W)), ('im_info', (1, 3))], label_shapes=None, force_rebind=False)
    mod.set_params(arg_params=arg_params, aux_params=aux_params, force_init=False)

    #testdir = '/mnt/6B133E147DED759E/2016_01_18_07_01_01'
    testdir = '/home/caizhendong/git/Deformable-ConvNets/data/traffic_sign/test/JPEGImages'
    files = [i for i in os.listdir(testdir) if i.endswith('.jpg')]

    for i,fn in enumerate(files):
        #if i == 0:
        #    continue
        det(mod, testdir + '/' + fn)#'/'+ str(i) + '.jpg')