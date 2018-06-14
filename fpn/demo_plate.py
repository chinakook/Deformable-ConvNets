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
from lxml.etree import Element, SubElement, tostring  
import pprint  
from xml.dom.minidom import parseString 
DST_SIZE = 1024
IMG_H = DST_SIZE
IMG_W = DST_SIZE
def write_xml(img_name,height,width,bbox,im_scales):
    node_root = Element('annotation')
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'train'
    node_filename = SubElement(node_root, 'filename')
    node_filename.text = img_name

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(width)
    node_height = SubElement(node_size, 'height')
    node_height.text = str(height)
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'
    for k in range(bbox.shape[0]):
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = "sign"
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(bbox[k][0]/im_scales)
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(bbox[k][1]/im_scales)
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(bbox[k][2]/im_scales)
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(bbox[k][3]/im_scales)
    
    xml = tostring(node_root, pretty_print=True)
    with open(img_name[:-3]+'xml','w') as f: ## Write document to file
            f.write(xml)
    f.close()
def det(mod, fn):
    raw_img = cv2.imdecode(np.fromfile(fn, dtype=np.uint8),-1)
    h,w = raw_img.shape[0], raw_img.shape[1]

    if h > DST_SIZE or w > DST_SIZE:
        if h > w:
            raw_img = cv2.copyMakeBorder(raw_img, 0, 0, 0, h-w, cv2.BORDER_CONSTANT, value=(128,128,128))
            s = DST_SIZE / float(h)
        else:
            raw_img = cv2.copyMakeBorder(raw_img, 0, w-h, 0, 0, cv2.BORDER_CONSTANT, value=(128,128,128))
            s = DST_SIZE / float(w)
        raw_img = cv2.resize(raw_img, (DST_SIZE, DST_SIZE))

    else:
        if h <= DST_SIZE:
            bottom = DST_SIZE - h
        if w <= DST_SIZE:
            right = DST_SIZE - w
        raw_img = cv2.copyMakeBorder(raw_img, 0, bottom, 0, right, cv2.BORDER_CONSTANT, value=(128,128,128))


    im_shape = [IMG_H,IMG_W] # reverse order
    img = cv2.resize(raw_img, (IMG_H,IMG_W))
    raw_h = img.shape[0]
    raw_w = img.shape[1]

    im_tensor = image.transform(img, [103.06,115.90,123.15])

    im_info = np.array([[  IMG_H,   IMG_W,   4.18300658e-01]])
    im_scales = 4.18300658e-01
    batch = mx.io.DataBatch([mx.nd.array(im_tensor), mx.nd.array(im_info)])

    start = time.time()
    mod.forward(batch)

    output_names = mod.output_names
    output_tensor = mod.get_outputs()
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
        indexes = np.where(scores[:, j] > 0.5)[0]
        cls_scores = scores[indexes, j, np.newaxis]
        cls_boxes = pred_boxes[indexes, j * 4:(j + 1) * 4]
        cls_dets = np.hstack((cls_boxes, cls_scores)).copy()
        all_cls_dets[j] = cls_dets
    
    for idx_class in range(1, num_classes):
        nms = py_nms_wrapper(0.3)
        keep = nms(all_cls_dets[idx_class])
        all_cls_dets[idx_class] = all_cls_dets[idx_class][keep, :]
    write_xml(fn,raw_h,raw_w,all_cls_dets[1],im_scales)
    # for i in range(all_cls_dets[1].shape[0]):
        
    #     cv2.rectangle(img, (int(all_cls_dets[1][i][0]), int(all_cls_dets[1][i][1]))
    #     ,(int(all_cls_dets[1][i][2]), int(all_cls_dets[1][i][3])),(0,0,255),2)

    # if img.shape[0] > 1024 or img.shape[1]> 1024:
    #     img = cv2.resize(img, (0,0), fx=0.3, fy=0.3)
    # cv2.imshow("w", img)
    # cv2.waitKey()

if __name__ == '__main__':
    sym, arg_params, aux_params = mx.model.load_checkpoint('test_traffic_tt100_one',0)

    mod = mx.mod.Module(symbol=sym, context=mx.gpu(0), data_names=['data', 'im_info'], label_names=None)
    mod.bind(for_training=False, data_shapes=[('data', (1, 3, IMG_H, IMG_W)), ('im_info', (1, 3))], label_shapes=None, force_rebind=False)
    mod.set_params(arg_params=arg_params, aux_params=aux_params, force_init=False)

    testdir = u'/home/caizhendong/git/Deformable-ConvNets/data/traffic_sign/train/JPEGImages'
    #testdir = u'/mnt/15F1B72E1A7798FD/Dataset/Tsinghua_Tencent_100K/data/train/JPEGImages'
    files = [i for i in os.listdir(testdir) if i.endswith('.jpg')]

    for i,fn in enumerate(files):
        #if i == 0:
        #    continue
        det(mod, testdir + '/' + fn)#'/'+ str(i) + '.jpg')
