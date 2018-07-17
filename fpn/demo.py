# -*- coding: utf-8 -*-
from __future__ import absolute_import

import _init_paths
from natsort import natsort
import os
from utils import image
from symbols import *
from bbox.bbox_transform import bbox_pred, clip_boxes
import math
import time
import cv2
import numpy as np
import mxnet as mx
from nms.nms import py_nms_wrapper, py_softnms_wrapper
from lxml.etree import Element, SubElement, tostring  
import pprint  
from xml.dom.minidom import parseString


def write_xml(img_name, height, width, bboxes, extra=None):
    node_root = Element('annotation')
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'train'
    node_filename = SubElement(node_root, 'filename')
    node_filename.text = os.path.basename(img_name)
    node_path = SubElement(node_root, 'path')
    node_path.text = img_name
    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(width)
    node_height = SubElement(node_size, 'height')
    node_height.text = str(height)
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'
    for k in range(bboxes.shape[0]):
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = "plate"
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(bboxes[k][0])
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(bboxes[k][1])
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(bboxes[k][2])
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(bboxes[k][3])

        if extra is not None:
            node_extra = SubElement(node_object, 'extra')
            node_extra.text = extra[k]
    
    xml = tostring(node_root, pretty_print=True)
    with open(img_name[:-3]+'xml','w') as f: ## Write document to file
        f.write(xml)
    f.close()

class Detector(object):
    ALIGN = 32
    def __init__(self, prefix, epoch, use_rpn_only=False):
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        self.mod = mx.mod.Module(symbol=sym, context=[mx.gpu(0)], data_names=['data', 'im_info'], label_names=None)
        self.mod.bind(for_training=False, data_shapes=[('data', (1, 3, 512, 512)), ('im_info', (1, 3))], label_shapes=None, force_rebind=False)
        self.mod.set_params(arg_params=arg_params, aux_params=aux_params, force_init=False)
        self.use_rpn_only = use_rpn_only

    def det(self, raw_img):
        h,w = raw_img.shape[0], raw_img.shape[1]

        norm_h = ((h + self.ALIGN - 1) / self.ALIGN) * self.ALIGN
        norm_w = ((w + self.ALIGN - 1) / self.ALIGN) * self.ALIGN
        pad_h = norm_h - h
        pad_w = norm_w - w

        img = cv2.copyMakeBorder(raw_img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(127,127,127))

        im_shape = [norm_h, norm_w] # reverse order
        im_scales = 1
        im_tensor = image.transform(img, [103.06,115.90,123.15])#,0.017

        im_info = np.array([[norm_h, norm_w, im_scales]])
        #4.18300658e-01
        batch = mx.io.DataBatch([mx.nd.array(im_tensor), mx.nd.array(im_info)])

        start = time.time()
        self.mod.forward(batch)

        output_names = self.mod.output_names
        output_tensor = self.mod.get_outputs()
        output_tensor[0].wait_to_read()
        print ("time", time.time()-start, "secs.")

        output = dict(zip(output_names ,output_tensor))
        if self.use_rpn_only:
            rois = output['rois_output'].asnumpy()[:, 1:]
            scores = output['rois_score'].asnumpy()
            #pred_boxes = clip_boxes(pred_boxes, im_shape[-2:])
            keep = np.where(scores> 0.7)[0]
            rois, scores = rois[keep], scores[keep]
            cls_dets = np.hstack((rois, scores))
            nms = py_nms_wrapper(0.3)

            keep = nms(cls_dets)
            all_boxes = cls_dets[keep, :]
            return all_boxes / im_scales
        else:
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
            
            nms = py_nms_wrapper(0.3)

            for idx_class in range(1, num_classes):
                keep = nms(all_cls_dets[idx_class])
                all_cls_dets[idx_class] = all_cls_dets[idx_class][keep, :]
            return all_cls_dets[1] / im_scales
            # xml_fn = fn[:-4] + '.xml'
            # if not os.path.exists(xml_fn):
            #     write_xml(fn,h,w,all_cls_dets[1],im_scales)

if __name__ == '__main__':

    detector = Detector('model/test_plate',0)

    testdir = u'/mnt/15F1B72E1A7798FD/DK2/plates_det/bt'
    files = [i for i in os.listdir(testdir) if i.endswith('.jpg')]
    files = natsort(files)

    #mx.profiler.set_config(profile_all=True, filename='profile_imageiter.json')
    #mx.profiler.set_state('run')
    for i,fn in enumerate(files):
        fullfn = testdir + '/' + fn
        raw_img = cv2.imdecode(np.fromfile(fullfn, dtype=np.uint8),-1)
        bboxes = detector.det(raw_img)
        write_xml(fullfn, raw_img.shape[0], raw_img.shape[1], bboxes)
    #mx.profiler.set_state('stop')