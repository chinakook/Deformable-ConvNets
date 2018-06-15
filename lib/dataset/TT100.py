# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Haozhi Qi, from py-faster-rcnn (https://github.com/rbgirshick/py-faster-rcnn)
# --------------------------------------------------------


import cPickle
import cv2
import os
import numpy as np
import PIL
import imdb
from imdb import IMDB

class TT100(IMDB):
    def __init__(self, image_set, root_path, devkit_path, result_path=None, mask_size=-1, binary_thresh=None):
        """
        fill basic information to initialize imdb
        :param image_set: 2007_trainval, 2007_test, etc
        :param root_path: 'selective_search_data' and 'cache'
        :param devkit_path: data and results
        :return: imdb object
        """              
        
        super(TT100, self).__init__('TT100', image_set, root_path, devkit_path, result_path)  # set self.name
        self.image_set = image_set
        self.root_path = root_path
        self.devkit_path = devkit_path
        self.data_path =  os.path.join(devkit_path,image_set)

        # self.classes = ['__background__',  # always index 0
        #                 'sign']
        self.classes = ['__background__','i1', 'i10', 'i11', 'i12', 'i13', 'i14', 'i15', 'i2', 'i3', 'i4', 'i5', 'il100', 'il110', 'il50', 'il60', 'il70', 'il80', 'il90', 'io', 'ip', 'p1', 'p10', 'p11', 'p12', 'p13', 'p14', 'p15', 'p16', 'p17', 'p18', 'p19', 'p2', 'p20', 'p21', 'p22', 'p23', 'p24', 'p25', 'p26', 'p27', 'p28', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'pa10', 'pa12', 'pa13', 'pa14', 'pa8', 'pb', 'pc', 'pg', 'ph1.5', 'ph2', 'ph2.1', 'ph2.2', 'ph2.4', 'ph2.5', 'ph2.8', 'ph2.9', 'ph3', 'ph3.2', 'ph3.5', 'ph3.8', 'ph4', 'ph4.2', 'ph4.3', 'ph4.5', 'ph4.8', 'ph5', 'ph5.3', 'ph5.5', 'pl10', 'pl100', 'pl110', 'pl120', 'pl15', 'pl20', 'pl25', 'pl30', 'pl35', 'pl40', 'pl5', 'pl50', 'pl60', 'pl65', 'pl70', 'pl80', 'pl90', 'pm10', 'pm13', 'pm15', 'pm1.5', 'pm2', 'pm20', 'pm25', 'pm30', 'pm35', 'pm40', 'pm46', 'pm5', 'pm50', 'pm55', 'pm8', 'pn', 'pne', 'po', 'pr10', 'pr100', 'pr20', 'pr30', 'pr40', 'pr45', 'pr50', 'pr60', 'pr70', 'pr80', 'ps', 'pw2', 'pw2.5', 'pw3', 'pw3.2', 'pw3.5', 'pw4', 'pw4.2', 'pw4.5', 'w1', 'w10', 'w12', 'w13', 'w16', 'w18', 'w20', 'w21', 'w22', 'w24', 'w28', 'w3', 'w30', 'w31', 'w32', 'w34', 'w35', 'w37', 'w38', 'w41', 'w42', 'w43', 'w44', 'w45', 'w46', 'w47', 'w48', 'w49', 'w5', 'w50', 'w55', 'w56', 'w57', 'w58', 'w59', 'w60', 'w62', 'w63', 'w66', 'w8', 'wo', 'i6', 'i7', 'i8', 'i9', 'ilx', 'p29', 'w29', 'w33', 'w36', 'w39', 'w4', 'w40', 'w51', 'w52', 'w53', 'w54', 'w6', 'w61', 'w64', 'w65', 'w67', 'w7', 'w9', 'pax', 'pd', 'pe', 'phx', 'plx', 'pmx', 'pnl', 'prx', 'pwx', 'w11', 'w14', 'w15', 'w17', 'w19', 'w2', 'w23', 'w25', 'w26', 'w27', 'pl0', 'pl4', 'pl3', 'pm2.5', 'ph4.4', 'pn40', 'ph3.3', 'ph2.6']
        # self.classes=['__background__','w01', 'w02', 'w10', 'w11', 'w17', 'w19', 'w31', 'w32', 
        #                 'w34', 'w35', 'w37', 'w41', 'w43', 'w45', 's02', 's03', 's07', 's11', 
        #                 's13', 's14', 's19', 's22', 's25', 's27', 's28', 's29', 's30', 's31', 
        #                 's32', 's33', 's35', 's36', 's37', 'p02', 'p05', 'p06', 'p07', 'p08', 
        #                 'p09', 'p10', 'p11', 'p12', 'p13', 'p16', 'p19', 'p21', 'p23', 'p24', 
        #                 'p25', 'p26', 'p27', 'p36']
        self.num_classes = len(self.classes)
        self.image_set_index = self.load_image_set_index()
        self.num_images = len(self.image_set_index)
        print 'num_images', self.num_images
        self.mask_size = mask_size
        self.binary_thresh = binary_thresh

        self.config = {'comp_id': 'comp4',
                       'use_diff': False,
                       'min_size': 2}

    def load_image_set_index(self):
        """
        find out which indexes correspond to given image set (train or val)
        :return:
        """
        image_set_index_file = os.path.join(self.data_path,  'Main', self.image_set + '.txt')
        assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
        with open(image_set_index_file) as f:
            image_set_index = [x.strip() for x in f.readlines()]
        return image_set_index

    def image_path_from_index(self, index):
        """
        given image index, find out full path
        :param index: index of a specific image
        :return: full path of this image
        """
        image_file = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')
        assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
        return image_file


    def gt_roidb(self):
        """
        return ground truth image regions database
        :return: imdb[image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self.load_pascal_annotation(index) for index in self.image_set_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb



    def load_pascal_annotation(self, index):
        """
        for a given index, load image and bounding boxes info from XML file
        :param index: index of a specific image
        :return: record['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        import xml.etree.ElementTree as ET
        roi_rec = dict()
        roi_rec['image'] = self.image_path_from_index(index)

        filename = os.path.join(self.data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        size = tree.find('size')
        h_t = float(size.find('height').text)
        w_t = float(size.find('width').text)
        roi_rec['height'] = float(size.find('height').text)
        roi_rec['width'] = float(size.find('width').text)
        #im_size = cv2.imread(roi_rec['image'], cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION).shape
        #assert im_size[0] == roi_rec['height'] and im_size[1] == roi_rec['width']

        objs = tree.findall('object')
        if not self.config['use_diff']:
            non_diff_objs = [obj for obj in objs if int(obj.find('difficult').text) == 0]
            objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        class_to_index = dict(zip(self.classes, range(self.num_classes)))
        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = max(float(bbox.find('xmin').text) - 1,0)
            y1 = max(float(bbox.find('ymin').text) - 1,0)
            x2 = min(float(bbox.find('xmax').text) - 1 , w_t)
            y2 = min(float(bbox.find('ymax').text) - 1, h_t)
            cls = class_to_index[obj.find('name').text.lower().strip()]
            if(cls>0):
                cls = 1
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

        roi_rec.update({'boxes': boxes,
                        'gt_classes': gt_classes,
                        'gt_overlaps': overlaps,
                        'max_classes': overlaps.argmax(axis=1),
                        'max_overlaps': overlaps.max(axis=1),
                        'flipped': False})
        return roi_rec


if __name__ == '__main__':    
    pass
