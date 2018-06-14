# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import mxnet as mx
import numpy as np
from mxnet.executor_manager import _split_input_slice
import os
import random
import cv2
from utils.image import tensor_vstack
from config.config import config
from rpn.rpn import get_rpn_testbatch, get_rpn_batch, assign_pyramid_anchor,tt_get_rpn_batch
from rcnn import get_rcnn_testbatch


def tt_par_assign_anchor_wrapper(cfg, iroidb, feat_sym, feat_strides, anchor_scales, anchor_ratios, allowed_border):
    # get testing data for multigpu
    data, rpn_label = tt_get_rpn_batch(iroidb, cfg)
    data_shape = {k: v.shape for k, v in data.items()}
    del data_shape['im_info']

    # add gt_boxes to data for e2e
    data['gt_boxes'] = rpn_label['gt_boxes'][np.newaxis, :, :]

    feat_shape = [y[1] for y in [x.infer_shape(**data_shape) for x in feat_sym]]
    label = assign_pyramid_anchor(feat_shape, rpn_label['gt_boxes'], data['im_info'], cfg,
                                  feat_strides, anchor_scales, anchor_ratios, allowed_border)
    return {'data': data, 'label': label}

def par_assign_anchor_wrapper(cfg, iroidb, feat_sym, feat_strides, anchor_scales, anchor_ratios, allowed_border):
    # get testing data for multigpu
    data, rpn_label = get_rpn_batch(iroidb, cfg)
    data_shape = {k: v.shape for k, v in data.items()}
    del data_shape['im_info']

    # add gt_boxes to data for e2e
    data['gt_boxes'] = rpn_label['gt_boxes'][np.newaxis, :, :]

    feat_shape = [y[1] for y in [x.infer_shape(**data_shape) for x in feat_sym]]
    label = assign_pyramid_anchor(feat_shape, rpn_label['gt_boxes'], data['im_info'], cfg,
                                  feat_strides, anchor_scales, anchor_ratios, allowed_border)
    return {'data': data, 'label': label}

class TestLoader(mx.io.DataIter):
    def __init__(self, roidb, batch_size=1, shuffle=False,
                 has_rpn=False):
        super(TestLoader, self).__init__()

        # save parameters as properties
        self.roidb = roidb
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.has_rpn = has_rpn

        # infer properties from roidb
        self.size = len(self.roidb)
        self.index = np.arange(self.size)

        # decide data and label names (only for training)
        if has_rpn:
            self.data_name = ['data', 'im_info']
        else:
            self.data_name = ['data', 'rois']
        self.label_name = None

        # status variable for synchronization between get_data and get_label
        self.cur = 0
        self.data = None
        self.label = None
        self.im_info = None

        # get first batch to fill in provide_data and provide_label
        self.reset()
        self.get_batch()

    @property
    def provide_data(self):
        return [(k, v.shape) for k, v in zip(self.data_name, self.data)]

    @property
    def provide_label(self):
        return None

    def reset(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            return self.im_info, \
                   mx.io.DataBatch(data=self.data, label=self.label,
                                   pad=self.getpad(), index=self.getindex(),
                                   provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def get_batch(self):
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        roidb = [self.roidb[self.index[i]] for i in range(cur_from, cur_to)]
        if self.has_rpn:
            data, label, im_info = get_rpn_testbatch(roidb)
        else:
            data, label, im_info = get_rcnn_testbatch(roidb)
        self.data = [mx.nd.array(data[name]) for name in self.data_name]
        self.im_info = im_info

class PyramidAnchorIterator(mx.io.DataIter):
    # pool = Pool(processes=4)
    def __init__(self, feat_sym, roidb, cfg, batch_size=1, shuffle=False, ctx=None, work_load_list=None,
                 feat_strides=(4, 8, 16, 32, 64), anchor_scales=(8, ), anchor_ratios=(0.5, 1, 2), allowed_border=0,
                 aspect_grouping=False):
        """
        This Iter will provide roi data to Fast R-CNN network
        :param feat_sym: to infer shape of assign_output
        :param roidb: must be preprocessed
        :param batch_size: must divide BATCH_SIZE(128)
        :param shuffle: bool
        :param ctx: list of contexts
        :param work_load_list: list of work load
        :param aspect_grouping: group images with similar aspects
        :return: AnchorLoader
        """
        super(PyramidAnchorIterator, self).__init__()

        # save parameters as properties
        self.feat_sym = feat_sym
        self.roidb = roidb
        self.cfg = cfg
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = [mx.cpu()]
        self.work_load_list = work_load_list
        self.feat_strides = feat_strides
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.allowed_border = allowed_border
        self.aspect_grouping = aspect_grouping

        # infer properties from roidb
        self.size = len(roidb)
        self.index = np.arange(self.size)

        # decide data and label names
        if self.cfg.TRAIN.END2END:
            self.data_name = ['data', 'im_info', 'gt_boxes']
        else:
            self.data_name = ['data']
        self.feat_pyramid_level = np.log2(self.cfg.network.RPN_FEAT_STRIDE).astype(int)
        # self.label_name = ['label_p' + str(x) for x in self.feat_pyramid_level] +\
        #                   ['bbox_target_p' + str(x) for x in self.feat_pyramid_level] +\
        #                   ['bbox_weight_p' + str(x) for x in self.feat_pyramid_level]

        self.label_name = ['label', 'bbox_target', 'bbox_weight']

        # status variable for synchronization between get_data and get_label
        self.cur = 0
        self.batch = None
        self.data = None
        self.label = None

        # get first batch to fill in provide_data and provide_label
        self.reset()
        self.get_batch_parallel()

    @property
    def provide_data(self):
        return [(k, v.shape) for k, v in zip(self.data_name, self.data)]

    @property
    def provide_label(self):
        return [(k, v.shape) for k, v in zip(self.label_name, self.label)]

    # @property
    # def provide_data_single(self):
    #     return [(k, v.shape) for k, v in zip(self.data_name, self.data[0])]

    # @property
    # def provide_label_single(self):
    #     return [(k, v.shape) for k, v in zip(self.label_name, self.label[0])]

    def reset(self):
        self.cur = 0
        if self.shuffle:
            if self.aspect_grouping:
                widths = np.array([r['width'] for r in self.roidb])
                heights = np.array([r['height'] for r in self.roidb])
                horz = (widths >= heights)
                vert = np.logical_not(horz)
                horz_inds = np.where(horz)[0]
                vert_inds = np.where(vert)[0]
                inds = np.hstack((np.random.permutation(horz_inds), np.random.permutation(vert_inds)))
                extra = inds.shape[0] % self.batch_size
                inds_ = np.reshape(inds[:-extra], (-1, self.batch_size))
                row_perm = np.random.permutation(np.arange(inds_.shape[0]))
                inds[:-extra] = np.reshape(inds_[row_perm, :], (-1,))
                self.index = inds
            else:
                np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def next(self):
        if self.iter_next():
            self.get_batch_parallel()
            # self.get_batch()
            self.cur += self.batch_size
            return mx.io.DataBatch(data=self.data, label=self.label,
                                   pad=self.getpad(), index=self.getindex(),
                                   provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def infer_shape(self, max_data_shape=None, max_label_shape=None):
        """ Return maximum data and label shape for single gpu """
        if max_data_shape is None:
            max_data_shape = []
        if max_label_shape is None:
            max_label_shape = []
        max_shapes = dict(max_data_shape + max_label_shape)
        input_batch_size = max_shapes['data'][0]
        im_info = [[max_shapes['data'][2], max_shapes['data'][3], 1.0]]

        feat_shape = [y[1] for y in [x.infer_shape(**max_shapes) for x in self.feat_sym]]
        label = assign_pyramid_anchor(feat_shape, np.zeros((0, 5)), im_info, self.cfg,
                                      self.feat_strides, self.anchor_scales, self.anchor_ratios, self.allowed_border)
        label = [label[k] for k in self.label_name]
        label_shape = [(k, tuple([input_batch_size] + list(v.shape[1:]))) for k, v in zip(self.label_name, label)]

        return max_data_shape, label_shape

    def get_batch_parallel(self):
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        roidb = [self.roidb[self.index[i]] for i in range(cur_from, cur_to)]
        # decide multi device slice
        work_load_list = self.work_load_list
        ctx = self.ctx
        if work_load_list is None:
            work_load_list = [1] * len(ctx)
        assert isinstance(work_load_list, list) and len(work_load_list) == len(ctx), \
            "Invalid settings for work load. "
        slices = _split_input_slice(self.batch_size, work_load_list)

        #rst = []
        data_list = []
        label_list = []
        for idx, islice in enumerate(slices):
            iroidb = [roidb[i] for i in range(islice.start, islice.stop)]
            aaa = par_assign_anchor_wrapper(self.cfg, iroidb, self.feat_sym, self.feat_strides, self.anchor_scales,
                                                 self.anchor_ratios, self.allowed_border)
            #rst.append(aaa)
            data_list.append(aaa['data'])
            label_list.append(aaa["label"])

        all_data = dict()
        for key in self.data_name:
            all_data[key] = tensor_vstack([batch[key] for batch in data_list])

        all_label = dict()
        for key in self.label_name:
            #pad = -1 if key == 'label' else 0
            all_label[key] = tensor_vstack([batch[key] for batch in label_list])
        #all_data = [_['data'] for _ in rst]
        #all_label = [_['label'] for _ in rst]
        # self.data = [[mx.nd.array(data[key]) for key in self.data_name] for data in all_data]
        # self.label = [[mx.nd.array(label[key]) for key in self.label_name] for label in all_label]
        self.data = [mx.nd.array(all_data[key]) for key in self.data_name]
        self.label = [mx.nd.array(all_label[key]) for key in self.label_name]

    # def get_batch(self):
    #     # slice roidb
    #     cur_from = self.cur
    #     cur_to = min(cur_from + self.batch_size, self.size)
    #     roidb = [self.roidb[self.index[i]] for i in range(cur_from, cur_to)]

    #     # decide multi device slice
    #     work_load_list = self.work_load_list
    #     ctx = self.ctx
    #     if work_load_list is None:
    #         work_load_list = [1] * len(ctx)
    #     assert isinstance(work_load_list, list) and len(work_load_list) == len(ctx), \
    #         "Invalid settings for work load. "
    #     slices = _split_input_slice(self.batch_size, work_load_list)

    #     # get testing data for multigpu
    #     data_list = []
    #     label_list = []
    #     for islice in slices:
    #         iroidb = [roidb[i] for i in range(islice.start, islice.stop)]
    #         data, label = get_rpn_batch(iroidb)
    #         data_list.append(data)
    #         label_list.append(label)

        # pad data first and then assign anchor (read label)
        # data_tensor = tensor_vstack([batch['data'] for batch in data_list])
        # for data, data_pad in zip(data_list, data_tensor):
        #     data['data'] = data_pad[np.newaxis, :]

        # new_label_list = []
        # for data, label in zip(data_list, label_list):
        #     # infer label shape
        #     data_shape = {k: v.shape for k, v in data.items()}
        #     del data_shape['im_info']
        #     _, feat_shape, _ = self.feat_sym.infer_shape(**data_shape)
        #     feat_shape = [int(i) for i in feat_shape[0]]

        #     # add gt_boxes to data for e2e
        #     #data['gt_boxes'] = label['gt_boxes'][np.newaxis, :, :]

        #     # assign anchor for label
        #     label = par_assign_anchor_wrapper(self.cfg, iroidb, self.feat_sym, self.feat_strides, self.anchor_scales,
        #                                           self.anchor_ratios, self.allowed_border)
        #     new_label_list.append(label)

        # all_data = dict()
        # for key in self.data_name:
        #     all_data[key] = tensor_vstack([batch[key] for batch in data_list])

        # all_label = dict()
        # for key in self.label_name:
        #     pad = -1 if key == 'label' else 0
        #     all_label[key] = tensor_vstack([batch[key] for batch in new_label_list], pad=pad)

        # self.data = [mx.nd.array(all_data[key]) for key in self.data_name]
        # self.label = [mx.nd.array(all_label[key]) for key in self.label_name]
class TT_PyramidAnchorIterator(mx.io.DataIter):
    
    # pool = Pool(processes=4)
    def __init__(self, feat_sym, roidb, cfg, batch_size=1, shuffle=False, ctx=None, work_load_list=None,
                 feat_strides=(4, 8, 16, 32, 64), anchor_scales=(8, ), anchor_ratios=(0.5, 1, 2), allowed_border=0,
                 aspect_grouping=False):
        """
        This Iter will provide roi data to Fast R-CNN network
        :param feat_sym: to infer shape of assign_output
        :param roidb: must be preprocessed
        :param batch_size: must divide BATCH_SIZE(128)
        :param shuffle: bool
        :param ctx: list of contexts
        :param work_load_list: list of work load
        :param aspect_grouping: group images with similar aspects
        :return: AnchorLoader
        """
        super(TT_PyramidAnchorIterator, self).__init__()

        # save parameters as properties
        self.feat_sym = feat_sym
        self.roidb = roidb
        self.cfg = cfg
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = [mx.cpu()]
        self.work_load_list = work_load_list
        self.feat_strides = feat_strides
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.allowed_border = allowed_border
        self.aspect_grouping = aspect_grouping

        # infer properties from roidb
        self.size = len(roidb)
        self.index = np.arange(self.size)

        # decide data and label names
        if self.cfg.TRAIN.END2END:
            self.data_name = ['data', 'im_info', 'gt_boxes']
        else:
            self.data_name = ['data']
        self.feat_pyramid_level = np.log2(self.cfg.network.RPN_FEAT_STRIDE).astype(int)
        # self.label_name = ['label_p' + str(x) for x in self.feat_pyramid_level] +\
        #                   ['bbox_target_p' + str(x) for x in self.feat_pyramid_level] +\
        #                   ['bbox_weight_p' + str(x) for x in self.feat_pyramid_level]

        self.label_name = ['label', 'bbox_target', 'bbox_weight']

        # status variable for synchronization between get_data and get_label
        self.cur = 0
        self.batch = None
        self.data = None
        self.label = None

        # get first batch to fill in provide_data and provide_label
        self.reset()
        self.get_batch_parallel()

    @property
    def provide_data(self):
        return [(k, v.shape) for k, v in zip(self.data_name, self.data)]

    @property
    def provide_label(self):
        return [(k, v.shape) for k, v in zip(self.label_name, self.label)]
  

    def reset(self):
        self.cur = 0
        if self.shuffle:
            if self.aspect_grouping:
                widths = np.array([r['width'] for r in self.roidb])
                heights = np.array([r['height'] for r in self.roidb])
                horz = (widths >= heights)
                vert = np.logical_not(horz)
                horz_inds = np.where(horz)[0]
                vert_inds = np.where(vert)[0]
                inds = np.hstack((np.random.permutation(horz_inds), np.random.permutation(vert_inds)))
                extra = inds.shape[0] % self.batch_size
                inds_ = np.reshape(inds[:-extra], (-1, self.batch_size))
                row_perm = np.random.permutation(np.arange(inds_.shape[0]))
                inds[:-extra] = np.reshape(inds_[row_perm, :], (-1,))
                self.index = inds
            else:
                np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def next(self):
        if self.iter_next():
            self.get_batch_parallel()
            # self.get_batch()
            self.cur += self.batch_size
            return mx.io.DataBatch(data=self.data, label=self.label,
                                   pad=self.getpad(), index=self.getindex(),
                                   provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def infer_shape(self, max_data_shape=None, max_label_shape=None):
        """ Return maximum data and label shape for single gpu """
        if max_data_shape is None:
            max_data_shape = []
        if max_label_shape is None:
            max_label_shape = []
        max_shapes = dict(max_data_shape + max_label_shape)
        input_batch_size = max_shapes['data'][0]
        im_info = [[max_shapes['data'][2], max_shapes['data'][3], 1.0]]

        feat_shape = [y[1] for y in [x.infer_shape(**max_shapes) for x in self.feat_sym]]
        label = assign_pyramid_anchor(feat_shape, np.zeros((0, 5)), im_info, self.cfg,
                                      self.feat_strides, self.anchor_scales, self.anchor_ratios, self.allowed_border)
        label = [label[k] for k in self.label_name]
        label_shape = [(k, tuple([input_batch_size] + list(v.shape[1:]))) for k, v in zip(self.label_name, label)]

        return max_data_shape, label_shape

    def get_batch_parallel(self):
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        roidb = [self.roidb[self.index[i]] for i in range(cur_from, cur_to)]
        # decide multi device slice
        work_load_list = self.work_load_list
        ctx = self.ctx
        if work_load_list is None:
            work_load_list = [1] * len(ctx)
        assert isinstance(work_load_list, list) and len(work_load_list) == len(ctx), \
            "Invalid settings for work load. "
        slices = _split_input_slice(self.batch_size, work_load_list)

        #rst = []
        data_list = []
        label_list = []
        for idx, islice in enumerate(slices):
            iroidb = [roidb[i] for i in range(islice.start, islice.stop)]
            aaa = tt_par_assign_anchor_wrapper(self.cfg, iroidb, self.feat_sym, self.feat_strides, self.anchor_scales,
                                                 self.anchor_ratios, self.allowed_border)
            #rst.append(aaa)
            data_list.append(aaa['data'])
            label_list.append(aaa["label"])

        all_data = dict()
        for key in self.data_name:
            all_data[key] = tensor_vstack([batch[key] for batch in data_list])

        all_label = dict()
        for key in self.label_name:
            #pad = -1 if key == 'label' else 0
            all_label[key] = tensor_vstack([batch[key] for batch in label_list])
        #all_data = [_['data'] for _ in rst]
        #all_label = [_['label'] for _ in rst]
        # self.data = [[mx.nd.array(data[key]) for key in self.data_name] for data in all_data]
        # self.label = [[mx.nd.array(label[key]) for key in self.label_name] for label in all_label]
        self.data = [mx.nd.array(all_data[key]) for key in self.data_name]
        self.label = [mx.nd.array(all_label[key]) for key in self.label_name]