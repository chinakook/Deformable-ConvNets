# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by chinakook
# --------------------------------------------------------

import cPickle
import mxnet as mx
from utils.symbol import Symbol
from operator_py.pyramid_proposal import *
from operator_py.proposal_target import *
from operator_py.fpn_roi_pooling import *
from operator_py.box_annotator_ohem import *

from symbol_dpn_b import *

class resnet_v2_50_fpn_rcnn(Symbol):
    def __init__(self):
        """
        Use __init__ to define parameter network needs
        """
        self.shared_param_list = ['rpn_conv', 'rpn_cls_score', 'rpn_bbox_pred']
        self.shared_param_dict = {}
        for name in self.shared_param_list:
            self.shared_param_dict[name + '_weight'] = mx.sym.Variable(name + '_weight')
            self.shared_param_dict[name + '_bias'] = mx.sym.Variable(name + '_bias')

    def get_resnet_backbone(self, data, with_dilated=False, with_dconv=False, with_dpyramid=False, eps=1e-5):

        sym = mx.symbol.load('fpn/symbols/resnet50-symbol.json')

        layers = sym.get_internals()

        c2 = layers["resnetv20_stage2_activation0_output"]
        c3 = layers["resnetv20_stage3_activation0_output"]
        c4 = layers["resnetv20_stage4_activation0_output"]
        c5 = layers["resnetv20_relu1_fwd_output"]

        return c2, c3, c4, c5

    def get_fpn_feature(self, c2, c3, c4, c5, feature_dim=256):

        # lateral connection
        fpn_p5_1x1 = mx.symbol.Convolution(data=c5, kernel=(1, 1), pad=(0, 0), stride=(1, 1), num_filter=feature_dim, name='fpn_p5_1x1')
        fpn_p4_1x1 = mx.symbol.Convolution(data=c4, kernel=(1, 1), pad=(0, 0), stride=(1, 1), num_filter=feature_dim, name='fpn_p4_1x1')
        fpn_p3_1x1 = mx.symbol.Convolution(data=c3, kernel=(1, 1), pad=(0, 0), stride=(1, 1), num_filter=feature_dim, name='fpn_p3_1x1')
        fpn_p2_1x1 = mx.symbol.Convolution(data=c2, kernel=(1, 1), pad=(0, 0), stride=(1, 1), num_filter=feature_dim, name='fpn_p2_1x1')
        # top-down connection
        fpn_p5_upsample = mx.symbol.UpSampling(fpn_p5_1x1, scale=2, sample_type='nearest', name='fpn_p5_upsample')
        fpn_p4_plus = mx.sym.ElementWiseSum(*[fpn_p5_upsample, fpn_p4_1x1], name='fpn_p4_sum')
        fpn_p4_upsample = mx.symbol.UpSampling(fpn_p4_plus, scale=2, sample_type='nearest', name='fpn_p4_upsample')
        fpn_p3_plus = mx.sym.ElementWiseSum(*[fpn_p4_upsample, fpn_p3_1x1], name='fpn_p3_sum')
        fpn_p3_upsample = mx.symbol.UpSampling(fpn_p3_plus, scale=2, sample_type='nearest', name='fpn_p3_upsample')
        fpn_p2_plus = mx.sym.ElementWiseSum(*[fpn_p3_upsample, fpn_p2_1x1], name='fpn_p2_sum')
        # FPN feature
        fpn_p6 = mx.sym.Convolution(data=c5, kernel=(3, 3), pad=(1, 1), stride=(2, 2), num_filter=feature_dim, name='fpn_p6')
        fpn_p5 = mx.symbol.Convolution(data=fpn_p5_1x1, kernel=(3, 3), pad=(1, 1), stride=(1, 1), num_filter=feature_dim, name='fpn_p5')
        fpn_p4 = mx.symbol.Convolution(data=fpn_p4_plus, kernel=(3, 3), pad=(1, 1), stride=(1, 1), num_filter=feature_dim, name='fpn_p4')
        fpn_p3 = mx.symbol.Convolution(data=fpn_p3_plus, kernel=(3, 3), pad=(1, 1), stride=(1, 1), num_filter=feature_dim, name='fpn_p3')
        fpn_p2 = mx.symbol.Convolution(data=fpn_p2_plus, kernel=(3, 3), pad=(1, 1), stride=(1, 1), num_filter=feature_dim, name='fpn_p2')

        return fpn_p2, fpn_p3, fpn_p4, fpn_p5, fpn_p6

    def get_rpn_subnet(self, data, num_anchors, suffix):
        rpn_conv = mx.sym.Convolution(data=data, kernel=(3, 3), pad=(1, 1), num_filter=512, name='rpn_conv_' + suffix,
                                      weight=self.shared_param_dict['rpn_conv_weight'], bias=self.shared_param_dict['rpn_conv_bias'])
        rpn_relu = mx.sym.Activation(data=rpn_conv, act_type='relu', name='rpn_relu_' + suffix)
        rpn_cls_score = mx.sym.Convolution(data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name='rpn_cls_score_' + suffix,
                                           weight=self.shared_param_dict['rpn_cls_score_weight'], bias=self.shared_param_dict['rpn_cls_score_bias'])
        rpn_bbox_pred = mx.sym.Convolution(data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name='rpn_bbox_pred_' + suffix,
                                           weight=self.shared_param_dict['rpn_bbox_pred_weight'], bias=self.shared_param_dict['rpn_bbox_pred_bias'])

        # n x (2*A) x H x W => n x 2 x (A*H*W)
        rpn_cls_score_t1 = mx.sym.Reshape(data=rpn_cls_score, shape=(0, 2, -1, 0), name='rpn_cls_score_t1_' + suffix)
        rpn_cls_score_t2 = mx.sym.Reshape(data=rpn_cls_score_t1, shape=(0, 2, -1), name='rpn_cls_score_t2_' + suffix)
        rpn_cls_prob = mx.sym.softmax(data=rpn_cls_score_t1, axis=1, name='rpn_cls_prob_' + suffix)
        #rpn_cls_prob = mx.sym.SoftmaxActivation(data=rpn_cls_score_t1, mode='channel', name='rpn_cls_prob_' + suffix)
        rpn_cls_prob_t = mx.sym.Reshape(data=rpn_cls_prob, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_prob_t_' + suffix)
        rpn_bbox_pred_t = mx.sym.Reshape(data=rpn_bbox_pred, shape=(0, 0, -1), name='rpn_bbox_pred_t_' + suffix)
        return rpn_cls_score_t2, rpn_cls_prob_t, rpn_bbox_pred_t, rpn_bbox_pred

    def get_symbol(self, cfg, is_train=True):

        # config alias for convenient
        num_classes = cfg.dataset.NUM_CLASSES
        num_reg_classes = (2 if cfg.CLASS_AGNOSTIC else num_classes)

        data = mx.sym.Variable(name="data")
        im_info = mx.sym.Variable(name="im_info")

        # shared convolutional layers
        res2, res3, res4, res5 = self.get_resnet_backbone(data)
        fpn_p2, fpn_p3, fpn_p4, fpn_p5, fpn_p6 = self.get_fpn_feature(res2, res3, res4, res5)

        rpn_cls_score_p2, rpn_prob_p2, rpn_bbox_loss_p2, rpn_bbox_pred_p2 = self.get_rpn_subnet(fpn_p2, cfg.network.NUM_ANCHORS, 'p2')
        rpn_cls_score_p3, rpn_prob_p3, rpn_bbox_loss_p3, rpn_bbox_pred_p3 = self.get_rpn_subnet(fpn_p3, cfg.network.NUM_ANCHORS, 'p3')
        rpn_cls_score_p4, rpn_prob_p4, rpn_bbox_loss_p4, rpn_bbox_pred_p4 = self.get_rpn_subnet(fpn_p4, cfg.network.NUM_ANCHORS, 'p4')
        rpn_cls_score_p5, rpn_prob_p5, rpn_bbox_loss_p5, rpn_bbox_pred_p5 = self.get_rpn_subnet(fpn_p5, cfg.network.NUM_ANCHORS, 'p5')
        rpn_cls_score_p6, rpn_prob_p6, rpn_bbox_loss_p6, rpn_bbox_pred_p6 = self.get_rpn_subnet(fpn_p6, cfg.network.NUM_ANCHORS, 'p6')

        rpn_cls_prob_dict = {
            'rpn_cls_prob_stride64': rpn_prob_p6,
            'rpn_cls_prob_stride32': rpn_prob_p5,
            'rpn_cls_prob_stride16': rpn_prob_p4,
            'rpn_cls_prob_stride8': rpn_prob_p3,
            'rpn_cls_prob_stride4': rpn_prob_p2,
        }
        rpn_bbox_pred_dict = {
            'rpn_bbox_pred_stride64': rpn_bbox_pred_p6,
            'rpn_bbox_pred_stride32': rpn_bbox_pred_p5,
            'rpn_bbox_pred_stride16': rpn_bbox_pred_p4,
            'rpn_bbox_pred_stride8': rpn_bbox_pred_p3,
            'rpn_bbox_pred_stride4': rpn_bbox_pred_p2,
        }
        arg_dict = dict(rpn_cls_prob_dict.items() + rpn_bbox_pred_dict.items())

        if is_train:
            rpn_label = mx.sym.Variable(name='label')
            rpn_bbox_target = mx.sym.Variable(name='bbox_target')
            rpn_bbox_weight = mx.sym.Variable(name='bbox_weight')
            gt_boxes = mx.sym.Variable(name="gt_boxes")

            rpn_cls_score = mx.sym.Concat(rpn_cls_score_p2, rpn_cls_score_p3, rpn_cls_score_p4, rpn_cls_score_p5, rpn_cls_score_p6, dim=2)
            rpn_bbox_loss = mx.sym.Concat(rpn_bbox_loss_p2, rpn_bbox_loss_p3, rpn_bbox_loss_p4, rpn_bbox_loss_p5, rpn_bbox_loss_p6, dim=2)
            # RPN classification loss
            rpn_cls_output = mx.sym.SoftmaxOutput(data=rpn_cls_score, label=rpn_label, multi_output=True, normalization='valid',
                                                  use_ignore=True, ignore_label=-1, name='rpn_cls_prob')
            # bounding box regression
            rpn_bbox_loss = rpn_bbox_weight * mx.sym.smooth_l1(name='rpn_bbox_loss_l1', scalar=3.0, data=(rpn_bbox_loss - rpn_bbox_target))
            rpn_bbox_loss = mx.sym.MakeLoss(name='rpn_bbox_loss', data=rpn_bbox_loss, grad_scale=1.0 / cfg.TRAIN.RPN_BATCH_SIZE)

            aux_dict = {
                'op_type': 'pyramid_proposal', 'name': 'rois',
                'im_info': im_info, 'feat_stride': tuple(cfg.network.RPN_FEAT_STRIDE),
                'scales': tuple(cfg.network.ANCHOR_SCALES), 'ratios': tuple(cfg.network.ANCHOR_RATIOS),
                'rpn_pre_nms_top_n': cfg.TRAIN.RPN_PRE_NMS_TOP_N, 'rpn_post_nms_top_n': cfg.TRAIN.RPN_POST_NMS_TOP_N,
                'threshold': cfg.TRAIN.RPN_NMS_THRESH, 'rpn_min_size': cfg.TRAIN.RPN_MIN_SIZE
            }

            # ROI proposal
            rois = mx.sym.Custom(**dict(arg_dict.items() + aux_dict.items()))
            # ROI proposal target
            gt_boxes_reshape = mx.sym.Reshape(data=gt_boxes, shape=(-1, 5), name='gt_boxes_reshape')
            rois, label, bbox_target, bbox_weight \
                = mx.sym.Custom(rois=rois, gt_boxes=gt_boxes_reshape, op_type='proposal_target', num_classes=num_reg_classes, batch_images=cfg.TRAIN.BATCH_IMAGES,
                                batch_rois=cfg.TRAIN.BATCH_ROIS, cfg=cPickle.dumps(cfg), fg_fraction=cfg.TRAIN.FG_FRACTION)
        else:
            aux_dict = {
                'op_type': 'pyramid_proposal', 'name': 'rois',
                'im_info': im_info, 'feat_stride': tuple(cfg.network.RPN_FEAT_STRIDE),
                'scales': tuple(cfg.network.ANCHOR_SCALES), 'ratios': tuple(cfg.network.ANCHOR_RATIOS),
                'rpn_pre_nms_top_n': cfg.TEST.RPN_PRE_NMS_TOP_N, 'rpn_post_nms_top_n': cfg.TEST.RPN_POST_NMS_TOP_N,
                'threshold': cfg.TEST.RPN_NMS_THRESH, 'rpn_min_size': cfg.TEST.RPN_MIN_SIZE
            }
            # ROI proposal
            rois = mx.sym.Custom(**dict(arg_dict.items() + aux_dict.items()))

        roi_pool = mx.symbol.Custom(data_p2=fpn_p2, data_p3=fpn_p3, data_p4=fpn_p4, data_p5=fpn_p5,
                                    rois=rois, op_type='fpn_roi_pooling', name='fpn_roi_pooling')

        # 2 fc
        fc_new_1 = mx.symbol.FullyConnected(name='fc_new_1', data=roi_pool, num_hidden=1024)
        fc_new_1_relu = mx.sym.Activation(data=fc_new_1, act_type='relu', name='fc_new_1_relu')

        fc_new_2 = mx.symbol.FullyConnected(name='fc_new_2', data=fc_new_1_relu, num_hidden=1024)
        fc_new_2_relu = mx.sym.Activation(data=fc_new_2, act_type='relu', name='fc_new_2_relu')

        # cls_score/bbox_pred
        cls_score = mx.symbol.FullyConnected(name='cls_score', data=fc_new_2_relu, num_hidden=num_classes)
        bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=fc_new_2_relu, num_hidden=num_reg_classes * 4)

        if is_train:
            if cfg.TRAIN.ENABLE_OHEM:
                labels_ohem, bbox_weights_ohem = mx.sym.Custom(op_type='BoxAnnotatorOHEM', num_classes=num_classes,
                                                               num_reg_classes=num_reg_classes, roi_per_img=cfg.TRAIN.BATCH_ROIS_OHEM,
                                                               cls_score=cls_score, bbox_pred=bbox_pred, labels=label,
                                                               bbox_targets=bbox_target, bbox_weights=bbox_weight)
                cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=labels_ohem, normalization='valid', use_ignore=True, ignore_label=-1)
                bbox_loss_ = bbox_weights_ohem * mx.sym.smooth_l1(name='bbox_loss_', scalar=1.0, data=(bbox_pred - bbox_target))
                bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / cfg.TRAIN.BATCH_ROIS_OHEM)
                rcnn_label = labels_ohem
            else:
                cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=label, normalization='valid')
                bbox_loss_ = bbox_weight * mx.sym.smooth_l1(name='bbox_loss_', scalar=1.0, data=(bbox_pred - bbox_target))
                bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / cfg.TRAIN.BATCH_ROIS)
                rcnn_label = label

            # reshape output
            rcnn_label = mx.sym.Reshape(data=rcnn_label, shape=(cfg.TRAIN.BATCH_IMAGES, -1), name='label_reshape')
            cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TRAIN.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape')
            bbox_loss = mx.sym.Reshape(data=bbox_loss, shape=(cfg.TRAIN.BATCH_IMAGES, -1, 4 * num_reg_classes), name='bbox_loss_reshape')
            # group = mx.sym.Group([rpn_cls_output, rpn_bbox_loss, mx.sym.BlockGrad(cls_prob), mx.sym.BlockGrad(bbox_loss), mx.sym.BlockGrad(rcnn_label)])
            group = mx.sym.Group([rpn_cls_output, rpn_bbox_loss, cls_prob, bbox_loss, mx.sym.BlockGrad(rcnn_label)])
        else:
            cls_prob = mx.sym.SoftmaxActivation(name='cls_prob', data=cls_score)
            cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TEST.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape')
            bbox_pred = mx.sym.Reshape(data=bbox_pred, shape=(cfg.TEST.BATCH_IMAGES, -1, 4 * num_reg_classes), name='bbox_pred_reshape')
            group = mx.sym.Group([rois, cls_prob, bbox_pred])

        self.sym = group
        return group

    def init_weight_rcnn(self, cfg, arg_params, aux_params):
        arg_params['fc_new_1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fc_new_1_weight'])
        arg_params['fc_new_1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fc_new_1_bias'])
        arg_params['fc_new_2_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fc_new_2_weight'])
        arg_params['fc_new_2_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fc_new_2_bias'])
        arg_params['cls_score_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['cls_score_weight'])
        arg_params['cls_score_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['cls_score_bias'])
        arg_params['bbox_pred_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['bbox_pred_weight'])
        arg_params['bbox_pred_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['bbox_pred_bias'])

    def init_weight_fpn(self, cfg, arg_params, aux_params):
        arg_params['fpn_p6_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fpn_p6_weight'])
        arg_params['fpn_p6_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fpn_p6_bias'])
        arg_params['fpn_p5_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fpn_p5_weight'])
        arg_params['fpn_p5_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fpn_p5_bias'])
        arg_params['fpn_p4_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fpn_p4_weight'])
        arg_params['fpn_p4_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fpn_p4_bias'])
        arg_params['fpn_p3_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fpn_p3_weight'])
        arg_params['fpn_p3_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fpn_p3_bias'])
        arg_params['fpn_p2_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fpn_p2_weight'])
        arg_params['fpn_p2_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fpn_p2_bias'])

        arg_params['fpn_p5_1x1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fpn_p5_1x1_weight'])
        arg_params['fpn_p5_1x1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fpn_p5_1x1_bias'])
        arg_params['fpn_p4_1x1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fpn_p4_1x1_weight'])
        arg_params['fpn_p4_1x1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fpn_p4_1x1_bias'])
        arg_params['fpn_p3_1x1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fpn_p3_1x1_weight'])
        arg_params['fpn_p3_1x1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fpn_p3_1x1_bias'])
        arg_params['fpn_p2_1x1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fpn_p2_1x1_weight'])
        arg_params['fpn_p2_1x1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fpn_p2_1x1_bias'])

    def init_weight(self, cfg, arg_params, aux_params):
        for name in self.shared_param_list:
            arg_params[name + '_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict[name + '_weight'])
            arg_params[name + '_bias'] = mx.nd.zeros(shape=self.arg_shape_dict[name + '_bias'])
        self.init_weight_rcnn(cfg, arg_params, aux_params)
        self.init_weight_fpn(cfg, arg_params, aux_params)
    