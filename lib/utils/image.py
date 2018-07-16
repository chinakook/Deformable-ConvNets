import numpy as np
import os
import cv2
import random
from PIL import Image
from bbox.bbox_transform import clip_boxes


# TODO: This two functions should be merged with individual data loader
def get_image(roidb, config):
    """
    preprocess image and return processed roidb
    :param roidb: a list of roidb
    :return: list of img as in mxnet format
    roidb add new item['im_info']
    0 --- x (width, second dim of im)
    |
    y (height, first dim of im)
    """
    num_images = len(roidb)
    processed_ims = []
    processed_roidb = []
    for i in range(num_images):
        roi_rec = roidb[i]
        assert os.path.exists(roi_rec['image']), '%s does not exist'.format(roi_rec['image'])
        im = cv2.imread(roi_rec['image'], cv2.IMREAD_COLOR) #|cv2.IMREAD_IGNORE_ORIENTATION)
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        new_rec = roi_rec.copy()
        scale_ind = random.randrange(len(config.SCALES))
        target_size = config.SCALES[scale_ind][0]
        max_size = config.SCALES[scale_ind][1]
        im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        im_tensor = transform(im, config.network.PIXEL_MEANS, config.network.INV_STD)
        processed_ims.append(im_tensor)
        im_info = [im_tensor.shape[2], im_tensor.shape[3], im_scale]
        new_rec['boxes'] = clip_boxes(np.round(roi_rec['boxes'].copy() * im_scale), im_info[:2])
        new_rec['im_info'] = im_info
        processed_roidb.append(new_rec)
    return processed_ims, processed_roidb


def get_segmentation_image(segdb, config):
    """
    propocess image and return segdb
    :param segdb: a list of segdb
    :return: list of img as mxnet format
    """
    num_images = len(segdb)
    assert num_images > 0, 'No images'
    processed_ims = []
    processed_segdb = []
    processed_seg_cls_gt = []
    for i in range(num_images):
        seg_rec = segdb[i]
        assert os.path.exists(seg_rec['image']), '%s does not exist'.format(seg_rec['image'])
        im = np.array(cv2.imread(seg_rec['image']))

        new_rec = seg_rec.copy()

        scale_ind = random.randrange(len(config.SCALES))
        target_size = config.SCALES[scale_ind][0]
        max_size = config.SCALES[scale_ind][1]
        im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        im_tensor = transform(im, config.network.PIXEL_MEANS, config.network.INV_STD)
        im_info = [im_tensor.shape[2], im_tensor.shape[3], im_scale]
        new_rec['im_info'] = im_info

        seg_cls_gt = np.array(Image.open(seg_rec['seg_cls_path']))
        seg_cls_gt, seg_cls_gt_scale = resize(
            seg_cls_gt, target_size, max_size, stride=config.network.IMAGE_STRIDE, interpolation=cv2.INTER_NEAREST)
        seg_cls_gt_tensor = transform_seg_gt(seg_cls_gt)

        processed_ims.append(im_tensor)
        processed_segdb.append(new_rec)
        processed_seg_cls_gt.append(seg_cls_gt_tensor)

    return processed_ims, processed_seg_cls_gt, processed_segdb

def get_cropped_image(roidb, config):
    """
    preprocess image and return processed roidb
    :param roidb: a list of roidb
    :return: list of img as in mxnet format
    roidb add new item['im_info']
    0 --- x (width, second dim of im)
    |
    y (height, first dim of im)
    """
    def cal_iou(in_rect,gt_rect):
        assert (in_rect.shape[0] == gt_rect.shape[0]),"rect_size must be the same"
        iou_rct =  []
        for i in range(in_rect.shape[0]):
            i_rect = in_rect[i]
            g_rect = gt_rect[i]
            iou = float(((i_rect[2] - i_rect[0])*(i_rect[3] - i_rect[1])))/((g_rect[2] - g_rect[0])*(g_rect[3] - g_rect[1]) + 1)
            iou_rct.append(iou)
        return np.array(iou_rct)

    def crop_batch(roidb, config):
        assert os.path.exists(roidb['image']), '%s does not exist'.format(roidb['image'])
        im = cv2.imread(roidb['image'], cv2.IMREAD_COLOR) #|cv2.IMREAD_IGNORE_ORIENTATION)
        if roidb['flipped']:
            im = im[:, ::-1, :]
        label = roidb['boxes'].copy()

        scale_ind = random.randrange(len(config.SCALES))
        fix_size = config.SCALES[scale_ind][0]

        zoom = random.choice([0.5, 1, 2.0])
        if zoom < 1 or zoom > 1:
            im = cv2.resize(im,None,fx=zoom,fy=zoom,interpolation=cv2.INTER_CUBIC)

            pad_b = 0
            pad_r = 0
            nh, nw = im.shape[0], im.shape[1]
            if nh < fix_size:
                pad_b = fix_size - nh
                nh = fix_size
            if nw < fix_size:
                pad_r = fix_size - nw
                nw = fix_size

            im = cv2.copyMakeBorder(im, 0, pad_b, 0, pad_r, cv2.BORDER_CONSTANT, None, (127,127,127))

            label = label * zoom

        size = im.shape
        width = size[1]
        height = size[0]

        x_max = width - fix_size
        y_max = height  - fix_size
        label = label.astype(np.int64)
        rects=[]
        for j in range(100):

            x_g = 0 if x_max == 0 else np.random.randint(0,x_max)
            y_g = 0 if y_max == 0 else np.random.randint(0,y_max)
            x_y = np.array([x_g,y_g,x_g,y_g])
            label_g = label - x_y
            label_g[np.where(label_g<0)] = 0
            label_g[np.where(label_g>(fix_size-1))] = fix_size-1
            iou_rct = cal_iou(label_g,label)
            valid_num =  (iou_rct>0.7).sum()
            img_sub = im[y_g:y_g+fix_size,x_g:x_g+fix_size,:]
            if valid_num>0:
                idx = np.where(iou_rct>0.7)
                label_sub = label_g[idx[0]]
                rects = label_sub
                break
        return img_sub,np.array(rects)

    num_images = len(roidb)
    processed_ims = []
    processed_roidb = []
    for i in range(num_images):
        roi_rec = roidb[i]
        new_rec = roi_rec.copy()
        im_scale = 1.0
        im, box = crop_batch(new_rec, config)
        im_tensor = transform(im, config.network.PIXEL_MEANS, config.network.INV_STD)
        processed_ims.append(im_tensor)
        im_info = [im_tensor.shape[2], im_tensor.shape[3], im_scale]
        new_rec['gt_classes']=[]
        b_size = (box.size)/4
        for j in range(b_size):
            (new_rec['gt_classes']).append(1)
        new_rec['gt_classes'] = np.array(new_rec['gt_classes'])
        new_rec['boxes'] = box
        new_rec['im_info'] = im_info
        processed_roidb.append(new_rec)
    return processed_ims, processed_roidb


def resize(im, target_size, max_size, stride=0, interpolation = cv2.INTER_LINEAR):
    """
    only resize input image to target size and return scale
    :param im: BGR image input by opencv
    :param target_size: one dimensional size (the short side)
    :param max_size: one dimensional max size (the long side)
    :param stride: if given, pad the image to designated stride
    :param interpolation: if given, using given interpolation method to resize image
    :return:
    """
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=interpolation)

    if stride == 0:
        return im, im_scale
    else:
        # pad to product of stride
        im_height = int(np.ceil(im.shape[0] / float(stride)) * stride)
        im_width = int(np.ceil(im.shape[1] / float(stride)) * stride)
        im_channel = im.shape[2]
        padded_im = np.zeros((im_height, im_width, im_channel))
        padded_im[:im.shape[0], :im.shape[1], :] = im
        return padded_im, im_scale

def transform(im, pixel_means, inv_std=1.):
    """
    transform into mxnet tensor
    substract pixel size and transform to correct format
    :param im: [height, width, channel] in BGR
    :param pixel_means: [B, G, R pixel means]
    :return: [batch, channel, height, width]
    """
    im_tensor = np.zeros((1, 3, im.shape[0], im.shape[1]))
    for i in range(3):
        im_tensor[0, i, :, :] = inv_std * (im[:, :, 2 - i] - pixel_means[2 - i])
    return im_tensor

def transform_seg_gt(gt):
    """
    transform segmentation gt image into mxnet tensor
    :param gt: [height, width, channel = 1]
    :return: [batch, channel = 1, height, width]
    """
    gt_tensor = np.zeros((1, 1, gt.shape[0], gt.shape[1]))
    gt_tensor[0, 0, :, :] = gt[:, :]

    return gt_tensor

def transform_inverse(im_tensor, pixel_means, inv_std=1.):
    """
    transform from mxnet im_tensor to ordinary RGB image
    im_tensor is limited to one image
    :param im_tensor: [batch, channel, height, width]
    :param pixel_means: [B, G, R pixel means]
    :return: im [height, width, channel(RGB)]
    """
    assert im_tensor.shape[0] == 1
    im_tensor = im_tensor.copy()
    # put channel back
    channel_swap = (0, 2, 3, 1)
    im_tensor = im_tensor.transpose(channel_swap)
    im = im_tensor[0]
    assert im.shape[2] == 3
    im *= (1/inv_std)
    im += pixel_means[[2, 1, 0]]
    im = im.astype(np.uint8)
    return im

def tensor_vstack(tensor_list, pad=0):
    """
    vertically stack tensors
    :param tensor_list: list of tensor to be stacked vertically
    :param pad: label to pad with
    :return: tensor with max shape
    """
    ndim = len(tensor_list[0].shape)
    dtype = tensor_list[0].dtype
    islice = tensor_list[0].shape[0]
    dimensions = []
    first_dim = sum([tensor.shape[0] for tensor in tensor_list])
    dimensions.append(first_dim)
    for dim in range(1, ndim):
        dimensions.append(max([tensor.shape[dim] for tensor in tensor_list]))
    if pad == 0:
        all_tensor = np.zeros(tuple(dimensions), dtype=dtype)
    elif pad == 1:
        all_tensor = np.ones(tuple(dimensions), dtype=dtype)
    else:
        all_tensor = np.full(tuple(dimensions), pad, dtype=dtype)
    if ndim == 1:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice] = tensor
    elif ndim == 2:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice, :tensor.shape[1]] = tensor
    elif ndim == 3:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice, :tensor.shape[1], :tensor.shape[2]] = tensor
    elif ndim == 4:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice, :tensor.shape[1], :tensor.shape[2], :tensor.shape[3]] = tensor
    else:
        raise Exception('Sorry, unimplemented.')
    return all_tensor
