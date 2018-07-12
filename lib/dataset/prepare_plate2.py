# coding: utf-8
import os
import cv2
import numpy as np
import shutil

from xml.etree import ElementTree as ET
import codecs

def getfiles(path):
    l = []
    for root,dirs,paths in os.walk(path):
        for p in paths:
            l.append(os.path.join(root,p))
    limg = [i for i in l if i.endswith('.JPG') or i.endswith('.jpg') or i.endswith('.png') or i.endswith('.PNG')]
    lgt = [i[:-4]+'.xml' for i in limg]
    return limg, lgt

path = u'/mnt/15F1B72E1A7798FD/DK2/plates/plate_samples'

dst_dir = u'/home/dingkou/dev/Deformable-ConvNets/data/plates'

if os.path.exists(dst_dir) and os.path.isdir(dst_dir):
    shutil.rmtree(dst_dir)

os.mkdir(dst_dir)
os.mkdir(dst_dir + '/train')

anno_dir = dst_dir + '/train/Annotations'
img_dir = dst_dir + '/train/JPEGImages'
main_dir = dst_dir + '/train/Main'

os.mkdir(anno_dir)
os.mkdir(img_dir)
os.mkdir(main_dir)
    

imgs, gts = getfiles(path)

MAX_DST_SIZE=2448
MIN_PAD_SIZE=512

fmain = open(main_dir+'/train.txt', 'w+')

for i, fn in enumerate(imgs):
    img = cv2.imdecode(np.fromfile(fn, dtype=np.uint8),-1)
    h,w = img.shape[0], img.shape[1]
    
    anno = ET.parse(gts[i])
    obj_node=anno.getiterator("object")

    maxdim = max(w,h)

    if maxdim > MAX_DST_SIZE:
        s = MAX_DST_SIZE / float(maxdim)

        img = cv2.resize(img, None, None, s, s)
        for obj in obj_node:
            bndbox = obj.find('bndbox')
            xmin = bndbox.find('xmin')
            ymin = bndbox.find('ymin')
            xmax = bndbox.find('xmax')
            ymax = bndbox.find('ymax')
            xmin.text = str(int(int(xmin.text) * s))
            ymin.text = str(int(int(ymin.text) * s))
            xmax.text = str(int(int(xmax.text) * s))
            ymax.text = str(int(int(ymax.text) * s))

    nh, nw = img.shape[0], img.shape[1]

    pad_b = 0
    pad_r = 0
    if nh < MIN_PAD_SIZE:
        pad_b = MIN_PAD_SIZE - nh
        nh = MIN_PAD_SIZE
    if nw < MIN_PAD_SIZE:
        pad_r = MIN_PAD_SIZE - nw
        nw = MIN_PAD_SIZE

    img = cv2.copyMakeBorder(img, 0, pad_b, 0, pad_r, cv2.BORDER_CONSTANT, None, (127,127,127))
    size_node=anno.find("size")
    width_node = size_node.find("width")
    height_node = size_node.find("height")
    width_node.text = str(nw)
    height_node.text = str(nh)
    
    # print(width_node.text, height_node.text)
    # print(fn, gts[i])
    # for obj in obj_node:
    #     bndbox = obj.find('bndbox')
    #     xmin = bndbox.find('xmin')
    #     ymin = bndbox.find('ymin')
    #     xmax = bndbox.find('xmax')
    #     ymax = bndbox.find('ymax')
    #     cv2.rectangle(img, (int(xmin.text), int(ymin.text)), (int(xmax.text), int(ymax.text)), (0,0,255), 4)
        
    anno.write('%s/%04d.xml' % (anno_dir, i))
    fmain.write('%04d\n' % i)
    
    #shutil.copy(gts[i], '%04d.xml' % i)
    cv2.imwrite('%s/%04d.jpg' % (img_dir, i), img)

    # cv2.imshow("w", img)
    # cv2.waitKey()

    

fmain.close()