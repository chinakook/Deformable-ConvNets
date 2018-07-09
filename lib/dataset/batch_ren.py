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
    lgt = [i for i in l if i.endswith('.xml')]
    return limg, lgt

path = u'/mnt/15F1B72E1A7798FD/DK2/plates/plate_samples'

imgs, gts = getfiles(path)

DST_SIZE=1024

for i, fn in enumerate(imgs):
    img = cv2.imdecode(np.fromfile(fn, dtype=np.uint8),-1)
    h,w = img.shape[0], img.shape[1]
    
    anno = ET.parse(gts[i])
    obj_node=anno.getiterator("object")

    if h > DST_SIZE or w > DST_SIZE:
        if h > w:
            img = cv2.copyMakeBorder(img, 0, 0, 0, h-w, cv2.BORDER_CONSTANT, value=(128,128,128))
            s = DST_SIZE / float(h)
        else:
            img = cv2.copyMakeBorder(img, 0, w-h, 0, 0, cv2.BORDER_CONSTANT, value=(128,128,128))
            s = DST_SIZE / float(w)
        img = cv2.resize(img, (DST_SIZE, DST_SIZE))
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
#             cv2.rectangle(img, (int(xmin.text), int(ymin.text)), (int(xmax.text), int(ymax.text)), (0,0,255), 4)

    else:
        if h < DST_SIZE:
            bottom = DST_SIZE - h
        if w < DST_SIZE:
            right = DST_SIZE - w
        img = cv2.copyMakeBorder(img, 0, bottom, 0, right, cv2.BORDER_CONSTANT, value=(128,128,128))
#         for obj in obj_node:
#             bndbox = obj.find('bndbox')
#             xmin = bndbox.find('xmin')
#             ymin = bndbox.find('ymin')
#             xmax = bndbox.find('xmax')
#             ymax = bndbox.find('ymax')       
#             cv2.rectangle(img, (int(xmin.text), int(ymin.text)), (int(xmax.text), int(ymax.text)), (0,0,255), 4)
        
    anno.write('%04d.xml' % i)

    
    #shutil.copy(gts[i], '%04d.xml' % i)
    cv2.imwrite('%04d.jpg' % i, img)
