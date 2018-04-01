import mxnet as mx
from mxnet import gluon
import mxnet.gluon.model_zoo.vision as vision
import numpy as np


net = vision.resnet50_v2(pretrained=True)

net.hybridize()

data = mx.sym.var('data', shape=(1,3,224,224))
#data=mx.nd.random.uniform(shape=(1,3,224,224))

sym = net(data)

#mx.viz.plot_network(sym, shape={'data': (1,3,224,224)}).view()

#resnetv20_stage2_activation0
#resnetv20_stage3_activation0
#resnetv20_stage4_activation0
#resnetv20_relu1_fwd

mean= np.array([0.485, 0.456, 0.406])

std = np.array([0.229, 0.224, 0.225])

mean2 = 255*mean
inv_std = 1 / (255 * std)

print(mean2, inv_std)

#import netron
#netron.serve_file('resnet50-symbol.json')