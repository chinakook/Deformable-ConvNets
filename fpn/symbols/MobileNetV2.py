import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon as gluon
import mxnet.gluon.nn as nn
import mxnet.autograd as ag

def ConvBlock(channels, kernel_size, strides, **kwargs):
    out = nn.HybridSequential(**kwargs)
    with out.name_scope():
        out.add(
            nn.Conv2D(channels, kernel_size, strides=strides, padding=1, use_bias=False, prefix=""),
            nn.BatchNorm(scale=True, prefix="bn_"),
            nn.Activation('relu')
        )
    return out

def Conv1x1(channels, is_linear=False, **kwargs):
    out = nn.HybridSequential(**kwargs)
    with out.name_scope():
        out.add(
            nn.Conv2D(channels, 1, padding=0, use_bias=False, prefix=""),
            nn.BatchNorm(scale=True, prefix="bn_")
        )
        if not is_linear:
            out.add(nn.Activation('relu'))
    return out

def DWise(channels, strides, **kwargs):
    out = nn.HybridSequential(**kwargs)
    with out.name_scope():
        out.add(
            nn.Conv2D(channels, 3, strides=strides, padding=1, groups=channels, use_bias=False, prefix=""),
            nn.BatchNorm(scale=True, prefix="bn_"),
            nn.Activation('relu')
        )
    return out

class InvertedResidual(nn.HybridBlock):
    def __init__(self, inp, oup, t, strides, same_shape=True, **kwargs):
        super(InvertedResidual, self).__init__(**kwargs)
        self.same_shape = same_shape
        self.strides = strides
        with self.name_scope(): 
            self.bottleneck = nn.HybridSequential()
            self.bottleneck.add(
                Conv1x1(inp*t, prefix="expand_"),
                DWise(inp*t, self.strides, prefix="dwise_"),
                Conv1x1(oup, is_linear=True, prefix="linear_")
            )
            #if self.stride == 1 and not self.same_shape:
            #    self.conv_res = Conv1x1(oup)
    def hybrid_forward(self, F, x):
        out = self.bottleneck(x)
        if self.strides == 1 and self.same_shape:
            out = F.elemwise_add(out, x)
        # if self.stride == 1:
        #     if not self.same_shape:
        #         x = self.conv_res(x)
        #     out = F.elemwise_add(out, x)
        return out

def InvertedResidualSequence(t, inp, oup, repeats, first_strides, **kwargs):
    seq = nn.HybridSequential(**kwargs)
    with seq.name_scope():
        seq.add(InvertedResidual(inp, oup, t, first_strides, same_shape=False, prefix="1_"))
        curr_inp = oup
        for i in range(1, repeats):
            seq.add(InvertedResidual(curr_inp, oup, t, 1, prefix= "{}_".format(i+1)))
            curr_inp = oup
        return seq

class MobilenetV2(nn.HybridBlock):
    def __init__(self, num_classes=1000, width_mult=1.0, **kwargs):
        super(MobilenetV2, self).__init__(**kwargs)
        
        self.w = width_mult

        self.first_oup = 32 * self.w
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],  # -> 112x112
            [6, 24, 2, 2],  # -> 56x56
            [6, 32, 3, 2],  # -> 28x28
            [6, 64, 4, 2],  # -> 14x14
            [6, 96, 3, 1],  # -> 14x14
            [6, 160, 3, 2], # -> 7x7
            [6, 320, 1, 1], # -> 7x7          
        ]
        self.last_channels = int(1280*self.w) if self.w > 1.0 else 1280

        with self.name_scope():
            self.features = nn.HybridSequential()
            self.features.add(ConvBlock(self.first_oup, 3, 2, prefix="first_conv_"))
            inp = self.first_oup
            for i, (t, c, n, s) in enumerate(self.interverted_residual_setting):
                oup = c * self.w
                self.features.add(InvertedResidualSequence(t, inp, oup, n, s, prefix="stage{}_".format(i+1)))
                inp = oup

            # self.features.add(Conv1x1(self.last_channels))
            # self.features.add(nn.GlobalAvgPool2D())
            # self.features.add(nn.Flatten())
            # self.output = nn.Dense(num_classes)
    def hybrid_forward(self, F, x):
        x = self.features(x)
        #x = self.output(x)
        return x

net = MobilenetV2(1000,1, prefix="")

# save as symbol
data =mx.sym.var('data')
sym = net(data)

from kktools.rf import rf_summery
rfs = rf_summery(sym)
for rf in rfs.items():
    print(rf)
exit(0)

# plot network graph
#mx.viz.print_summary(sym, shape={'data':(8,3,224,224)})
mx.viz.plot_network(sym,shape={'data':(8,3,224,224)}, node_attrs={'shape':'oval','fixedsize':'fasl==false'}).view()


