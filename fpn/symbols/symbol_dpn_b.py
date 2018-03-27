import mxnet as mx
from symbol_basic import *

# - - - - - - - - - - - - - - - - - - - - - - -
# Standard Dual Path Unit
def DualPathFactory(data, num_1x1_a, num_3x3_b, num_1x1_c, name, inc, G, _type='normal'):
    kw = 3
    kh = 3
    pw = (kw-1)/2
    ph = (kh-1)/2

    # type
    if _type is 'proj':
        key_stride = 1
        has_proj   = True
    if _type is 'down':
        key_stride = 2
        has_proj   = True
    if _type is 'normal':
        key_stride = 1
        has_proj   = False

    # PROJ
    if type(data) is list:
        data_in  = mx.symbol.Concat(*[data[0], data[1]],  name=('%s_cat-input' % name))
    else:
        data_in  = data

    if has_proj:
        _, c1x1_w   = BN_AC_Conv( data=data_in, num_filter=(num_1x1_c+2*inc), kernel=( 1, 1), stride=(key_stride, key_stride), name=('%s_c1x1-w(s/%d)' %(name, key_stride)), pad=(0, 0))
        data_o1  = mx.symbol.slice_axis(data=c1x1_w, axis=1, begin=0,         end=num_1x1_c,         name=('%s_c1x1-w(s/%d)-split1' %(name, key_stride)))
        data_o2  = mx.symbol.slice_axis(data=c1x1_w, axis=1, begin=num_1x1_c, end=(num_1x1_c+2*inc), name=('%s_c1x1-w(s/%d)-split2' %(name, key_stride)))
    else:
        data_o1  = data[0]
        data_o2  = data[1]
        
    # MAIN
    _, c1x1_a = BN_AC_Conv( data=data_in, num_filter=num_1x1_a,       kernel=( 1,  1), pad=( 0,  0), name=('%s_c1x1-a'   % name))
    c1x1_bn_ac, c3x3_b = BN_AC_Conv( data=c1x1_a,  num_filter=num_3x3_b,       kernel=(kw, kh), pad=(pw, ph), name=('%s_c%dx%d-b' % (name,kw,kh)), stride=(key_stride,key_stride), num_group=G)
    c1x1_c = BN_AC( data=c3x3_b,  name=('%s_c1x1-c'  % name))
    c1x1_c1= Conv(  data=c1x1_c,  num_filter=num_1x1_c, kernel=( 1,  1), name=('%s_c1x1-c1' % name),         pad=( 0,  0))
    c1x1_c2= Conv(  data=c1x1_c,  num_filter=inc,       kernel=( 1,  1), name=('%s_c1x1-c2' % name),         pad=( 0,  0))
    
    # OUTPUTS
    summ   = mx.symbol.ElementWiseSum(*[data_o1, c1x1_c1],                        name=('%s_sum' % name))
    dense  = mx.symbol.Concat(        *[data_o2, c1x1_c2],                        name=('%s_cat' % name))

    return [summ, dense, c1x1_bn_ac]


