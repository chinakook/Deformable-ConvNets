import _init_paths

import os
import pprint
import argparse
import mxnet as mx
import json

from kktools.merge_conv_bn import merge

from symbols import *

from config.config import config, update_config
from utils.load_model import load_param
from utils.create_logger import create_logger

def parse_args():
    parser = argparse.ArgumentParser(description='Deploy a Faster R-CNN network')
    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--rpn', help='deploy rpn only', default=0, type=bool)

    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    pprint.pprint(config)

    sym_instance = eval(config.symbol + '.' + config.symbol)()
    sym_deploy_instance = eval(config.symbol + '_deploy.' + config.symbol)()
    if args.rpn:
        sym = sym_instance.get_symbol_rpn(config, is_train=False)
        sym_deploy = sym_deploy_instance.get_symbol_rpn(config, is_train=False)
    else:
        if config.TEST.HAS_RPN:
            sym = sym_instance.get_symbol(config, is_train=False)
            sym_deploy = sym_deploy_instance.get_symbol(config, is_train=False)
        else:
            sym = sym_instance.get_symbol_rcnn(config, is_train=False)

    #dotgraph = mx.viz.plot_network(sym, shape={'data' : (1,3,224,224), 'im_info' : (1,3)}, save_format='png')
    #dotgraph.render(config.symbol)

    _, final_output_path = create_logger(config.output_path, args.cfg, config.dataset.test_image_set)
    prefix = os.path.join(final_output_path, '..', '_'.join([iset for iset in config.dataset.image_set.split('+')]), config.TRAIN.model_prefix)
    arg_params, aux_params = load_param(prefix, config.TEST.test_epoch, process=True)
    
    arg_deploy = merge(sym, arg_params, aux_params)

    data_names = ['data', 'im_info']
    label_names = None

    mod = mx.mod.Module(symbol=sym_deploy, context=mx.gpu(0), data_names=data_names, label_names=label_names)
    mod.bind(for_training=False, data_shapes=[('data', (1, 3, 1024, 1024)), ('im_info', (1, 3))], label_shapes=None, force_rebind=False)
    mod.set_params(arg_params=arg_deploy, aux_params=aux_params, force_init=False)

    mod.save_checkpoint('test_plate',0)

if __name__ == '__main__':
    main()
