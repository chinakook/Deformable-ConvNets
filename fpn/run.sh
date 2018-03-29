MXNET_CUDNN_AUTOTUNE_DEFAULT=0 python fpn/train_end2end.py --cfg fpn/traffic_sign2.yaml
MXNET_CUDNN_AUTOTUNE_DEFAULT=0 python fpn/deploy.py --cfg fpn/traffic_sign2.yaml
cd fpn
MXNET_CUDNN_AUTOTUNE_DEFAULT=0 python demo.py