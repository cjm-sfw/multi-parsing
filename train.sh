#!/usr/bin/env sh

cp resnet101-5d3b4d8f.pth /root/.cache/torch/checkpoints/resnet101-5d3b4d8f.pth

python train.py configs/denseposeparsing/densepose_r101_fpn_8gpu_3x.py --gpus 1