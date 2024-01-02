#!/usr/bin/env python3
"""
ONNX model verification of perception models for OmniDet.

#usage: ./verify_onnx_models.py --config data/params.yaml

# author: Varun Ravi Kumar <rvarun7777@gmail.com>

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.
"""

import os
import sys
import time
import argparse
from typing import Optional
#import click
import cv2
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import onnxruntime
import torch
from PIL import Image
from matplotlib import pyplot as plt
import gray2color


#@click.version_option(version="1.0.0")
#@click.command()
#@click.option('-m', '--model', type=str, help='Input onnx model(.onnx) file')
#@click.option('-i', '--image', type=str, help='Input image file.')
#@on_exception_exit

mean = [128, 128, 128]
var = [128, 128, 128]
# mean = [123.675, 116.28, 103.53]
# var = [58.395, 57.12, 57.375]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        # help="training step in: %s" % TRAINING_STEPS,
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        # help="train config file path",
    )
    parser.add_argument(
    "--npz",
    type=str,
    required=False,
    # help="train config file path",
    )
    return parser.parse_args()





def main():
    args=parse_args()

    npz_file=np.load(args.npz)
    npz=npz_file['input']
    print(f"npz shape: {npz.shape}      npz.dtype: {npz.dtype}")
    # npz=np.expand_dims(npz,axis=0)
    print(f"npz shape: {npz.shape}      npz.dtype: {npz.dtype}")
    npz=npz.astype(np.float32)

    # image=cv2.imread(args.image)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = cv2.resize(image, (896, 896), interpolation=cv2.INTER_LINEAR)

    image=Image.open(args.image).convert('RGB').resize([896,896])
    # image=Image.open(args.image)
    # img=np.array(image,dtype="float32")
    img=np.array(image)
    img=cv2.cvtColor(img,cv2.COLOR_RGB2YUV)
    img= (img-mean)/var
    img=img.astype(np.float32)

    img = img.transpose(2,0,1)
    img_copy=img.copy()
    # img= (img-mean)/var
    # img=img.astype(np.float32)
    # img = img.transpose(2,0,1)
    # img_copy=img.copy()
    img=np.expand_dims(img,axis=0)

    # img= (img-mean)/var

    imgt=torch.from_numpy(img_copy)
    imgt=imgt.unsqueeze(0)
    ort_session = onnxruntime.InferenceSession(args.model,providers=['TensorrtExecutionProvider','CUDAExecutionProvider'])
    print(ort_session.get_providers())
    input = ort_session.get_inputs()[0].name
    output_name=ort_session.get_outputs()[0].name
    output = ort_session.run([output_name], {input:npz})
    print(type(output))
    print(output[0].shape)
    output_array=output[0]
    output_array=output_array.squeeze()
    out_img=Image.fromarray(output_array.astype(np.uint8))
    out_img.save('infer_outputnamed.jpg')

    gt_gray='/home/yzh/work/semidrive/infer_outputnamed.jpg'
    save_path='/home/yzh/work/semidrive/'
    image_gray = Image.open(gt_gray).convert('RGB')
    width,heigth = image_gray.size
    print(f"width:{width}   height:{heigth}")
    mask_np = image_gray.load()
    print(f"mask_np type: {type(mask_np)}")
    # print(np.unique(seg))
    gray2color.convert_color(mask_np,width,heigth)
    image_gray.save( os.path.join(save_path, 'infer_named_color4.png') )


if __name__ == "__main__":
    # load your predefined ONNX model
    main()
