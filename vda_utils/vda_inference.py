import os
from pathlib import Path
import numpy as np
import torch
import itertools

import cv2

from video_depth_anything.video_depth import VideoDepthAnything

from img_proc.image import loadImage, writeImage
from img_proc.depth_map import colorize_depth


def vda_inference(
        input_path : str,
        output_path: str,
        pretrained_model : str,
        extension : str):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    include_suffices = [extension.lower(), extension.upper()]

    if Path(input_path).is_dir():
        image_paths = sorted(itertools.chain(*(Path(input_path).rglob(f'*.{suffix}') for suffix in include_suffices)))
        # TODO : sort paths using Pyseq ?

    else:
        raise ValueError(f"Input path '{input_path}' is not a directory.")
    
    if len(image_paths) == 0:
        raise FileNotFoundError(f'No image files found in {input_path}')
    
    # load images
    frames = []

    for image_path in image_paths:
        frame, h, w, par = loadImage(str(image_path), incolorspace='acescg')
        frames.append(np.round(frame*255.0))
    frames = np.stack(frames, axis=0)

    # load model
    model_config = {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
    model = VideoDepthAnything(**model_config)
    model.load_state_dict(torch.load(pretrained_model, map_location='cpu'), strict=True)
    model = model.to(device).eval()

    # inference
    depths, _ = model.infer_video_depth(frames, 24.0, input_size=518, device=device, fp32=False)

    # save images
    for idx, image_path in enumerate(image_paths):
        save_path = Path(output_path, image_path.relative_to(input_path).parent, image_path.stem)
        save_path.mkdir(exist_ok=True, parents=True)

        # save color mapped depth for visualization
        depth = depths[idx]
        depth_vis = colorize_depth(1/depth, mask=None, normalize=True, cmap = 'Spectral')
        writeImage(str(save_path / 'depth_vis.png'), depth_vis, h_tgt=h, w_tgt=w, pixelAspectRatio=par)

        # save exr depth maps
        cv2.imwrite(str(save_path / 'depth.exr'), depth, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
