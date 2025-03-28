import os
from pathlib import Path
import numpy as np
import torch
import itertools

import cv2

from video_depth_anything.video_depth import OptimizedVideoDepthAnything

from img_proc.image import loadImage
from img_proc.depth_map import colorize_depth

import gc

def vda_inference(
        input_image_paths : list,
        output_path: str,
        pretrained_model : str):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load sample image to get dimensions and define placeholder
    sample_frame, h, w, par = loadImage(str(input_image_paths[0]), incolorspace='acescg')
    c = sample_frame.shape[2]
    frames = np.empty((len(input_image_paths), h, w, c), dtype=np.uint8)

    # load images
    for idx, image_path in enumerate(input_image_paths):
        frame, _, _, _ = loadImage(str(image_path), incolorspace='acescg')
        frames[idx] = np.round(frame*255.0).astype(np.uint8)

    org_video_len = frames.shape[0]

    # load model
    model_config = {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
    model = OptimizedVideoDepthAnything(**model_config)
    model.load_state_dict(torch.load(pretrained_model, map_location='cpu'), strict=True)
    model = model.to(device).eval()

    # inference
    depths = model.infer_video_depth(frames, h, w, org_video_len, input_size=518, device=device, fp32=False)

    # clear frames to save memory
    del frames
    gc.collect()

    # align depths for temporal consistency
    depths = model.align_depths(depths, org_video_len)

    # save images
    for idx, image_path in enumerate(input_image_paths):

        image_stem = str(image_path.stem)

        vis_path = Path(output_path, "depth_vis")
        vis_path.mkdir(exist_ok=True, parents=True)
        vis_file_path = image_stem + "_depth_vis.png"

        depth_path = Path(output_path, "depth")
        depth_path.mkdir(exist_ok=True, parents=True)
        depth_file_path = image_stem + "_depth.exr"

        # save color mapped depth for visualization
        depth = depths[idx]
        depth_vis = colorize_depth(1/depth, mask=None, normalize=True, cmap = 'Spectral')
        cv2.imwrite(str(vis_path / vis_file_path), cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR))

        # save exr depth maps
        cv2.imwrite(str(depth_path / depth_file_path), depth, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])

    # clear depths to save memory
    del depths
    gc.collect()
