from pathlib import Path
import sys

from typing import *
import itertools
import json

import cv2
import numpy as np
import torch
from tqdm import tqdm
import trimesh
import trimesh.visual

from moge.model import MoGeModel
from moge.utils.io import save_glb, save_ply
from moge.utils.vis import colorize_normal
import utils3d

# oiio image loading and conversion if exr
from img_proc.image import loadImage, writeImage
from img_proc.depth_map import colorize_depth


def moge_inference(
    input_image_paths: list,
    fov_x_: float,
    output_path: str,
    pretrained_model: str,
    threshold: float,
    ply: bool):

    device = torch.device('cuda')

    model = MoGeModel.from_pretrained(pretrained_model).to(device).eval()
    
    # inference on each image
    for image_path in (pbar := tqdm(input_image_paths, desc='Inference', disable=len(input_image_paths) <= 1)):
        
        # load images and convert exr from acescg -> sRGB
        img, h_ori, w_ori, par = loadImage(str(image_path), incolorspace='acescg') 
        image_tensor = torch.tensor(img, dtype=torch.float32, device=device).permute(2, 0, 1)
        
        # safe clamp between [0,1] in case of a wrong input cs 
        image_tensor = torch.clamp(image_tensor, 0, 1)
        
        output = model.infer(image_tensor, fov_x=fov_x_)
        points, depth, mask, intrinsics = output['points'].cpu().numpy(), output['depth'].cpu().numpy(), output['mask'].cpu().numpy(), output['intrinsics'].cpu().numpy()
        normals, normals_mask = utils3d.numpy.points_to_normals(points, mask=mask)
        normals = np.nan_to_num(normals, nan=0.0, posinf=1.0, neginf=0.0)

        # Write outputs
        # save_path = Path(output_path , image_path.stem)
        image_stem = str(image_path.stem)

        vis_path = Path(output_path, "depth_vis")
        vis_path.mkdir(exist_ok=True, parents=True)
        vis_file_path = image_stem + "_depth_vis.png"

        normal_path = Path(output_path, "normal")
        normal_path.mkdir(exist_ok=True, parents=True)
        normal_file_path = image_stem + "_normal.png"

        depth_path = Path(output_path, "depth")
        depth_path.mkdir(exist_ok=True, parents=True)
        depth_file_path = image_stem + "_depth.exr"

        cv2.imwrite(str(vis_path / vis_file_path), cv2.cvtColor(colorize_depth(depth), cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(normal_path / normal_file_path), cv2.cvtColor(colorize_normal(normals), cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(depth_path / depth_file_path), 1/depth, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])

        # cv2.imwrite(str(save_path / 'mask.png'), (mask * 255).astype(np.uint8))
        # cv2.imwrite(str(save_path / 'points.exr'), cv2.cvtColor(points, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])

        fov_x, fov_y = utils3d.numpy.intrinsics_to_fov(intrinsics)
        with open(Path(output_path) / 'fov.json', 'w') as f:
            json.dump({
                'fov_x': round(float(np.rad2deg(fov_x)), 2),
                'fov_y': round(float(np.rad2deg(fov_y)), 2),
            }, f)

        if ply:
            ply_path = Path(output_path, "mesh")
            ply_path.mkdir(exist_ok=True, parents=True)
            ply_file_path = image_stem + "_mesh.ply"

            faces, vertices, vertex_colors, vertex_uvs = utils3d.numpy.image_mesh(
                    points,
                    img.astype(np.float32),
                    utils3d.numpy.image_uv(width=w_ori, height=h_ori),
                    mask=mask & ~(utils3d.numpy.depth_edge(depth, rtol=threshold, mask=mask) & utils3d.numpy.normals_edge(normals, tol=5, mask=normals_mask)),
                    tri=True)
            # When exporting the model, follow the OpenGL coordinate conventions:
            # - world coordinate system: x right, y up, z backward.
            # - texture coordinate system: (0, 0) for left-bottom, (1, 1) for right-top.
            vertices, vertex_uvs = vertices * [1, -1, -1], vertex_uvs * [1, -1] + [0, 1]
            
            # save_glb(save_path / 'mesh.glb', vertices, faces, vertex_uvs, img)
            save_ply(ply_path / ply_file_path, vertices, faces, vertex_colors)
