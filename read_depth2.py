import os
import sys
import torch
import argparse
from datetime import datetime
from plyfile import PlyData, PlyElement
from typing import NamedTuple
from glob import glob
from PIL import Image
from typing import NamedTuple
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
import imageio
from datetime import datetime
from tqdm import tqdm

import numpy as np
import collections
import struct
import matplotlib.pyplot as plt
import re
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array
    errors : np.array = None
    

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))
CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                           for camera_model in CAMERA_MODELS])

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def storePly(path, xyz, rgb, xyzerr=None):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
            ('xyzerr', 'f4')]
    
    normals = np.zeros_like(xyz)
    if xyzerr is None:
        xyzerr = np.ones((xyz.shape[0],1))

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb, xyzerr), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_points3D_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    xyzs = None
    rgbs = None
    errors = None
    num_points = 0
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                num_points += 1


    xyzs = np.empty((num_points, 3))
    rgbs = np.empty((num_points, 3))
    errors = np.empty((num_points, 1))
    count = 0
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = np.array(float(elems[7]))
                xyzs[count] = xyz
                rgbs[count] = rgb
                errors[count] = error
                count += 1

    return xyzs, rgbs, errors

def read_points3D_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """


    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]

        xyzs = np.empty((num_points, 3))
        rgbs = np.empty((num_points, 3))
        errors = np.empty((num_points, 1))

        for p_id in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8*track_length,
                format_char_sequence="ii"*track_length)
            xyzs[p_id] = xyz
            rgbs[p_id] = rgb
            errors[p_id] = error
    return xyzs, rgbs, errors

def read_intrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                assert model == "PINHOLE", "While the loader support other types, the rest of the code assumes PINHOLE"
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras

def read_extrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images


def read_intrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8*num_params,
                                     format_char_sequence="d"*num_params)
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras


def read_extrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                       tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
    return images


def read_colmap_bin_array(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_dense.py

    :param path: path to the colmap binary file.
    :return: nd array with the floating point values in the value
    """
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()

def refineColmapWithIndex(path):
    """ 
    result
    'cam_extrinsics' and 'point3D.ply' contains the points observed in (at least 2) train-views 
    """
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    
    cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
    cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    
    xyz, rgb, err = read_points3D_binary(bin_path)
    total_ptsidxlist, train_ptsidxlist = [], []
    for tidx, key in enumerate(sorted(cam_extrinsics, key=lambda x:cam_extrinsics[x].name)):
        total_ptsidxlist.append(cam_extrinsics[key].point3D_ids) # cam_extrinsics number starts from 1
        train_ptsidxlist.append(cam_extrinsics[key].point3D_ids)
    
    ### valid 2D points (select the points in train-view)
    ptsidx, cnt = np.unique(np.concatenate(train_ptsidxlist), return_counts=True) # for 2D points (extr.xys, extr.point3D_ids)
    
    #valid_ptsidx = ptsidx[cnt>=min(2, len(train_index))][1:] # 2view -> 3view: more restrict condition (COLMAP uses 3-view observed feature points)
    valid_ptsidx = ptsidx[cnt>=min(2, 20)][1:]
    for tidx, key in enumerate(sorted(cam_extrinsics, key=lambda x:cam_extrinsics[x].name)):
        cam_valid = np.isin(cam_extrinsics[key].point3D_ids, valid_ptsidx)
        cam_extrinsics[key] = cam_extrinsics[key]._replace(point3D_ids = cam_extrinsics[key].point3D_ids[cam_valid],
                                                                 xys =  cam_extrinsics[key].xys[cam_valid])

    ### valid 3D points (removing the points only detected in 1 camera)
    ptsidx, cnt = np.unique(np.concatenate(total_ptsidxlist), return_counts=True) # for 3D points (xyz, rgb, err from points3D.bin)
    #valid_totalptsidx = ptsidx[cnt>=min(2, len(train_index))][1:] # remove invalid(-1) pts
    valid_totalptsidx = ptsidx[cnt>=min(2, 20)][1:] # remove invalid(-1) pts
    assert len(valid_totalptsidx)==len(xyz)
    
    valid3didx = np.isin(valid_totalptsidx, valid_ptsidx) # select the points seen from train-view
    xyz = xyz[valid3didx]
    rgb = rgb[valid3didx]
    err = err[valid3didx]
    
    ### save in ply format
    os.makedirs(os.path.join(path, 'plydummy'), exist_ok=True)
    ply_path = os.path.join(path, 'plydummy', f"points3D_{int(datetime.now().timestamp())}.ply") # k-shot train
    storePly(ply_path, xyz, rgb, xyzerr=err)
    
    return ply_path, cam_extrinsics, cam_intrinsics

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T               if 'x' in vertices else None
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0 if 'red' in vertices else None
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T              if 'nx' in vertices else None
    errors = vertices['xyzerr']/(np.min(vertices['xyzerr'] + 1e-8))                      if 'xyzerr' in vertices else None
    return BasicPointCloud(points=positions, colors=colors, normals=normals, errors=errors)



def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, pcd=None, resolution=2, train_idx=None, white_background=False):
    cam_infos = []
    model_zoe = None
    data_list = []
    for idx, key in enumerate(sorted(cam_extrinsics,key=lambda x:cam_extrinsics[x].name)):
        sys.stdout.write('\r')
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[0]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]

        if white_background:
            im_data = np.array(image.convert("RGBA"))
            bg = np.array([1, 1, 1, 0]) if white_background else np.array([0, 0, 0, 0])
            norm_data = im_data / 255.0
            arr = norm_data[:,:,:4] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGBA")
        
        depthmap, depth_weight,depth_error = None, None,None
        depthloss = 1e8
        if pcd is not None:
            depthmap = np.zeros((height // resolution, width // resolution))
            depth_weight = np.zeros((height // resolution, width // resolution))
            depth_error = np.zeros((height // resolution, width // resolution))
            K = np.array([[focal_length_x, 0, width // resolution / 2],
                          [0, focal_length_y, height // resolution / 2],
                          [0, 0, 1]])
            cam_coord = np.matmul(K, np.matmul(R.transpose(), pcd.points.transpose()) + T.reshape(3, 1))
            valid_idx = np.where(np.logical_and.reduce((cam_coord[2] > 0,
                                                        cam_coord[0] / cam_coord[2] >= 0,
                                                        cam_coord[0] / cam_coord[2] <= width // resolution - 1,
                                                        cam_coord[1] / cam_coord[2] >= 0,
                                                        cam_coord[1] / cam_coord[2] <= height // resolution - 1)))[0]
            pts_depths = cam_coord[-1:, valid_idx]
            cam_coord = cam_coord[:2, valid_idx] / cam_coord[-1:, valid_idx]
            depthmap[np.round(cam_coord[1]).astype(np.int32).clip(0, height // resolution - 1),
                     np.round(cam_coord[0]).astype(np.int32).clip(0, width // resolution - 1)] = pts_depths
                     
            depth_weight[np.round(cam_coord[1]).astype(np.int32).clip(0, height // resolution - 1),
                         np.round(cam_coord[0]).astype(np.int32).clip(0, width // resolution - 1)] = 1 / pcd.errors[valid_idx] if pcd.errors is not None else 1
            #depth_weight = depth_weight / depth_weight.max()

            depth_error[np.round(cam_coord[1]).astype(np.int32).clip(0, height // resolution - 1),
                        np.round(cam_coord[0]).astype(np.int32).clip(0, width // resolution - 1)] = pcd.errors[valid_idx] if pcd.errors is not None else 0

            # 初始化用于存储有效信息的列表
            valid_depths = []
            valid_weights = []
            valid_errors = []
            valid_coords = []

            # 遍历所有有效的索引
            
            for i in range(len(valid_idx)):
                # 计算出整数化后的像素坐标
                x = np.round(cam_coord[0, i]).astype(np.int32).clip(0, width // resolution - 1)
                y = np.round(cam_coord[1, i]).astype(np.int32).clip(0, height // resolution - 1)
                
                # 获取有效的深度、权重和误差信息
                #valid_depths.append(pts_depths[0, i])
                valid_depths.append(depthmap[y,x])
                if depthmap[y,x]==0:
                    print("ERROR!")

                if pcd.errors is not None:
                    valid_weights.append(1 / pcd.errors[valid_idx[i]])
                    valid_errors.append(pcd.errors[valid_idx[i]])
                else:
                    valid_weights.append(1)
                    valid_errors.append(0)
                
                # 存储对应的像素坐标
                valid_coords.append((x, y))

            # 将有效的数据存储到 data_list 中
            data_list.append({'id': key, 
                            'depth': np.array(valid_depths), 
                            'weight': np.array(valid_weights), 
                            'error': np.array(valid_errors), 
                            'coord': np.array(valid_coords)})

            target = depthmap.copy()
            #target = ((target != 0) * 255).astype(np.uint8)
            # output max min depth and total nonzero depth points
            print(f"Max depth: {target.max()}, Min depth: {target.min()}, Nonzero: {np.count_nonzero(target)},Mean depth:{np.sum(target)/np.count_nonzero(target)}")
            
            # get non_zero_indices  
            non_zero_indices = np.nonzero(target)
            # Enlarging the points for better visualization
            plt.figure(figsize=(12, 8))
            plt.imshow(depthmap, cmap='viridis', interpolation='nearest')
            plt.colorbar(label='Depth Value')
            plt.title('Depth Map Visualization')
            plt.xlabel('Width')
            plt.ylabel('Height')
            #print(f"Non-zero indices: {non_zero_indices}")
            # Highlight non-zero points more clearly
            for index in non_zero_indices:
                y, x = divmod(index, width // resolution-1)
                plt.plot(x, y, 'ro', markersize=10)
            
            # Save the enhanced visualization
            output_path_enhanced = '/mnt/data/depth_map_visualization_enhanced.png'
            plt.savefig(output_path_enhanced)
            plt.show()

            from matplotlib.colors import LinearSegmentedColormap
            cmap = LinearSegmentedColormap.from_list('black_white', [(1, 1, 1), (0, 0, 0)], N=256)

            # 可视化深度图
            plt.imshow(depthmap, cmap=cmap)
            plt.colorbar()
            plt.title(f"Depth Map {idx:03d}")
            
            plt.savefig(f"debug/{idx:03d}_depth.png")
            plt.close()
            
            # 保存深度图和权重
            np.save(f"debug/{idx:03d}_depthmap.npy", depthmap)
            print(f"shape:{depthmap.shape}")
            np.save(f"debug/{idx:03d}_depth_weight.npy", depth_weight)

    sys.stdout.write('\n')
    save_path = os.path.join("debug", "depth_data.npy")    
    np.save(save_path, data_list)
    print(f"Depth data saved to {save_path}")
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    args = parser.parse_args()
    reading_dir = "images" 
    ply_path, cam_extrinsics, cam_intrinsics = refineColmapWithIndex(args.path)
    pcd = fetchPly(ply_path)
    cam_infos = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, 
                                    images_folder=os.path.join(args.path, reading_dir), pcd=pcd)
