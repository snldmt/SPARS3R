# taken from wrivalib.metadata.utils
"""
Copyright © 2022-2023 The Johns Hopkins University Applied Physics Laboratory LLC

Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the “Software”), to 
deal in the Software without restriction, including without limitation the 
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or 
sell copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in 
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR 
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import numpy as np
from scipy_procrustes import orthogonal_procrustes
from scipy.spatial.transform import Rotation
from colmap_read_write_model import Image, Point3D, qvec2rotmat
from numba import njit
from numba_reduce import np_mean_along_axis


def colmap_image_to_extrinsics_c2w(image):
    bottom = np.float64([0.0, 0.0, 0.0, 1.0]).reshape((1, 4))
    R = qvec2rotmat(-image.qvec)
    t = image.tvec.reshape((3, 1))
    m = np.concatenate((np.concatenate((R, t), 1), bottom), 0)
    return np.linalg.inv(m)

@njit(cache=True)
def procrustes(data1, data2):
    # Adapted from scipy.spatial.procrustes (https://github.com/scipy/scipy/blob/main/scipy/spatial/_procrustes.py)
    # Copyright (c) 2001-2002 Enthought, Inc. 2003-2023, SciPy Developers.
    # All rights reserved.
    #
    # Redistribution and use in source and binary forms, with or without
    # modification, are permitted provided that the following conditions
    # are met:
    #
    # 1. Redistributions of source code must retain the above copyright
    #    notice, this list of conditions and the following disclaimer.
    #
    # 2. Redistributions in binary form must reproduce the above
    #    copyright notice, this list of conditions and the following
    #    disclaimer in the documentation and/or other materials provided
    #    with the distribution.
    #
    # 3. Neither the name of the copyright holder nor the names of its
    #    contributors may be used to endorse or promote products derived
    #    from this software without specific prior written permission.
    #
    # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    # "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    # LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
    # A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    # OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    # SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    # LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    # DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    # THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """
    Procrustes analysis, a similarity test for two data sets.

    :param data1: Matrix, n rows represent points in k (columns) space `data1` is the
        reference data, after it is standardised, the data from `data2` will be
        transformed to fit the pattern in `data1` (must have >1 unique points).
    :param data2: n rows of data in k space to be fit to `data1`.  Must be the  same
        shape ``(numrows, numcols)`` as data1 (must have >1 unique points).
    :return: float representing disparity; a dict specifying the rotation, scale and translation for the transformation
    """
    mtx1 = data1.astype(np.double).copy()
    mtx2 = data2.astype(np.double).copy()

    if mtx1.ndim != 2 or mtx2.ndim != 2:
        raise ValueError("Input matrices must be two-dimensional")
    if mtx1.shape != mtx2.shape:
        raise ValueError("Input matrices must be of same shape")
    if mtx1.size == 0:
        raise ValueError("Input matrices must be >0 rows and >0 cols")

    # translate all the data to the origin
    mtx1 -= np_mean_along_axis(mtx1, 0)
    mtx2 -= np_mean_along_axis(mtx2, 0)

    norm1 = np.linalg.norm(mtx1)
    norm2 = np.linalg.norm(mtx2)

    if norm1 == 0 or norm2 == 0:
        raise ValueError("Input matrices must contain >1 unique points")

    # change scaling of data (in rows) such that trace(mtx*mtx') = 1
    mtx1 /= norm1
    mtx2 /= norm2

    # transform mtx2 to minimize disparity
    R, s = orthogonal_procrustes(mtx1, mtx2)
    mtx2 = np.dot(mtx2, R.T) * s

    # measure the dissimilarity between the two datasets
    disparity = np.sum(np.square(mtx1 - mtx2))

    rotation = R.T
    scale = s * norm1 / norm2
    translation = np_mean_along_axis(data1, 0) - (
        np_mean_along_axis(data2, 0).dot(rotation) * scale
    )

    return disparity, rotation, scale, translation


def transform_colmap_model(model, tform):
    """
    Transforms a COLMAP model

    :param model: COLMAP dictionary objects (cameras, images, points3D) of model
    :param tform: a dict specifying the rotation, scale and translation for the transformation
    :return: COLMAP dictionary objects (cameras, images, points3D) of transformed model
    """
    # extract objects
    cameras, images, points3D = model

    # transform images
    for key, value in images.items():
        r = Rotation.from_quat(np.roll(value.qvec, -1)) * Rotation.from_matrix(tform["rotation"])
        t = tform["scale"] * value.tvec - r.as_matrix() @ tform["translation"]

        images[key] = Image(
            id=value.id,
            qvec=np.roll(r.as_quat(), 1),
            tvec=t,
            camera_id=value.camera_id,
            name=value.name,
            xys=value.xys,
            point3D_ids=value.point3D_ids,
        )

    # transform points3D
    for key, value in points3D.items():
        points3D[key] = Point3D(
            id=value.id,
            xyz=tform["scale"] * value.xyz @ tform["rotation"] + tform["translation"],
            rgb=value.rgb,
            error=value.error,
            image_ids=value.image_ids,
            point2D_idxs=value.point2D_idxs,
        )

    return cameras, images, points3D

from utils.utils_poses.lie_group_helper import SO3_to_quat, convert3x4_4x4
import torch



def store_transformed(model, s,R,t, mtx1_mean,norm1, mtx2_mean,norm2, s1):
    """
    Transforms a COLMAP model

    :param model: COLMAP dictionary objects (cameras, images, points3D) of model
    :param tform: a dict specifying the rotation, scale and translation for the transformation
    :return: COLMAP dictionary objects (cameras, images, points3D) of transformed model
    """
    # extract objects
    cameras, images, points3D = model
    

    R = R[None, :, :].astype(np.float32)  # (1, 3, 3)
    T = t[None, :, None].astype(np.float32)  # (1, 3, 1)
    s = float(s)
    # transform images
    all_angles_c2w = []
    for key, value in images.items():

        # R_w2c = Rotation.from_quat(np.roll(value.qvec, -1))
        # T_w2c = value.tvec
        # camera_w2c = np.concatenate([R_w2c.as_matrix(), T_w2c[:,np.newaxis]], axis=1)  # (N1, 3, 4)
        # camera_w2c = camera_w2c[np.newaxis,...]
        # camera_w2c = convert3x4_4x4(camera_w2c)  # (N1, 4, 4)
        # camera_c2w = np.array([np.linalg.inv(m) for m in camera_w2c])
        all_angles_c2w.append(colmap_image_to_extrinsics_c2w(value))
        print('1')
    all_angles_c2w = np.array(all_angles_c2w)
    all_angles_c2w[:,:3,-1] = ((all_angles_c2w[:,:3,-1]-mtx2_mean)/norm2)*s1

    for camera_c2w in all_angles_c2w:
        camera_c2w = camera_c2w[np.newaxis,...]
        R_c2w = camera_c2w[:, :3, :3]  # (N1, 3, 3)
        T_c2w = camera_c2w[:, :3, 3:4]  # (N1, 3, 1)
        R_c2w_aligned = R @ R_c2w  # (N1, 3, 3)
        t_c2w_aligned = s * (R @ T_c2w) + T  # (N1, 3, 1)
        t_c2w_aligned[:,:3,-1] = t_c2w_aligned[:,:3,-1]* norm1 + mtx1_mean

        traj_c2w_aligned = np.concatenate([R_c2w_aligned, t_c2w_aligned], axis=2)  # (N1, 3, 4)

            # append the last row
        traj_c2w_aligned = convert3x4_4x4(traj_c2w_aligned)  # (N1, 4, 4)
        traj_w2c_aligned= np.array([np.linalg.inv(m) for m in traj_c2w_aligned])
        R_w2c_aligned = traj_w2c_aligned[:, :3, :3]
        T_w2c_aligned = traj_w2c_aligned[:, :3, 3:4]

        R_w2c_rotation = Rotation.from_matrix(R_w2c_aligned[0])
        t = T_w2c_aligned[0,:3,0]

    for key, value in images.items():
        images[key] = Image(
            id=value.id,
            qvec=np.roll(R_w2c_rotation.as_quat(), 1),
            tvec=t,
            camera_id=value.camera_id,
            name=value.name,
            xys=value.xys,
            point3D_ids=value.point3D_ids,
        )

    # transform points3D
    for key, value in points3D.items():
        points3D[key] = Point3D(
            id=value.id,
            xyz=s * value.xyz @ R + T,
            rgb=value.rgb,
            error=value.error,
            image_ids=value.image_ids,
            point2D_idxs=value.point2D_idxs,
        )

    return cameras, images, points3D


def store_transformed_roma(model, s,R,T):
    """
    Transforms a COLMAP model

    :param model: COLMAP dictionary objects (cameras, images, points3D) of model
    :param tform: a dict specifying the rotation, scale and translation for the transformation
    :return: COLMAP dictionary objects (cameras, images, points3D) of transformed model
    """
    # extract objects
    cameras, images, points3D = model
    

    transform_matrix = torch.eye(4)
    transform_matrix[:3, :3] = R
    transform_matrix[:3, 3] = T
    transform_matrix[:3, :3] *= s
    transform_matrix = transform_matrix.numpy()
    # transform images
    all_angles_c2w = []
    for key, value in images.items():

        all_angles_c2w=colmap_image_to_extrinsics_c2w(value)
        all_angles_c2w = np.array(all_angles_c2w)[np.newaxis,...]
        aligned_all_angles_c2w =  transform_matrix @ all_angles_c2w
        aligned_all_angles_c2w[:,:3,:3] = aligned_all_angles_c2w [:,:3,:3] / s

        traj_w2c_aligned= np.array([np.linalg.inv(m) for m in aligned_all_angles_c2w])
        R_w2c_aligned = traj_w2c_aligned[:, :3, :3]
        T_w2c_aligned = traj_w2c_aligned[:, :3, 3:4]

        R_w2c_rotation = Rotation.from_matrix(R_w2c_aligned[0])
        t = T_w2c_aligned[0,:3,0]


        images[key] = Image(
            id=value.id,
            qvec=np.roll(R_w2c_rotation.as_quat(), 1),
            tvec=t,
            camera_id=value.camera_id,
            name=value.name,
            xys=value.xys,
            point3D_ids=value.point3D_ids,
        )

    # transform points3D
    for key, value in points3D.items():
        points3D[key] = Point3D(
            id=value.id,
            xyz=s * value.xyz @ R + T,
            rgb=value.rgb,
            error=value.error,
            image_ids=value.image_ids,
            point2D_idxs=value.point2D_idxs,
        )

    return cameras, images, points3D
