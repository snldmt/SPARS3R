import random

import numpy as np
from numpy.linalg import svd

from plyfile import PlyData, PlyElement
import open3d as o3d

import torch 
import roma


def pairwise_iou(masks):
    N = masks.size(0)
    masks_flat = masks.view(N, -1)
    intersection = torch.matmul(masks_flat.float(), masks_flat.T.float())
    area = masks_flat.sum(dim=1).view(-1, 1)
    union = area + area.T - intersection
    iou_matrix = intersection / union
    return iou_matrix

def get_sfm_points_in_mask(mask, sfm_points):
    """ Returns the SfM points that fall inside the binary mask. """
    # return [point for point in sfm_points if mask[int(round_python3(point[1])), int(round_python3(point[0]))]]
    
    # m = [mask[int(round_python3(y)), int(round_python3(x))].bool().detach().cpu().item() if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] else False for (x, y) in sfm_points]
    m = [mask[int(round_python3(y)), int(round_python3(x))].item() if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] else False for (x, y) in sfm_points]
    
    points = sfm_points[m]
    return points, m

def ransac_procrustes(P1, P2, 
                      num_samples=20, max_iterations=3000, threshold=0.1, inlier_ratio=0.99):
    P1 = P1.numpy()
    P2 = P2.numpy()
    
    best_inliers = []
    best_transformation = None
    best_res = None
    best_iter = -1

    N = P1.shape[0]
    required_inliers = int(N * inlier_ratio)

    for i in range(max_iterations):
        # Randomly select a subset of corresponding points
        indices = random.sample(range(N), num_samples)
        P1_sample = P1[indices]
        P2_sample = P2[indices]

        (_, R, s, T) = procrustes(P2_sample, P1_sample)
        
        # Apply the transformation to all points
        P1_transformed = transform_points(P1, R,s,T)

        # Calculate residuals (errors)
        residuals = np.linalg.norm(P1_transformed - P2, axis=1)

        # Determine inliers
        inliers = np.where(residuals < threshold)[0]

        # Update the best transformation if more inliers are found
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_transformation = (s, R, T)
            best_res = residuals
            best_iter = i

        # Early stopping if the desired inlier ratio is achieved
        if len(best_inliers) > required_inliers:
            print(f'early stop because of enough inliers, {len(best_inliers)}/{required_inliers}')
            break

    return best_transformation, best_inliers, best_res, best_iter

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    normals = np.zeros_like(xyz)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))
    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def save_ply(path, xyz, rgb):
    storePly(path=path, xyz=xyz, rgb=rgb)
    pcd = o3d.io.read_point_cloud(str(path))
    pcd.estimate_normals()
    tpcd = o3d.t.geometry.PointCloud.from_legacy(pcd)
    tpcd.point.colors = (tpcd.point.colors * 255).to(o3d.core.Dtype.UInt8)  # type: ignore
    o3d.t.io.write_point_cloud(str(path), tpcd)

def orthogonal_procrustes(A:np.ndarray, B:np.ndarray):
    assert A.ndim == 2
    assert B.ndim == 2
    #if A.shape != B.shape:
    #    raise ValueError('the shapes of A and B differ ({} vs {})'.format(
    #        A.shape, B.shape))
    assert A.shape[0] == B.shape[0]
    assert A.shape[1] == B.shape[1]

    # Be clever with transposes, with the intention to save memory.
    u, w, vt = svd(B.T.dot(A).T)
    R = u.dot(vt)
    scale = w.sum()
    return R, scale

def np_mean_along_axis(arr: np.ndarray, axis: int):
    assert arr.ndim == 2  # only supports 2D arrays
    assert axis in (0, 1)
    if axis == 0:
        result = np.empty(arr.shape[1], dtype=arr.dtype)
        for i in range(len(result)):
            result[i] = np.mean(arr[:, i])
    else:
        result = np.empty(arr.shape[0], dtype=arr.dtype)
        for i in range(len(result)):
            result[i] = np.mean(arr[i, :])
    return result
    
def procrustes(data1, data2):    
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

def transform_points(points:np.ndarray, rotation:np.ndarray, scale, translation:np.ndarray):
    assert len(points.shape) == 2
    assert points.shape[1] == 3
    assert len(rotation.shape) == 2
    assert rotation.shape[0] == 3
    assert rotation.shape[1] == 3
    # dot == matmul
    return np.dot(points, np.linalg.inv(rotation).transpose()) * scale + translation.reshape((1, 3))

def round_python3(number):
    rounded = round(number)
    if abs(number - rounded) == 0.5:
        return 2.0 * round(number / 2.0)
    return rounded

def rigid_points_registration(pts1, pts2, conf):
    w = conf.ravel() if conf is not None else None
    R, T, s = roma.rigid_points_registration(
        pts1.reshape(-1, 3), pts2.reshape(-1, 3), 
        # weights=conf.ravel(), 
        weights=w, 
        compute_scaling=True)
    return s, R, T  # return un-scaled (R, T)

def sRT_to_4x4(scale, R, T, device):
    trf = torch.eye(4, device=device)
    trf[:3, :3] = R * scale
    trf[:3, 3] = T.ravel()  # doesn't need scaling
    return trf

def geotrf(Trf, pts, ncol=None, norm=False):
    """ Apply a geometric transformation to a list of 3-D points.

    H: 3x3 or 4x4 projection matrix (typically a Homography)
    p: numpy/torch/tuple of coordinates. Shape must be (...,2) or (...,3)

    ncol: int. number of columns of the result (2 or 3)
    norm: float. if != 0, the resut is projected on the z=norm plane.

    Returns an array of projected 2d points.
    """
    assert Trf.ndim >= 2
    # if isinstance(Trf, np.ndarray):
    #     pts = np.asarray(pts)
    # elif isinstance(Trf, torch.Tensor):
    #     pts = torch.as_tensor(pts, dtype=Trf.dtype)

    # adapt shape if necessary
    output_reshape = pts.shape[:-1]
    ncol = ncol or pts.shape[-1]

    # optimized code
    if (isinstance(Trf, torch.Tensor) and isinstance(pts, torch.Tensor) and
            Trf.ndim == 3 and pts.ndim == 4):
        d = pts.shape[3]
        if Trf.shape[-1] == d:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf, pts)
        elif Trf.shape[-1] == d + 1:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf[:, :d, :d], pts) + Trf[:, None, None, :d, d]
        else:
            raise ValueError(f'bad shape, not ending with 3 or 4, for {pts.shape=}')
    else:
        if Trf.ndim >= 3:
            n = Trf.ndim - 2
            assert Trf.shape[:n] == pts.shape[:n], 'batch size does not match'
            Trf = Trf.reshape(-1, Trf.shape[-2], Trf.shape[-1])

            if pts.ndim > Trf.ndim:
                # Trf == (B,d,d) & pts == (B,H,W,d) --> (B, H*W, d)
                pts = pts.reshape(Trf.shape[0], -1, pts.shape[-1])
            elif pts.ndim == 2:
                # Trf == (B,d,d) & pts == (B,d) --> (B, 1, d)
                pts = pts[:, None, :]

        if pts.shape[-1] + 1 == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf[..., :-1, :] + Trf[..., -1:, :]
        elif pts.shape[-1] == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf
        else:
            pts = Trf @ pts.T
            if pts.ndim >= 2:
                pts = pts.swapaxes(-1, -2)

    if norm:
        pts = pts / pts[..., -1:]  # DONT DO /= BECAUSE OF WEIRD PYTORCH BUG
        if norm != 1:
            pts *= norm

    res = pts[..., :ncol].reshape(*output_reshape, ncol)
    # res = pts[..., :ncol].view(*output_reshape, ncol)

    return res

