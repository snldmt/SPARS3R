# Author: Jason Bunk @ Mayachitra
import os
import numpy as np
from colmap_read_write_model import qvec2rotmat, Image, Camera
from utils.transform_utils import procrustes, transform_colmap_model, store_transformed, store_transformed_roma
from numba import njit
from numba_reduce import np_std_along_axis
from utils.utils_poses.align_traj import align_ate_c2b_use_a2b
import roma

# colmap stores w2c, but c2w extrinsic matrix contains world camera position
# @njit(cache=True)
def colmap_image_to_extrinsics_c2w(image):
    bottom = np.float64([0.0, 0.0, 0.0, 1.0]).reshape((1, 4))
    R = qvec2rotmat(image.qvec)
    t = image.tvec.reshape((3, 1))
    m = np.concatenate((np.concatenate((R, t), 1), bottom), 0)
    return np.linalg.inv(m)

def rigid_points_registration(pts1, pts2, conf=None):
    R, T, s = roma.rigid_points_registration(
        pts1.reshape(-1, 3), pts2.reshape(-1, 3), weights=conf, compute_scaling=True)
    return s, R, T

@njit(cache=True)
def calculate_rotation_error(q_true, q_guess):

    acosme = np.abs(np.dot(q_true, q_guess)) / max(
        1e-15, np.linalg.norm(q_true) * np.linalg.norm(q_guess)
    )
    assert acosme > -0.0001 and acosme < 1.0001, f"{acosme} -- {q_true}, {q_guess}"

    return np.rad2deg(2.0 * np.arccos(min(1., acosme)))

# transform a point cloud which exists in world coordinate space
@njit(cache=True)
def transform_points(points:np.ndarray, rotation:np.ndarray, scale, translation:np.ndarray):
    assert len(points.shape) == 2
    assert points.shape[1] == 3
    assert len(rotation.shape) == 2
    assert rotation.shape[0] == 3
    assert rotation.shape[1] == 3
    # dot == matmul
    return np.dot(points, np.linalg.inv(rotation).transpose()) * scale + translation.reshape((1, 3))


# produce 4 points: one is the camera position, and 3 points are offset from the camera using the rotation direction vectors
@njit(cache=True)
def extrinsic_c2w_to_axis_cloud(c2w: np.ndarray, scale: float):
    assert len(c2w.shape) == 2  # , f"{c2w.shape} -- expected (4,4) or (3,4)"
    assert int(c2w.shape[0]) in (3, 4)
    assert int(c2w.shape[1]) == 4
    ret = np.copy(c2w) # should be c2w so that first 3 columns are coordinate system directions
    ret[:3,:3] = c2w[:3,:3] * scale + c2w[:3,3:4] # shift coordinate directions to center around camera position
    return ret[:3].transpose() # return row vectors

# A rotation-invariant radius-like measure of a point cloud
@njit(cache=True)
def cloud_radius(cloud: np.ndarray):
    cst = cloud.astype(np.float64)
    cst = np_std_along_axis(cst, axis=0)
    return np.sqrt(np.square(cst).sum())

def built_initial_opt_state():
    return (
        np.inf, # least_dist
        -9, # most_num_inliers
        np.zeros((3, 3), dtype=np.float64), # best_rotation
        np.nan, # best_scale
        np.empty((3,), dtype=np.float64), # best_translation
    )

def convert_opt_tuple_to_dict(opt):
    assert len(opt) == 5, f"{len(opt)}\n{opt}"
    return {
        "dist_cost": opt[0],
        "num_inliers": opt[1],
        "rotation": opt[2],
        "scale": opt[3],
        "translation": opt[4],
    }

# This only needs 2 camera poses, so can be robust in 5-image datasets
# It minimizes both positional and rotational error between any pair of points
@njit(cache=True)
def compute_robust_alignment_using_pairs(
    c2w_fixed: np.ndarray,
    c2w_movng: np.ndarray,
    ransac_thresholds: np.ndarray,
    least_dist: np.float64,
    most_num_inliers: int,
    best_rotation: np.ndarray,
    best_scale: np.float64,
    best_translation: np.array,
):
    # inputs must be (N,4,4) or (N,3,4) extrinsic "c2w" camera-to-world matrices
    assert len(c2w_fixed) == len(c2w_movng)
    assert len(c2w_fixed) == len(ransac_thresholds)
    squared_dist_target = np.square(ransac_thresholds)

    # extract xyz position from each c2w matrix
    cloud_movng = np.ascontiguousarray(c2w_movng[:, :3, 3])
    cloud_fixed = np.ascontiguousarray(c2w_fixed[:, :3, 3])

    assert len(cloud_movng.shape) == 2  # , str(cloud_movng.shape)

    radius_movng = cloud_radius(cloud_movng)
    radius_fixed = cloud_radius(cloud_fixed)

    # camsfixed = np.empty((2,3), dtype=np.float64)
    # camsmovng = np.empty((2,3), dtype=np.float64)

    # for all pairs of camera points
    for lidx in range(len(cloud_movng) - 1):
        for ridx in range(lidx + 1, len(cloud_movng)):
            # with 2 camera points we get a sense of scale as the distance between the points
            scale_fixed = np.linalg.norm(c2w_fixed[lidx,:3,3] - c2w_fixed[ridx,:3,3])
            scale_movng = np.linalg.norm(c2w_movng[lidx,:3,3] - c2w_movng[ridx,:3,3])
            # if the 2 camera points had the same position, use point cloud radius as an alternative sense of scale
            # it's still useful to compute the transformation since we compute a rotation
            if scale_fixed <= 1e-9 or scale_movng <= 1e-9:
                scale_fixed = radius_fixed
                scale_movng = radius_movng
            # create new little point clouds which have 8 points (8 fixed and 8 moving), and register those
            # each of the 2 camera positions becomes 4 points: the camera + 3 coordinate axis vector offsets
            lc_fixed = np.concatenate((extrinsic_c2w_to_axis_cloud(c2w_fixed[lidx], scale_fixed), extrinsic_c2w_to_axis_cloud(c2w_fixed[ridx], scale_fixed)))
            lc_movng = np.concatenate((extrinsic_c2w_to_axis_cloud(c2w_movng[lidx], scale_movng), extrinsic_c2w_to_axis_cloud(c2w_movng[ridx], scale_movng)))

            (_, rotation, scale, translation) = procrustes(lc_fixed, lc_movng)
            # gather camera centers
            # for offs,zidx in enumerate((lidx, ridx)):
            #     camsfixed[offs] = c2w_fixed[zidx,:3,3]
            #     camsmovng[offs] = c2w_movng[zidx,:3,3]
            # (_, rotation, scale, translation) = procrustes(camsfixed, camsmovng)

            tcl_movng = transform_points(cloud_movng, rotation, scale, translation)
            sqdists = np.square(tcl_movng - cloud_fixed).sum(axis=1)

            # distances relative to threshold
            sqdists = np.divide(sqdists, squared_dist_target)

            # maximize number of "inliers" = number of guessed cameras that are less than "distance_target" meters from desired position
            inliermask = np.less_equal(sqdists, 1.0)
            numinliers = np.count_nonzero(inliermask)

            if numinliers >= most_num_inliers:
                if numinliers == 0:
                    inlierdists = sqdists.mean()
                else:
                    inlierdists = sqdists[inliermask].mean()

                if numinliers > most_num_inliers or (numinliers == most_num_inliers and inlierdists < least_dist):
                    least_dist = inlierdists
                    most_num_inliers = numinliers
                    best_rotation = rotation
                    best_scale = scale
                    best_translation = translation

    return least_dist, most_num_inliers, best_rotation, best_scale, best_translation


# This can be better at minimizing positional error,
# but can be worse at minimizing rotational error because it ignores the camera rotations!
@njit(cache=True)
def compute_robust_alignment_position_triplets(
    c2w_fixed: np.ndarray,
    c2w_movng: np.ndarray,
    ransac_thresholds: np.ndarray,
    least_dist: np.float64,
    most_num_inliers: int,
    best_rotation: np.ndarray,
    best_scale: np.float64,
    best_translation: np.array,
):
    # inputs must be (N,4,4) or (N,3,4) extrinsic "c2w" camera-to-world matrices
    assert len(c2w_fixed) == len(c2w_movng)
    assert len(c2w_fixed) == len(ransac_thresholds)
    squared_dist_target = np.square(ransac_thresholds)

    # extract xyz position from each c2w matrix
    cloud_movng = np.ascontiguousarray(c2w_movng[:, :3, 3])
    cloud_fixed = np.ascontiguousarray(c2w_fixed[:, :3, 3])

    assert len(cloud_movng.shape) == 2  # , str(cloud_movng.shape)

    camsfixed = np.empty((3,3), dtype=np.float64)
    camsmovng = np.empty((3,3), dtype=np.float64)

    # radius_movng = cloud_radius(cloud_movng)
    # radius_fixed = cloud_radius(cloud_fixed)

    # for all triplets of camera points
    for lidx in range(len(cloud_movng) - 1):
        for ridx in range(lidx + 1, len(cloud_movng)):
            for tidx in range(len(cloud_movng)):
                if tidx == lidx or tidx == ridx:
                    continue

                # gather camera centers
                for offs,zidx in enumerate((lidx, ridx, tidx)):
                    camsfixed[offs] = c2w_fixed[zidx,:3,3]
                    camsmovng[offs] = c2w_movng[zidx,:3,3]

                # lc_fixed = np.concatenate((extrinsic_c2w_to_axis_cloud(c2w_fixed[lidx], radius_fixed), extrinsic_c2w_to_axis_cloud(c2w_fixed[ridx], radius_fixed), extrinsic_c2w_to_axis_cloud(c2w_fixed[tidx], radius_fixed)))
                # lc_movng = np.concatenate((extrinsic_c2w_to_axis_cloud(c2w_movng[lidx], radius_movng), extrinsic_c2w_to_axis_cloud(c2w_movng[ridx], radius_movng), extrinsic_c2w_to_axis_cloud(c2w_movng[tidx], radius_movng)))

                # (_, rotation, scale, translation) = procrustes(lc_fixed, lc_movng)
                # compute 7DOF rigid alignment between the 3 fixed points and the 3 "moving" points
                (_, rotation, scale, translation) = procrustes(camsfixed, camsmovng)

                tcl_movng = transform_points(cloud_movng, rotation, scale, translation)
                sqdists = np.square(tcl_movng - cloud_fixed).sum(axis=1)

                # distances relative to threshold
                sqdists = np.divide(sqdists, squared_dist_target)

                # maximize number of "inliers" = number of guessed cameras that are less than "distance_target" meters from desired position
                inliermask = np.less_equal(sqdists, 1.0)
                numinliers = np.count_nonzero(inliermask)

                if numinliers >= most_num_inliers:
                    if numinliers == 0:
                        inlierdists = sqdists.mean()
                    else:
                        inlierdists = sqdists[inliermask].mean()

                    if numinliers > most_num_inliers or (numinliers == most_num_inliers and inlierdists < least_dist):
                        least_dist = inlierdists
                        most_num_inliers = numinliers
                        best_rotation = rotation
                        best_scale = scale
                        best_translation = translation

    return least_dist, most_num_inliers, best_rotation, best_scale, best_translation


def robust_register_colmap_models_dm(
    model_moving,
    model_fixed,
    ransac_distance_target: float,
    true_key_to_ransac_threshold: dict = {},
):
    """
    Registers the moving COLMAP model to the fixed COLMAP model.
    This method is more robust to outliers than "register_colmap_models".

    :param model_moving: COLMAP dictionary objects (cameras, images, points3D) of moving model
    :param model_fixed: COLMAP dictionary objects (cameras, images, points3D) of fixed model
    :return: COLMAP dictionary objects (cameras, images, points3D) of registered model
    """
    # extract objects
    cameras_movng, images_movng, points3D_movng = model_moving
    cameras_fixed, images_fixed, points3D_fixed = model_fixed
    print('images_movng',len(images_movng.keys()))
    print('images_fixed',len(images_fixed.keys()))
    # find matching images
    name2key_movng = {image.name: key for key, image in images_movng.items()}
    name2key_fixed = {image.name: key for key, image in images_fixed.items()}
    print(set(name2key_movng.keys()), set(name2key_fixed.keys()))
    sorted_names = sorted(list(set(name2key_movng.keys()) & set(name2key_fixed.keys())))

    assert len(sorted_names) > 0, "no intersection between colmap model keys: are you sure they correspond to the same dataset?"
    print(sorted_names,'#####')

    sorted_keys_movng = [name2key_movng[name] for name in sorted_names]
    sorted_keys_fixed = [name2key_fixed[name] for name in sorted_names]
    print(sorted_keys_movng,'$$$$$$$')
    print(sorted_keys_fixed,'$$$$$##')
    ransac_thresholds = np.float64([ 1.0 for tk in sorted_keys_fixed ])
    print(ransac_thresholds,'#######@@#')
    assert ransac_thresholds.min() > 1e-9, f"min {ransac_thresholds.min()}, max {ransac_thresholds.max()}, median {np.median(ransac_thresholds)}"

    c2w_movng = np.stack([colmap_image_to_extrinsics_c2w(images_movng[key]) for key in sorted_keys_movng])


    c2w_fixed = np.stack([colmap_image_to_extrinsics_c2w(images_fixed[key]) for key in sorted_keys_fixed])
    # c2w_fixed = np.load('/cis/home/yguo/3dgs/InstantSplat/output/eval/mips_uniform/stump/12_views/pose/pose_1000.npy')

    assert len(c2w_movng) == len(sorted_names)

    opt = built_initial_opt_state()
    if len(c2w_movng) > 2:
        opt = compute_robust_alignment_position_triplets(c2w_fixed, c2w_movng, ransac_thresholds, *opt)
    opt = compute_robust_alignment_using_pairs(c2w_fixed, c2w_movng, ransac_thresholds, *opt)

    best_transform = convert_opt_tuple_to_dict(opt)
    # model_registered = transform_colmap_model(model_moving, best_transform)
    cameras_registered, images_registered, points3D_registered = transform_colmap_model(model_moving, best_transform)
    
    # cameras_movng_save = {2: Camera(id=2, model=cameras_movng[1].model, width=cameras_movng[1].width, height=cameras_movng[1].height,
    #                         params=cameras_movng[1].params)}
    # cameras_movng_save = {}
    # cameras_fixed.update(cameras_movng_save)

    images_registered_save = {}
    camera_correspondence = {}
    fixed_count, moving_count = 0, 0
    for key, curr_image in images_registered.items():
        # training cameras in moving
        if key in sorted_keys_movng:
            idx = sorted_keys_movng.index(key)
            image_fixed = images_fixed[sorted_keys_fixed[idx]]

            camera_correspondence[curr_image.camera_id] = image_fixed.camera_id

            images_registered_save[key] = Image(
                id=curr_image.id, qvec=image_fixed.qvec, tvec=image_fixed.tvec,
                camera_id=curr_image.camera_id, name=image_fixed.name,
                xys=image_fixed.xys, point3D_ids=image_fixed.point3D_ids)
            
            moving_count += 1
        else:

            images_registered_save[key] = Image(
                id=curr_image.id, qvec=curr_image.qvec, tvec=curr_image.tvec,
                camera_id=curr_image.camera_id, name=curr_image.name,
                xys=np.zeros((1,2)), point3D_ids=(1)*np.ones((1,)).astype('uint8'))

            fixed_count += 1
    
    for k, v in camera_correspondence.items():
        cameras_movng[k] = Camera(id=k, model=cameras_fixed[v].model, width=cameras_fixed[v].width, height=cameras_fixed[v].height, params=cameras_fixed[v].params)

    print(moving_count)
    print(fixed_count)

    return cameras_movng, images_registered, images_registered_save, points3D_registered


from utils.utils_poses.ATE.align_utils import alignTrajectory
from utils.utils_poses.lie_group_helper import SO3_to_quat, convert3x4_4x4
import torch
import scipy

def align_pose(pose1, pose2):
    mtx1 = np.array(pose1, dtype=np.double, copy=True)
    mtx2 = np.array(pose2, dtype=np.double, copy=True)

    if mtx1.ndim != 2 or mtx2.ndim != 2:
        raise ValueError("Input matrices must be two-dimensional")
    if mtx1.shape != mtx2.shape:
        raise ValueError("Input matrices must be of same shape")
    if mtx1.size == 0:
        raise ValueError("Input matrices must be >0 rows and >0 cols")

    # translate all the data to the origin
    mtx1_mean = np.mean(mtx1, 0)
    mtx2_mean = np.mean(mtx2, 0)
    mtx1 -= np.mean(mtx1, 0)
    mtx2 -= np.mean(mtx2, 0)

    norm1 = np.linalg.norm(mtx1)
    norm2 = np.linalg.norm(mtx2)

    if norm1 == 0 or norm2 == 0:
        raise ValueError("Input matrices must contain >1 unique points")

    # change scaling of data (in rows) such that trace(mtx*mtx') = 1
    mtx1 /= norm1
    mtx2 /= norm2

    # transform mtx2 to minimize disparity
    R, s = scipy.linalg.orthogonal_procrustes(mtx1, mtx2)
    mtx2 = mtx2 * s

    return mtx1, mtx2, mtx1_mean,norm1, mtx2_mean,norm2, s


def align_ate_c2b_use_a2b(traj_a, traj_b, traj_c=None, method='sim3'):
    """Align c to b using the sim3 from a to b.
    :param traj_a:  (N0, 3/4, 4) torch tensor
    :param traj_b:  (N0, 3/4, 4) torch tensor
    :param traj_c:  None or (N1, 3/4, 4) torch tensor
    :return:        (N1, 4,   4) torch tensor
    """
    device = traj_a.device
    if traj_c is None:
        traj_c = traj_a.clone()

    traj_a = traj_a.float().cpu().numpy()
    traj_b = traj_b.float().cpu().numpy()
    traj_c = traj_c.float().cpu().numpy()

    R_a = traj_a[:, :3, :3]  # (N0, 3, 3)
    t_a = traj_a[:, :3, 3]  # (N0, 3)
    quat_a = SO3_to_quat(R_a)  # (N0, 4)

    R_b = traj_b[:, :3, :3]  # (N0, 3, 3)
    t_b = traj_b[:, :3, 3]  # (N0, 3)
    quat_b = SO3_to_quat(R_b)  # (N0, 4)

    # This function works in quaternion.
    # scalar, (3, 3), (3, ) gt = R * s * est + t.
    s, R, t = alignTrajectory(t_a, t_b, quat_a, quat_b, method=method)

    # # reshape tensors
    # R = R[None, :, :].astype(np.float32)  # (1, 3, 3)
    # t = t[None, :, None].astype(np.float32)  # (1, 3, 1)
    # s = float(s)

    # R_c = traj_c[:, :3, :3]  # (N1, 3, 3)
    # t_c = traj_c[:, :3, 3:4]  # (N1, 3, 1)

    # R_c_aligned = R @ R_c  # (N1, 3, 3)
    # t_c_aligned = s * (R @ t_c) + t  # (N1, 3, 1)
    # traj_c_aligned = np.concatenate([R_c_aligned, t_c_aligned], axis=2)  # (N1, 3, 4)

    # # append the last row
    # traj_c_aligned = convert3x4_4x4(traj_c_aligned)  # (N1, 4, 4)

    # traj_c_aligned = torch.from_numpy(traj_c_aligned).to(device)
    return s,R,t  # (N1, 4, 4)


def robust_register_colmap_models_ins(model_moving,
    model_fixed,
    ransac_distance_target: float,
    true_key_to_ransac_threshold: dict = {},
):
    """
    Registers the moving COLMAP model to the fixed COLMAP model.
    This method is more robust to outliers than "register_colmap_models".

    :param model_moving: COLMAP dictionary objects (cameras, images, points3D) of moving model
    :param model_fixed: COLMAP dictionary objects (cameras, images, points3D) of fixed model
    :return: COLMAP dictionary objects (cameras, images, points3D) of registered model
    """
    # extract objects
    cameras_movng, images_movng, points3D_movng = model_moving
    cameras_fixed, images_fixed, points3D_fixed = model_fixed
    print('images_movng',len(images_movng.keys()))
    print('images_fixed',len(images_fixed.keys()))
    # find matching images
    name2key_movng = {image.name: key for key, image in images_movng.items()}
    name2key_fixed = {image.name: key for key, image in images_fixed.items()}
    print(set(name2key_movng.keys()), set(name2key_fixed.keys()))
    sorted_names = sorted(list(set(name2key_movng.keys()) & set(name2key_fixed.keys())))

    assert len(sorted_names) > 0, "no intersection between colmap model keys: are you sure they correspond to the same dataset?"
    print(sorted_names,'#####')

    sorted_keys_movng = [name2key_movng[name] for name in sorted_names]
    sorted_keys_fixed = [name2key_fixed[name] for name in sorted_names]
    print(sorted_keys_movng,'$$$$$$$')
    print(sorted_keys_fixed,'$$$$$##')
    ransac_thresholds = np.float64([ 1.0 for tk in sorted_keys_fixed ])
    print(ransac_thresholds,'#######@@#')
    assert ransac_thresholds.min() > 1e-9, f"min {ransac_thresholds.min()}, max {ransac_thresholds.max()}, median {np.median(ransac_thresholds)}"

    c2w_movng = np.stack([colmap_image_to_extrinsics_c2w(images_movng[key]) for key in sorted_keys_movng])
    c2w_fixed = np.stack([colmap_image_to_extrinsics_c2w(images_fixed[key]) for key in sorted_keys_fixed])

    assert len(c2w_movng) == len(sorted_names)


    s, R, T = rigid_points_registration(torch.from_numpy(c2w_movng[:,:3,3]), torch.from_numpy(c2w_fixed[:,:3,3]))
    # trans_fix_align, trans_moving_align, mtx1_mean,norm1, mtx2_mean,norm2, s1 = align_pose(c2w_fixed[:, :3, -1], c2w_movng[:, :3, -1])
    # c2w_fixed[:, :3, -1] = trans_fix_align
    # c2w_movng[:, :3, -1] = trans_moving_align
    # s,R,T = align_ate_c2b_use_a2b(torch.tensor(c2w_fixed), torch.tensor(c2w_movng))



    # model_registered = transform_colmap_model(model_moving, best_transform)
    # cameras_registered, images_registered, points3D_registered = store_transformed(model_moving, s,R,T,mtx1_mean,norm1, mtx2_mean,norm2, s1)
    cameras_registered, images_registered, points3D_registered = store_transformed_roma(model_moving, s,R,T)
    
    # cameras_movng_save = {2: Camera(id=2, model=cameras_movng[1].model, width=cameras_movng[1].width, height=cameras_movng[1].height,
    #                         params=cameras_movng[1].params)}
    # cameras_movng_save = {}
    # cameras_fixed.update(cameras_movng_save)

    images_registered_save = {}
    camera_correspondence = {}
    fixed_count, moving_count = 0, 0
    for key, curr_image in images_registered.items():
        # training cameras in moving
        if key in sorted_keys_movng:
            idx = sorted_keys_movng.index(key)
            image_fixed = images_fixed[sorted_keys_fixed[idx]]

            camera_correspondence[curr_image.camera_id] = image_fixed.camera_id

            images_registered_save[key] = Image(
                id=curr_image.id, qvec=image_fixed.qvec, tvec=image_fixed.tvec,
                camera_id=curr_image.camera_id, name=image_fixed.name,
                xys=image_fixed.xys, point3D_ids=image_fixed.point3D_ids)
            
            moving_count += 1
        else:

            images_registered_save[key] = Image(
                id=curr_image.id, qvec=curr_image.qvec, tvec=curr_image.tvec,
                camera_id=curr_image.camera_id, name=curr_image.name,
                xys=np.zeros((1,2)), point3D_ids=(1)*np.ones((1,)).astype('uint8'))

            fixed_count += 1
    
    for k, v in camera_correspondence.items():
        cameras_movng[k] = Camera(id=k, model=cameras_fixed[v].model, width=cameras_fixed[v].width, height=cameras_fixed[v].height, params=cameras_fixed[v].params)

    print(moving_count)
    print(fixed_count)

    return cameras_movng, images_registered, images_registered_save, points3D_registered