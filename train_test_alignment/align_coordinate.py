#!/usr/bin/env python3
# Author: Jason Bunk
import os
import argparse
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import scipy

from colmap_read_write_model import (
    read_images_text,
    read_images_binary,
    read_points3D_text,
    read_points3D_binary,
    read_cameras_text,
    read_cameras_binary,
    write_model, 
    read_model,
    Image,
)
from robust_coordinate_system_alignment_orig import (
    robust_register_colmap_models_dm, robust_register_colmap_models_ins,
    colmap_image_to_extrinsics_c2w,
    calculate_rotation_error
)
from utils.utils_poses.ATE import compute_rpe, compute_ATE


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

    return mtx1, mtx2, R

def fix_train_only(orig_path, train_path, final_output_path):
    cameras_train, images_train, points3D_train = read_model(train_path)
    cameras_orig, images_orig, points3D_orig = read_model(orig_path)

    train_names = [images_orig[x].name for x in images_orig.keys()]
    name2key = {}
    for key in images_orig.keys():
        name2key[images_orig[key].name] = key

    for key in images_train.keys():
        if images_train[key].name in train_names:
            # print('train', images_train[key].name)
            orig_img = images_orig[name2key[images_train[key].name]]
            curr_img = images_train[key]
            images_train[key] = Image(
                id=curr_img.id, qvec=curr_img.qvec, tvec=curr_img.tvec,
                camera_id=curr_img.camera_id, name=curr_img.name,
                xys=orig_img.xys, point3D_ids=orig_img.point3D_ids)
        else:
            curr_img = images_train[key]
            images_train[key] = Image(
                id=curr_img.id, qvec=curr_img.qvec, tvec=curr_img.tvec,
                camera_id=curr_img.camera_id, name=curr_img.name,
                xys=np.zeros((1,2)), point3D_ids=(-1)*np.ones((1,)).astype('uint8'))
    
    write_model(cameras_train, images_train, points3D_train, final_output_path)

def read_model_images(impath: str):
    if impath.endswith(".txt"):
        ret = read_images_text(impath)
    elif impath.endswith(".bin"):
        ret = read_images_binary(impath)
    else:
        assert 0, impath
    
    r2 = {}
    for key, val in ret.items():
        r2[key] = val._replace(name=os.path.basename(val.name))
    return r2

def read_model_points3D(impath: str):
    if impath.endswith(".txt"):
        ret = read_points3D_text(impath)
    elif impath.endswith(".bin"):
        print(impath)
        ret = read_points3D_binary(impath)
    else:
        assert 0, impath
    return ret

def read_model_cameras(impath: str):
    if impath.endswith(".txt"):
        ret = read_cameras_text(impath)
    elif impath.endswith(".bin"):
        print(impath)
        ret = read_cameras_binary(impath)
    else:
        assert 0, impath
    return ret

def pose_metrics(c2ws_est_aligned, poses_gt,string, scene_path):
    rot_errs = []
    pos_errs = []
    for ii in range(len(poses_gt)):
        q_true = Rotation.from_matrix(poses_gt[ii, :3, :3]).as_quat()
        q_gues = Rotation.from_matrix(c2ws_est_aligned[ii, :3, :3]).as_quat()
        rot_error = calculate_rotation_error(q_true, q_gues)

        pos_error = np.linalg.norm(c2ws_est_aligned[ii, :3, 3] - poses_gt[ii, :3, 3])
        rot_errs.append(rot_error)
        pos_errs.append(pos_error)
    print('POS ERROR:',np.mean(pos_errs))
    print('ROT ERROR:',np.mean(rot_errs))

    ate = compute_ATE(poses_gt,
                    c2ws_est_aligned)
    rpe_trans, rpe_rot,_,_ = compute_rpe(
        poses_gt, c2ws_est_aligned)
    print("\n")
    print(   
        "{}_RPE_trans: {:.3f}".format(string, rpe_trans*100),
        '& {}_RPE_rot: ' "{:.3f}".format(string,rpe_rot * 180 / np.pi),
        "& {}_ATE: {:.3f}".format(string,ate))
    print("\n")

    with open(os.path.join(scene_path, f"pose_train.txt"), 'w') as f:
            f.write("{}_RPE_trans: {:.04f}, {}_RPE_rot: {:.04f}, {}_ATE: {:.04f} \n".format(
                string,
                rpe_trans*100,
                string,
                rpe_rot * 180 / np.pi,
                string,
                ate))
            f.write(' '.join(['POS ERROR: ', str(np.mean(pos_errs)), 'ROT ERROR: ',str(np.mean(rot_errs)),'\n']))
            f.close()

def measure_difference_in_colmap_poses_by_aligning_coordinate_systems(
    true_images_txt_or_bin: str, 
    guess_images_txt_or_bin: str, 
    output_path: str,
    ransac_translation_error_threshold: float = 0.5,
):
    try:
        images_true = read_model_images(os.path.join(true_images_txt_or_bin, 'images.bin'))
        camera_true = read_model_cameras(os.path.join(true_images_txt_or_bin, 'cameras.bin'))
        points_true = read_model_points3D(os.path.join(true_images_txt_or_bin, 'points3D.bin'))
        images_gues = read_model_images(os.path.join(guess_images_txt_or_bin, 'images.bin'))
        camera_gues = read_model_cameras(os.path.join(guess_images_txt_or_bin, 'cameras.bin'))
        
    except:
        images_true = read_model_images(os.path.join(true_images_txt_or_bin, 'images.txt'))
        camera_true = read_model_cameras(os.path.join(true_images_txt_or_bin, 'cameras.txt'))
        points_true = read_model_points3D(os.path.join(true_images_txt_or_bin, 'points3D.txt'))
        images_gues = read_model_images(os.path.join(guess_images_txt_or_bin, 'images.txt'))
        camera_gues = read_model_cameras(os.path.join(guess_images_txt_or_bin, 'cameras.txt'))

    print('Length of points_true', len(points_true))

    true_key_to_ransac_threshold = {}


    (camera_save, images_gues, images_gues_save, point3d) = robust_register_colmap_models_dm(
        (camera_gues, images_gues, {}),
        (camera_true, images_true, {}),
        ransac_distance_target=ransac_translation_error_threshold,
        true_key_to_ransac_threshold=true_key_to_ransac_threshold,
    )    

    os.makedirs(output_path,exist_ok=True)
    name2key_gues = {image.name: key for key, image in images_gues.items()}
    name2key_true = {image.name: key for key, image in images_true.items()}
    intersecting_names = sorted(list(set(name2key_gues.keys()) & set(name2key_true.keys())))
    difference = sorted(list(set(name2key_gues.keys()) - set(name2key_true.keys())))
    sorted_keys_gues = [name2key_gues[name] for name in intersecting_names]
    sorted_keys_true = [name2key_true[name] for name in intersecting_names]

    c2w_gues = np.stack([colmap_image_to_extrinsics_c2w(images_gues[key]) for key in sorted_keys_gues]).astype(np.float64)
    c2w_true = np.stack([colmap_image_to_extrinsics_c2w(images_true[key]) for key in sorted_keys_true]).astype(np.float64)

    rot_errs = []
    pos_errs = []
    remove_list = []
    trans_gt_align, trans_ours_align, _ = align_pose(c2w_true[:, :3, -1],
                                                            c2w_gues[:, :3, -1])

    c2w_true[:, :3, -1] = trans_gt_align
    c2w_gues[:, :3, -1] = trans_ours_align
    pose_metrics(c2w_gues, c2w_true,'ours', output_path)
    # return pos_errs, rot_errs

    for ii in range(len(c2w_true)):
        q_true = Rotation.from_matrix(c2w_true[ii, :3, :3]).as_quat()
        q_gues = Rotation.from_matrix(c2w_gues[ii, :3, :3]).as_quat()
        
        rot_error = calculate_rotation_error(q_true, q_gues)

        pos_error = np.linalg.norm(c2w_true[ii, :3, 3] - c2w_gues[ii, :3, 3])
        if pos_error > 1.0 or pos_error == np.nan:
            remove_list.append(sorted_keys_gues[ii])
        rot_errs.append(rot_error)
        pos_errs.append(pos_error)


    print('Remove', remove_list)
    
    for remove_key in remove_list:
        print(images_gues[remove_key])
        
    images_gues_remove = {key: value for key, value in images_gues_save.items() if key not in remove_list}
    print('Length after removal', len(images_gues_remove))
    # points_true = {}
    
    write_model(camera_save, images_gues_remove, points_true, output_path)

    return pos_errs, rot_errs


def evaluate_colmap_poses_by_aligning_coordinate_systems(
    true_images_txt_or_bin: str,  #train only
    guess_images_txt_or_bin: str,  #train+test
    output_path: str,
    # final_output_path: str,
    translation_error_threshold: float,
    print_prefix:str="",
):
    pos_errs, rot_errs = measure_difference_in_colmap_poses_by_aligning_coordinate_systems(
        true_images_txt_or_bin,
        guess_images_txt_or_bin,
        output_path,
        ransac_translation_error_threshold=translation_error_threshold,
    )
    print('POS ERROR:',pos_errs)
    print('ROT ERROR:',rot_errs)
    
    rot_thresh = 5
    num_images_rot_correct = np.less(rot_errs, rot_thresh).astype(np.int64).sum()
    num_images_transl_corr = np.less(pos_errs, translation_error_threshold).astype(np.int64).sum()

    print(print_prefix+f"ground truth has {len(pos_errs)} cameras, gave predictions for {np.count_nonzero(np.isfinite(pos_errs))} cameras")
    print(print_prefix+f"rotation: {num_images_rot_correct} of {len(pos_errs)} correct within {rot_thresh} degrees; median pose error {np.median(rot_errs)}; mean pose error {np.mean(rot_errs)}")
    print(print_prefix+f"position: {num_images_transl_corr} of {len(pos_errs)} correct within {translation_error_threshold} meters; median pose error {np.median(pos_errs)}; mean pose error {np.mean(pos_errs)}")

    return num_images_rot_correct, num_images_transl_corr, len(pos_errs)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--true_images", type=str,default='train_sparse/sfm')
    parser.add_argument("--guessimages", type=str,default='train+test_sparse/sfm')
    parser.add_argument("--output_path", type=str,default='train+test_aligned')
    parser.add_argument("-t", "--translation_error_threshold", type=float, default=1.)
    args = parser.parse_args()

    evaluate_colmap_poses_by_aligning_coordinate_systems(
        args.true_images,
        args.guessimages,
        args.output_path,
        # args.final_output_path,
        args.translation_error_threshold,
        print_prefix="  ",
    )
