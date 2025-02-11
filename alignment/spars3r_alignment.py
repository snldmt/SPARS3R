import os
import torch

from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
from PIL import Image
from plyfile import PlyData
import matplotlib

from segment_anything import sam_model_registry, SamPredictor

from read_model import read_model
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from utils import round_python3, ransac_procrustes, transform_points, get_sfm_points_in_mask, pairwise_iou, rigid_points_registration, sRT_to_4x4, geotrf, save_ply

import warnings
warnings.filterwarnings("ignore")


def show_points(coords, labels, ax, marker_size=150, gray=False):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    # gray=True
    if gray:
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='gray', marker='*', s=marker_size, edgecolor='white', linewidth=0.25)#, alpha=0.5)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='gray', marker='*', s=marker_size, edgecolor='white', linewidth=0.25)#, alpha=0.5)
    else:
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=0.25)#, alpha=0.5)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=0.25)#, alpha=0.5)

def show_mask(mask, ax, random_color=False, num=0):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    color = np.array([151,61,141, 255*0.6]) / 255
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)  

def same_mask(mask1, mask2, thresh=0.7):
    iou = np.logical_and(mask1, mask2).sum().astype(float)
    # print(iou/mask1.sum(), iou/mask2.sum())
    if iou/mask1.sum() > thresh or iou/mask2.sum() > thresh:
        return True
    else:
        return False

device = 'cuda'

parser = ArgumentParser(description="Training script parameters")
parser.add_argument( "--scene", type=str, default='', help="scene name")
parser.add_argument("--colmap_pts_thresh", type=int, default=10, help="at least this number of sfm points in a mask")
parser.add_argument("--min_num_outliers", type=float, default=0.1, help="if num of outliers/allpts >= min_num_outliers, then do SAM")
parser.add_argument("--align_error_threshold", type=float, default=1.5, help="error threshold for aligning dust3r and colmap points")
parser.add_argument("--iou_thresh", type=float, default=0.5, help="if two mask have iou > iou_thresh, they are the same mask")
parser.add_argument("--inout_thresh", type=float, default=0.3, help="inlier outlier threshold")
parser.add_argument("--source_path", type=str, default='', help="source path")
parser.add_argument( "--dust3r_or_mast3r", type=str, default='mast3r', help="use dust3r or mast3r")
parser.add_argument( "--sam_prefix", type=str, default='MASt3R_ioSAM', help="prefix for new sparse path")
parser.add_argument( "--orig_colmap_path", type=str, default='', help="train test aligned sfm path")
parser.add_argument( "--use_dust3r_mask", type=str, default='True', help="if use dust3r_mask")


args = parser.parse_args()


scene = args.scene
assert scene != '', 'Please specify the scene name'
print('scene:', scene)

source_path = f'{args.source_path}/{scene}'
print('source_path:', source_path)

orig_colmap_path = os.path.join(source_path, args.orig_colmap_path)
cameras, images, points3D = read_model(orig_colmap_path, '.txt')

if 'mast3r' in args.dust3r_or_mast3r:
    new_colmap_path = os.path.join(source_path,
                                f'sparse_{args.sam_prefix}_ptsth{args.colmap_pts_thresh}_IoU{args.iou_thresh}_alignerrth{args.align_error_threshold}_inoutth{args.inout_thresh}_minout{args.min_num_outliers}')

elif args.dust3r_or_mast3r == 'dust3r':
    assert 'mast3r' not in args.sam_prefix, 'now dust3r, should not have mast3r in the sam_prefix'
    assert 'MASt3R' not in args.sam_prefix, 'now dust3r, should not have mast3r in the sam_prefix'
    new_colmap_path = os.path.join(source_path,
                                f'sparse_{args.sam_prefix}_ptsth{args.colmap_pts_thresh}_IoU{args.iou_thresh}_alignerrth{args.align_error_threshold}_inoutth{args.inout_thresh}_minout{args.min_num_outliers}')

else:
    raise ValueError('args.dust3r_or_mast3r should be either dust3r or mast3r: {}'.format(args.dust3r_or_mast3r))


if not os.path.exists(new_colmap_path):
    os.makedirs(new_colmap_path)
print('new_colmap_path:', new_colmap_path)

vis_dir = os.path.join(new_colmap_path, 'vis')
os.makedirs(vis_dir, exist_ok=True)


train_list = f'{source_path}/train.txt'
if not os.path.exists(train_list):
    train_list = f'{source_path}/train_list.txt'
with open(train_list,'r') as f:
    train_images = [e.strip() for e in f.readlines()]
print(len(train_images), train_images[:5])

all_points_across_images = []
all_rgb_across_images = []
delete_because_of_align_error_too_large = 0
total_num_loops = 0


# Loop through each training camera
for num_c, image_name in enumerate(tqdm(train_images)):
    
    name = os.path.splitext(image_name)[0]
    print(name)
    
    os.makedirs(os.path.join(vis_dir,name),exist_ok=True)

    image = cv2.imread(source_path+'/images/'+image_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    H,W,_ = image.shape # [H,W,3]

    cam = [e for e in images.values() if e.name == image_name][0]

    m = np.array([True if e in points3D else False for e in cam.point3D_ids])
    point3D_ids = cam.point3D_ids[m]
    pts_2D = cam.xys[m]

    m = (pts_2D[:,0] >= 0) * (pts_2D[:,1] >= 0) * (pts_2D[:,0] < W) * (pts_2D[:,1] < H)
    point_loc = pts_2D[m].astype(int) 
    ids_loc = point3D_ids[m]

    ############################################################### global fusion alignment
    colmap_points = np.stack([points3D[e].xyz for e in ids_loc])

    dust3r_mask = np.load(f"{source_path}/{args.dust3r_or_mast3r}/individual_img_pts/{cam.name[:-4] + '_mask.npy'}")
    this_image_dust3r_points = np.load(f"{source_path}/{args.dust3r_or_mast3r}/individual_img_pts/{cam.name[:-4] + '.npy'}")
    DH, DW = this_image_dust3r_points.shape[:2]
    rgb_image = cv2.resize(image, (DW,DH))
    
    xys = point_loc / np.array([[W,H]])
    xys = np.clip(xys, 0, 1) 
    xys = (xys * np.array([[DW, DH]]))
    xys = np.array([round_python3(e) for e in xys.flatten()]).reshape(xys.shape)
    xys = np.minimum(xys, np.array([[DW-1,DH-1]]))
    xys = xys.astype(int)
    dust3r_points = np.stack([this_image_dust3r_points[y,x] for x,y in xys])

    dust3r_points = torch.from_numpy(dust3r_points).float()
    colmap_points = torch.from_numpy(colmap_points).float()

    best_transformation, best_inliers, best_res, best_iter = ransac_procrustes(dust3r_points, colmap_points,
                                                                            num_samples=min(10,len(dust3r_points)), 
                                                                            max_iterations=3000, 
                                                                            threshold=0.1,
                                                                            inlier_ratio=0.99
                                                                            )
    print('global align, best: #inliers',len(best_inliers), '{:.2f}%'.format(len(best_inliers)/len(dust3r_points)*100), 'iter:',best_iter)
    gs, gR, gT = best_transformation
    trf_d_points = transform_points(dust3r_points.numpy(), gR,gs,gT)
    trf_d_points = torch.from_numpy(trf_d_points)


    residuals = torch.linalg.norm(trf_d_points - colmap_points, axis=1)
    median_err = torch.median(residuals)
    print('&'*20)
    print('median error:', median_err)
    print('&'*20)
    outliers_mask = residuals > args.inout_thresh
    outliers_mask = outliers_mask.numpy()


    ratio = outliers_mask.sum() / outliers_mask.shape[0]
    num_outliers = outliers_mask.sum()
    if ratio < args.min_num_outliers or num_outliers < args.colmap_pts_thresh or median_err < 0.1:
        print('ratio:', ratio, 'num_outliers:', num_outliers)
        print('args.min_num_outliers:', args.min_num_outliers, 'args.colmap_pts_thresh:', args.colmap_pts_thresh)

        all_mast3r_points = this_image_dust3r_points[dust3r_mask].reshape(-1,3)
        all_mast3r_points = torch.from_numpy(all_mast3r_points).float()
        colors = rgb_image[dust3r_mask].reshape(-1,3)
        ga_colors = colors
        
        ga_trfed_mast3r_points = transform_points(all_mast3r_points.numpy(), gR,gs,gT)

        all_points_across_images.append(ga_trfed_mast3r_points)
        all_rgb_across_images.append(ga_colors)
        print('NOTE:', '+'*20)
        print('outliers_mask < min_num_outliers, skip')
        print('NOTE:', '+'*20)
        continue

    ############################################################### Semantic Outlier Alignment
    point_loc = point_loc[outliers_mask]
    xys_val = point_loc.copy() # outlier sfm

    ids_loc = ids_loc[outliers_mask]
    ids_loc_copy = ids_loc.copy()
    
    #load SAM
    sam = sam_model_registry["vit_h"](checkpoint="sam_weights/sam_vit_h_4b8939.pth")
    sam.cuda()
    predictor = SamPredictor(sam)

    plt.imshow(image)
    show_points(point_loc, np.ones_like((point_loc[:,0])), plt.gca()) 
    plt.savefig(os.path.join(vis_dir,name,'all_points_sfm.jpg'))
    plt.clf()

    predictor.set_image(image)

    proj = np.zeros((image.shape[0],image.shape[1])).astype(bool)
    proj_id = np.zeros((image.shape[0],image.shape[1])).astype('uint32')

    proj[point_loc[:,1],point_loc[:,0]] = True
    proj_id[point_loc[:,1],point_loc[:,0]] = ids_loc


    clusters = []
    attempts = 0
    while point_loc.shape[0] != 0:
        #prompt one point to get a mask
        #print(attempts,point_loc.shape[0])
        input_point = point_loc[[0]]
        input_label = np.array([1])
        # with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        best_idx = np.argsort(scores)[-1]
        mask, score = masks[best_idx], scores[best_idx]
        # mask = mask.astype(np.bool)
        #find shared points in this mask and update
        update = np.logical_and(proj, mask)
        new_points = np.where(update)
        new_points = np.stack([new_points[1], new_points[0]],-1)
        #end if already have more than 10 points, save as a cluster
        if new_points.shape[0] > args.colmap_pts_thresh:
            #print('find cluster', new_points.shape[0])
            proj = np.logical_and(proj, ~mask)
            new_ids = []
            for point in new_points:
                new_ids.append(proj_id[point[1]][point[0]])
                # print(new_ids[-1])
            print(new_points.shape)
            clusters.append([new_points, np.asarray(new_ids), mask,input_point])
        elif new_points.shape[0] == 0:
            #print('no more relationship found!')
            #remove this point, can probably be done more efficiently
            proj = np.logical_and(proj, ~mask)
            #sometimes the prompted mask does not include the original prompted point, weird but okay
            proj[input_point[0][1],input_point[0][0]] = False
        else:
            #not enough, try reprompting with all the points
            new_label = np.ones((new_points.shape[0]))
            # with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            masks, scores, logits = predictor.predict(
                point_coords=new_points,
                point_labels=new_label,
                multimask_output=True,
            )   
            best_idx = np.argsort(scores)[-1]
            mask_new, score_new = masks[best_idx], scores[best_idx]
            # mask_new = mask_new.astype(np.bool)
            update = np.logical_and(proj, mask)
            new_points_new = np.where(update)
            new_points_new = np.stack([new_points_new[1], new_points_new[0]],-1)
            if new_points_new.shape[0] > args.colmap_pts_thresh:
                #print('find cluster after second try', new_points.shape[0])
                all_mask = np.logical_or(mask_new, mask)
                proj = np.logical_and(proj, ~all_mask)
                new_ids = []
                for point in new_points:
                    new_ids.append(proj_id[point[1]][point[0]])
                for point in new_points_new:
                    new_ids.append(proj_id[point[1]][point[0]])
                    #print(new_ids[-1])
                print(new_points.shape)
                clusters.append([new_points_new, np.asarray(new_ids), all_mask,input_point])
            else:
                #print('no more relationship found after second try!')
                all_mask = np.logical_or(mask_new, mask)
                proj = np.logical_and(proj, ~all_mask)
                proj[input_point[0][1],input_point[0][0]] = False
        #get the remaining points, and move on
        point_loc = np.where(proj)
        point_loc = np.stack([point_loc[1], point_loc[0]],-1)
        attempts += 1


    if True:
        print('address overlap')
        update_clusters = []
        leftover_clusters = clusters
        while len(leftover_clusters) >= 1:
            #print(len(update_clusters))
            if len(leftover_clusters) == 1:
                curr_cluster = leftover_clusters[0]
                be_merged = False
                for j, cluster_j in enumerate(update_clusters):
                    # print('update_cluster',j)
                    #doesn't handle more than 2 clusters that are the same? 
                    if same_mask(curr_cluster[2], cluster_j[2], thresh=args.iou_thresh):
                        be_merged = True
                        new_points = np.concatenate([curr_cluster[0], cluster_j[0]])
                        new_ids = np.concatenate([curr_cluster[1], cluster_j[1]])
                        new_mask = np.logical_or(curr_cluster[2], cluster_j[2])
                        curr_cluster = [new_points, new_ids, new_mask, cluster_j[3]]
                        # popped_cluster.append(j)
                        break
                if be_merged:
                    update_clusters[j] = curr_cluster
                    #print('merged with updated cluster ',j)
                else:
                    update_clusters.append(curr_cluster)
                break
            curr_cluster = leftover_clusters[0]
            leftover_clusters = leftover_clusters[1:]
            popped_cluster = []
            for j, cluster_j in enumerate(leftover_clusters):
                #print('leftover_cluster',j)
                #doesn't handle more than 2 clusters that are the same? 
                if same_mask(curr_cluster[2], cluster_j[2], thresh=args.iou_thresh):
                    new_points = np.concatenate([curr_cluster[0], cluster_j[0]])
                    new_ids = np.concatenate([curr_cluster[1], cluster_j[1]])
                    new_mask = np.logical_or(curr_cluster[2], cluster_j[2])
                    curr_cluster = [new_points, new_ids, new_mask, cluster_j[3]]
                    popped_cluster.append(j)
                    #print('merged ',j)
            if len(popped_cluster) > 0:
                #pop those cluster/select left over cluster
                leftover_ind = [x for x in range(len(leftover_clusters)) if not x in popped_cluster]
                #print(popped_cluster,leftover_ind)
                leftover_clusters = [leftover_clusters[x] for x in leftover_ind]
            be_merged = False
            for j, cluster_j in enumerate(update_clusters):
                # print('update_cluster',j)
                #doesn't handle more than 2 clusters that are the same? 
                if same_mask(curr_cluster[2], cluster_j[2], thresh=args.iou_thresh):
                    be_merged = True
                    new_points = np.concatenate([curr_cluster[0], cluster_j[0]])
                    new_ids = np.concatenate([curr_cluster[1], cluster_j[1]])
                    new_mask = np.logical_or(curr_cluster[2], cluster_j[2])
                    curr_cluster = [new_points, new_ids, new_mask, cluster_j[3]]
                    # popped_cluster.append(j)
                    break
            if be_merged:
                update_clusters[j] = curr_cluster
                #print('merged with updated cluster ',j)
            else:
                update_clusters.append(curr_cluster)
    

    global_mask = np.zeros((H,W), dtype=bool) # To track the final segmented mask
    new_clusters = []
    verbose = True
    
    from functools import partial
    def get_num_sfmpts(mask, sfm_points_2d):
        points_in_mask, pm = get_sfm_points_in_mask(mask[2], sfm_points_2d)
        return len(points_in_mask)#, mask[2].sum().item()
    func = partial(get_num_sfmpts, sfm_points_2d=xys_val)
    
    update_clusters = sorted(update_clusters, key=func, reverse=True)

    for i, curr_cluster in enumerate(update_clusters):
        mask = curr_cluster[2]
        mask = np.logical_and(mask, ~global_mask)        
        if mask.sum() == 0:
            if verbose:
                print('skip mask {} because mask.sum() == 0'.format(i+1))
            continue

        points_in_mask, pm = get_sfm_points_in_mask(mask, xys_val)
        if len(points_in_mask) == 0:
            if verbose:
                print('mask {} contains no SfM points, skip'.format(i+1))
            continue
        
        global_mask = np.logical_or(global_mask, mask)
        
        curr_cluster[2] = mask
        new_clusters.append(curr_cluster)

    print('pairwise IoU:')
    refined_masks_values = np.stack([e[2] for e in new_clusters], axis=0)
    print(pairwise_iou(torch.from_numpy(refined_masks_values)))

    for i, cluster in enumerate(new_clusters):
        plt.title('cluster '+str(i)+' num of points '+str(len(cluster[0])))
        plt.imshow(image)
        show_points(cluster[0], np.ones_like((cluster[0][:,0])), plt.gca())
        show_mask(cluster[2], plt.gca())
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(vis_dir+f'/{name}'+'/cluster_'+str(i)+'.jpg')
        plt.clf()

        Image.fromarray(cluster[2]).save(vis_dir+f'/{name}'+'/cluster_'+str(i)+'_mask.jpg')
    
    global_mask = np.logical_or.reduce([e[2] for e in new_clusters])
    plt.imshow(image)
    show_mask(global_mask, plt.gca())
    plt.savefig(vis_dir+f'/{name}'+'/global_mask.jpg')
    plt.clf()

    dust3r_mask = np.load(f"{source_path}/{args.dust3r_or_mast3r}/individual_img_pts/{cam.name[:-4] + '_mask.npy'}")
    if args.use_dust3r_mask == 'False':
        raise NotImplementedError
        print('use_dust3r_mask is False!')
        dust3r_mask = np.ones_like(dust3r_mask).astype(bool)

    show_mask(dust3r_mask, plt.gca())
    plt.savefig(vis_dir+f'/{name}'+'/dust3r_mask.jpg')
    plt.clf()

    this_image_dust3r_points = np.load(f"{source_path}/{args.dust3r_or_mast3r}/individual_img_pts/{cam.name[:-4] + '.npy'}")

    global_mask = TF.resize(torch.from_numpy(global_mask).unsqueeze(0).unsqueeze(0), 
                        size=dust3r_mask.shape,
                        interpolation=InterpolationMode.NEAREST).squeeze()
    ga_mask = np.logical_and(~global_mask.numpy(), dust3r_mask)


    plt.imshow(rgb_image)
    show_mask(ga_mask, plt.gca())
    plt.savefig(vis_dir+f'/{name}'+'/global_align_mask.jpg')
    plt.clf()

    
    all_mast3r_points = torch.from_numpy(this_image_dust3r_points[ga_mask]).float()
    ga_trfed_mast3r_points = transform_points(all_mast3r_points.numpy(), gR,gs,gT)

    pts = np.stack(ga_mask.nonzero()).T 
    ga_colors = np.stack([rgb_image[r,c] for r,c in pts])


    with torch.no_grad():
        all_points = []
        all_rgbs = []

        for i, mask in enumerate(refined_masks_values):
            points_in_mask, pm = get_sfm_points_in_mask(mask, xys_val)
            if len(points_in_mask) < args.colmap_pts_thresh:
                continue
            colmap_points = np.stack([points3D[e].xyz for e in ids_loc[pm]])

            mask = TF.resize(torch.from_numpy(mask).unsqueeze(0).unsqueeze(0), 
                             size=dust3r_mask.shape,
                             interpolation=InterpolationMode.NEAREST).squeeze()
            mask = np.logical_and(mask.numpy(), dust3r_mask)
            
            if mask.sum() < args.colmap_pts_thresh:
                print(i, 'after merging with dust3r mask, mask size < 10, skip!!!!!!!!!!!!!!!!!! len(points_in_mask)', len(points_in_mask))
                continue
            
            # print('dust3r size', DW, DH)
            xys = points_in_mask / np.array([[W,H]])
            xys = np.clip(xys, 0, 1)
            xys = (xys * np.array([[DW, DH]]))
            xys = np.array([round_python3(e) for e in xys.flatten()]).reshape(xys.shape)
            xys = np.minimum(xys, np.array([[DW-1,DH-1]]))
            xys = xys.astype(int)
            dust3r_points = np.stack([this_image_dust3r_points[y,x] for x,y in xys])

            dust3r_points = torch.from_numpy(dust3r_points).float()
            colmap_points = torch.from_numpy(colmap_points).float()


            all_pts_of_mask = np.stack(mask.nonzero()).T # row col
            all_dust3r_points = np.array([this_image_dust3r_points[r,c] for r,c in all_pts_of_mask])
            if True:
                try:
                    best_transformation, best_inliers, best_res, best_iter = ransac_procrustes(dust3r_points, colmap_points,
                                                                                            num_samples=min(10,len(dust3r_points)), 
                                                                                            max_iterations=3000, 
                                                                                            threshold=0.1,
                                                                                            inlier_ratio=0.99)
        
                    
                    print(i,'best: #inliers',len(best_inliers), '{:.2f}%'.format(len(best_inliers)/len(dust3r_points)*100), 'iter:',best_iter)#, best_res)
                    try:
                        s, R, T = best_transformation
                        failed_use_roma = False
                    except:
                        failed_use_roma = True

                    if failed_use_roma:
                        print(':(  :(  :(  :(  :(')
                        print('RANSAC cannot resolve this problem')
                        print(':(  :(  :(  :(  :(')
                        s,R,T = rigid_points_registration(dust3r_points, colmap_points, None)
                        trf = sRT_to_4x4(s, R, T, device=dust3r_points.device)
                        trf_d_points = geotrf(trf, 
                                            dust3r_points)
                        error = torch.abs(trf_d_points - colmap_points).mean(0)
                        print(i,'mean L1 error after trf:', error)

                        if torch.any(error > args.align_error_threshold):
                            delete_because_of_align_error_too_large += 1
                            print(i,'*'*10, 'delete because of align error too large', 'align_error_threshold:', args.align_error_threshold)
                            total_num_loops += 1
                            continue

                        trf_d_points = geotrf(trf, 
                                            torch.from_numpy(all_dust3r_points).float())
                        trf_d_points = trf_d_points.numpy()

                    else:
                        trf_d_points = transform_points(dust3r_points.numpy(), R,s,T)
                        error = np.abs(trf_d_points - colmap_points.numpy()).mean(0)
                        print(i,'mean L1 error after trf:', error)
                        
                        if np.any(error > args.align_error_threshold):
                            delete_because_of_align_error_too_large += 1
                            print(i,'*'*10, 'delete because of align error too large', 'align_error_threshold:', args.align_error_threshold)
                            total_num_loops += 1
                            continue

                        trf_d_points = transform_points(all_dust3r_points, R,s,T)
                
                except:
                    breakpoint()
            

            dust3r_rgb = np.stack([rgb_image[r,c] for r,c in all_pts_of_mask])

            all_points.append(trf_d_points)
            all_rgbs.append(dust3r_rgb)

            total_num_loops += 1

    if len(all_points) == 0:
        continue

    all_points = np.concatenate(all_points)
    all_rgbs = np.concatenate(all_rgbs)
    
    all_points = np.concatenate([ga_trfed_mast3r_points, all_points])
    all_rgbs = np.concatenate([ga_colors, all_rgbs])

    all_points_across_images.append(all_points)
    all_rgb_across_images.append(all_rgbs)

    del sam, predictor, image
    torch.cuda.empty_cache()

all_points_across_images = np.concatenate(all_points_across_images)
all_rgb_across_images = np.concatenate(all_rgb_across_images)
all_rgb_across_images = all_rgb_across_images.astype(np.uint8)

print('-'*20)
print('total # points:', len(all_points_across_images))
print('-'*20)


ply_file_path = os.path.join(new_colmap_path, f'SAM_aligned_{args.dust3r_or_mast3r}_points.ply')
save_ply(ply_file_path, all_points_across_images, all_rgb_across_images)
# storePly(ply_file_path, all_points_across_images, all_rgb_across_images)

### save new sparse folder
if not os.path.exists(f'{new_colmap_path}/0'):
    os.makedirs(f'{new_colmap_path}/0')
os.system(f"cp {orig_colmap_path}/* {new_colmap_path}/0/") 
os.system(f'rm {new_colmap_path}/0/*.bin {new_colmap_path}/0/*.ply')
points3D = [f'{999999+pid} {pt[0]} {pt[1]} {pt[2]} {color[0]} {color[1]} {color[2]} 0 0 0\n' for pid, (pt, color) in enumerate(zip(all_points_across_images, all_rgb_across_images))]
print('write points3D', points3D[:3])
with open(f'{new_colmap_path}/0/points3D.txt','a') as f:
    f.writelines(points3D)



