import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from copy import deepcopy
#import kornia as K
#import kornia.feature as KF
#import torch
from time import time
import cv2
import numpy as np
import math
from tqdm import tqdm
#from joblib import Parallel, delayed
import sys
import gc
import itertools
import pdb
import re
from scipy.spatial.transform import Rotation as R

def load_h5(filename):
    '''Loads dictionary from hdf5 file'''
    dict_to_load = {}

    if not os.path.exists(filename):
        print('Cannot find file {}'.format(filename))
        return dict_to_load

    with h5py.File(filename, 'r') as f:
        keys = [key for key in f.keys()]
        for key in keys:
            dict_to_load[key] = f[key][()]
    return dict_to_load


def quaternion_from_matrix(matrix, isprecise=False):
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4, ))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]

        # symmetric matrix K
        K = np.array([[m00 - m11 - m22, 0.0, 0.0, 0.0],
                      [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                      [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                      [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]])
        K /= 3.0

        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]

    if q[0] < 0.0:
        np.negative(q, q)

    return q

def detect_sift(image, feature_number = 8000, eps = 1e-7):
    det = cv2.SIFT_create(feature_number)    
    kps, descs = det.detectAndCompute(image, None)
    keypoints = np.array([[k.pt[0], k.pt[1], k.angle / 180.0 * math.pi, k.size] for k in kps])    
    descs /= (descs.sum(axis=1, keepdims=True) + eps)
    descs = np.sqrt(descs)
    return keypoints, descs

def normalize_keypoints(keypoints, K):
    '''Normalize keypoints using the calibration data.'''
    C_x = K[0, 2]
    C_y = K[1, 2]
    f_x = K[0, 0]
    f_y = K[1, 1]
    keypoints = (keypoints - np.array([[C_x, C_y]])) / np.array([[f_x, f_y]])

    return keypoints

def evaluate_R_t(R_gt, t_gt, R, t, q_gt=None):
    t = t.flatten()
    t_gt = t_gt.flatten()

    eps = 1e-15

    if q_gt is None:
        q_gt = quaternion_from_matrix(R_gt)
    q = quaternion_from_matrix(R)
    q = q / (np.linalg.norm(q) + eps)
    q_gt = q_gt / (np.linalg.norm(q_gt) + eps)
    loss_q = np.maximum(eps, (1.0 - np.sum(q * q_gt)**2))
    err_q = np.arccos(1 - 2 * loss_q)

    t = t / (np.linalg.norm(t) + eps)
    t_gt = t_gt / (np.linalg.norm(t_gt) + eps)
    loss_t = np.maximum(eps, (1.0 - np.sum(t * t_gt)**2))
    err_t = np.arccos(np.sqrt(1 - loss_t))

    if np.sum(np.isnan(err_q)) or np.sum(np.isnan(err_t)):
        # This should never happen! Debug here
        print(R_gt, t_gt, R, t, q_gt)
        import IPython
        IPython.embed()

    return err_q, err_t

def eval_essential_matrix(p1n, p2n, E, dR, dt):
    if len(p1n) != len(p2n):
        raise RuntimeError('Size mismatch in the keypoint lists')

    if p1n.shape[0] < 5:
        return np.pi, np.pi / 2

    if E.size > 0:
        _, R, t, _ = cv2.recoverPose(E, p1n, p2n)
        err_q, err_t = evaluate_R_t(dR, dt, R, t)
    else:
        err_q = np.pi
        err_t = np.pi / 2

    return err_q, err_t

# A function the relative pose between two images given the absolute ones
def get_relative_pose(R1, R2, T1, T2):
    R12 = R2 @ R1.T
    T12 = T2 - R12 @ T1
    return R12, np.squeeze(T12)

# A function to calculate the essential matrix from the relative pose
def get_essential_matrix(R12, T12):
    T12x = np.array([[0, -T12[2], T12[1]],
                     [T12[2], 0, -T12[0]],
                     [-T12[1], T12[0], 0]])
    E = T12x @ R12
    return E

# A function to compute the Sampson error for a given set of correspondences
def compute_sampson_error(correspondences, E):
    num_pts = correspondences.shape[0]
    pts1 = correspondences[:, 0:2]
    pts2 = correspondences[:, 2:4]
    
    # Ensure points are in homogeneous coordinates
    ones = np.ones((pts1.shape[0], 1))
    pts1_h = np.hstack([pts1, ones])
    pts2_h = np.hstack([pts2, ones])

    # Compute the epipolar lines
    F_times_x1 = np.dot(E, pts1_h.T).T
    Ft_times_x2 = np.dot(E.T, pts2_h.T).T

    # Evaluate the Sampson distance
    numerator = np.square(np.sum(pts2_h * F_times_x1, axis=1))

    # Compute the denominator term
    F_times_x1_squared = np.square(F_times_x1[:, :2])
    Ft_times_x2_squared = np.square(Ft_times_x2[:, :2])
    denominator = F_times_x1_squared[:, 0] + F_times_x1_squared[:, 1] + Ft_times_x2_squared[:, 0] + Ft_times_x2_squared[:, 1]

    # Avoid division by zero
    denominator = np.where(denominator == 0, 1, denominator)

    # Compute the Sampson error
    sampson_error = numerator / denominator
    
    return sampson_error

def PhotoTourism(data_path, scene, output_path, output_path_list):
    # Loading data
    K1_K2 = load_h5(f'{data_path}/{scene}/K1_K2.h5')
    R = load_h5(f'{data_path}/{scene}/R.h5')
    T = load_h5(f'{data_path}/{scene}/T.h5')
    matches = load_h5(f'{data_path}/{scene}/matches.h5')
    matches_scores = load_h5(f'{data_path}/{scene}/match_conf.h5')

    # Iterating through pairs
    images = {}
    for pair in K1_K2.keys():
        image_src = pair.split('-')[0]
        image_dst = pair.split('-')[1]

        if image_src in images.keys():
            images[image_src].append(image_dst)
        else:
            images[image_src] = [image_dst]

    triplets = []
    pair_number = 10000
    for image in images.keys():
        if len(images[image]) < 2:
            continue
        
        for subset in itertools.combinations(images[image], 2):
            triplets.append((image, subset[0], subset[1]))
            if len(triplets) >= pair_number:
                break
        if len(triplets) >= pair_number:
            break

    bf = cv2.BFMatcher()
    epsilon = 1
    crossCheck = False
    database = h5py.File(output_path, 'w')
    success = 0
    with open(output_path_list, "w") as list_file:
        for i in tqdm(range(len(triplets))):
            triplet = triplets[i]
            img1 = triplet[0]
            img2 = triplet[1]
            img3 = triplet[2]
            label = f"{img1}-{img2}-{img3}"

            if label in database.keys():
                continue
            
            # Detect features in the 1st image
            if f"{img1}-feat" in database.keys():
                feat1 = np.array(database[f"{img1}-feat"])
                desc1 = np.array(database[f"{img1}-desc"])
            else:
                image1 = cv2.imread(os.path.join(data_path, scene, "images", img1 + ".jpg"))
                feat1, desc1 = detect_sift(image1)
                database.create_dataset(f"{img1}-feat", data=feat1)
                database.create_dataset(f"{img1}-desc", data=desc1)
            
            # Detect features in the 2nd image
            if f"{img2}-feat" in database.keys():
                feat2 = np.array(database[f"{img2}-feat"]) 
                desc2 = np.array(database[f"{img2}-desc"]) 
            else:
                image2 = cv2.imread(os.path.join(data_path, scene, "images", img2 + ".jpg"))
                feat2, desc2 = detect_sift(image2)
                database.create_dataset(f"{img2}-feat", data=feat2)
                database.create_dataset(f"{img2}-desc", data=desc2)
            
            # Detect features in the 3rd image
            if f"{img3}-feat" in database.keys():
                feat3 = np.array(database[f"{img3}-feat"])
                desc3 = np.array(database[f"{img3}-desc"]) 
            else:
                image3 = cv2.imread(os.path.join(data_path, scene, "images", img3 + ".jpg"))
                feat3, desc3 = detect_sift(image3)
                database.create_dataset(f"{img3}-feat", data=feat3)
                database.create_dataset(f"{img3}-desc", data=desc3)
            
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
            if crossCheck:
                matches12 = bf.knnMatch(desc1, desc2, k=1)
                matches13 = bf.knnMatch(desc1, desc3, k=1)
                matches23 = bf.knnMatch(desc2, desc3, k=1)
            else:
                matches12 = bf.knnMatch(desc1, desc2, k=2)
                matches13 = bf.knnMatch(desc1, desc3, k=2)
                matches23 = bf.knnMatch(desc2, desc3, k=2)
            triplet_correspondences = []
            
            for i in range(len(matches12)):
                if crossCheck:
                    try:
                        m12 = matches12[i][0]
                        queryIdx12 = m12.queryIdx # In image 1
                        trainIdx12 = m12.trainIdx # In image 2
                        
                        m13 = matches13[i][0]
                        queryIdx13 = m13.queryIdx # In image 1
                        trainIdx13 = m13.trainIdx # In image 3
                        
                        m23 = matches23[trainIdx12][0]
                        queryIdx23 = m23.queryIdx # In image 2
                        trainIdx23 = m23.trainIdx # In image 3
                    except:
                        continue
                    
                    snn12 = 0
                    snn13 = 0
                    snn23 = 0
                else:
                    m12, n12 = matches12[i]
                    queryIdx12 = m12.queryIdx # In image 1
                    trainIdx12 = m12.trainIdx # In image 2
                    
                    m13, n13 = matches13[i]
                    queryIdx13 = m13.queryIdx # In image 1
                    trainIdx13 = m13.trainIdx # In image 3
                    
                    m23, n23 = matches23[trainIdx12]
                    queryIdx23 = m23.queryIdx # In image 2
                    trainIdx23 = m23.trainIdx # In image 3
                
                    snn12 = m12.distance / n12.distance
                    snn13 = m13.distance / n13.distance
                    snn23 = m23.distance / n23.distance
                
                # Cycle consistency check
                if trainIdx12 == queryIdx23 and trainIdx13 == trainIdx23:
                    kp1 = feat1[m12.queryIdx]
                    kp2 = feat2[m12.trainIdx]
                    kp3 = feat3[m13.trainIdx]
                    triplet_correspondences.append([kp1[0], kp1[1], kp2[0], kp2[1], kp3[0], kp3[1], snn12, snn13, snn23])
            triplet_correspondences = np.array(triplet_correspondences)
            
            database.create_dataset(label, data=triplet_correspondences)
            list_file.write(f"{img1} {img2} {img3}\n")
            
            success += 1
            if success >= 5000:
                break
    database.close()

def ETH3D(data_path, scene, output_path, output_path_list):
    # Loading data
    cameras = open(f'{data_path}/{scene}/dslr_calibration_undistorted/cameras.txt', 'r')
    cam_data = None
    # cam_data: (no.cam) X [width, height, focal_x, focal_y, cx, cy]
    while(True):
        line = cameras.readline()
        if(line == ''):
            break
        if(line[:-3] == '# Number of cameras:'):
            cam_data = np.zeros((int(line[-2]), 6), dtype=np.float32)
        if(line[0] == '#'):
            continue
        if(cam_data is None):
            break
        tok = line.split(' ')
        cam_data[int(tok[0])] = np.array([float(tok[2]), float(tok[3]), float(tok[4]), float(tok[5]), float(tok[6]), float(tok[7])])

    # image_data: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    gt_matrices = {}
    image_data = open(f'{data_path}/{scene}/dslr_calibration_undistorted/images.txt', 'r')
    while(True):
        line = image_data.readline()
        tmp_tok = re.split(', |:',line)
        if(line == ''):
            break
        if(line[0] == '#'):
            continue
        tok = re.split(' |\n',line)
        # Quaternion stored as scalar-last (X, Y, Z, W)
        gt_matrices[tok[-2]] = {'q': [float(tok[2]), float(tok[3]), float(tok[4]), float(tok[1])], 
                                't': [float(tok[5]), float(tok[6]), float(tok[7])], 
                                'cam': int(tok[8])}
        line = image_data.readline()
        index3d = np.array([float(x) for x in re.split(' |\n', line)[:-1]]).reshape(-1, 3)
        gt_matrices[tok[-2]]['index3d'] = set(index3d[:, 2])

    # Write cam matrices on h5 file
    image_names = gt_matrices.keys()
    '''
    camera_params = h5py.File(f"/datagrid/personal/tzamocha/datasets/ETH3D/{scene}/parameters.h5", 'w')
    for image in image_names:
        # camera index: gt_matrices[image]['cam']
        r = R.from_quat(gt_matrices[image]['q'])
        camera_params.create_dataset(f"{image}-R", data=r.as_matrix())
        camera_params.create_dataset(f"{image}-T", data=np.array(gt_matrices[image]['t']))
        intrinsic = cam_data[gt_matrices[image]['cam']]
        K = np.array([[intrinsic[2], 0, intrinsic[4]], [0, intrinsic[3], intrinsic[5]], [0, 0, 1]])
        camera_params.create_dataset(f"{image}-K", data=K)
        camera_params.create_dataset(f"{image}-width", data=intrinsic[0])
        camera_params.create_dataset(f"{image}-height", data=intrinsic[1])
    camera_params.close()
    '''
    images = {}
    for img1 in image_names:
        for img2 in image_names:
            pointsInCommon = len(gt_matrices[img1]['index3d'].intersection(gt_matrices[img2]['index3d']))
            if(img1 == img2):
                continue
            if(pointsInCommon < 500):
                print(f"Skipping {img1} {img2}: {pointsInCommon} 3D points in common.")
                continue
            if(img1 in images.keys()):
                images[img1].append(img2)
            else:
                images[img1] = [img2]

    triplets = []
    pair_number = 10000
    for image in images.keys():
        if len(images[image]) < 2:
            continue
        
        for subset in itertools.combinations(images[image], 2):
            triplets.append((image, subset[0], subset[1]))
            if len(triplets) >= pair_number:
                break
        if len(triplets) >= pair_number:
            break
    
    bf = cv2.BFMatcher()
    epsilon = 1
    crossCheck = False
    database = h5py.File(output_path, 'w')
    success = 0

    with open(output_path_list, "w") as list_file:
        for i in tqdm(range(len(triplets))):
            triplet = triplets[i]
            img1 = triplet[0]
            img2 = triplet[1]
            img3 = triplet[2]
            label = f"{img1}-{img2}-{img3}"
            label12 = f"{img1}-{img2}"
            label13 = f"{img1}-{img3}"

            if label in database.keys():
                continue
            
            # Detect features in the 1st image
            if f"{img1}-feat" in database.keys():
                feat1 = np.array(database[f"{img1}-feat"])
                desc1 = np.array(database[f"{img1}-desc"])
            else:
                image1 = cv2.imread(os.path.join(data_path, scene, "images", img1))
                feat1, desc1 = detect_sift(image1, feature_number=5000)
                database.create_dataset(f"{img1}-feat", data=feat1)
                database.create_dataset(f"{img1}-desc", data=desc1)
            
            # Detect features in the 2nd image
            if f"{img2}-feat" in database.keys():
                feat2 = np.array(database[f"{img2}-feat"]) 
                desc2 = np.array(database[f"{img2}-desc"]) 
            else:
                image2 = cv2.imread(os.path.join(data_path, scene, "images", img2))
                feat2, desc2 = detect_sift(image2, feature_number=5000)
                database.create_dataset(f"{img2}-feat", data=feat2)
                database.create_dataset(f"{img2}-desc", data=desc2)
            
            # Detect features in the 3rd image
            if f"{img3}-feat" in database.keys():
                feat3 = np.array(database[f"{img3}-feat"])
                desc3 = np.array(database[f"{img3}-desc"]) 
            else:
                image3 = cv2.imread(os.path.join(data_path, scene, "images", img3))
                feat3, desc3 = detect_sift(image3, feature_number=5000)
                database.create_dataset(f"{img3}-feat", data=feat3)
                database.create_dataset(f"{img3}-desc", data=desc3)

            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
            if crossCheck:
                matches12 = bf.knnMatch(desc1, desc2, k=1)
                matches13 = bf.knnMatch(desc1, desc3, k=1)
                matches23 = bf.knnMatch(desc2, desc3, k=1)
            else:
                matches12 = bf.knnMatch(desc1, desc2, k=2)
                matches13 = bf.knnMatch(desc1, desc3, k=2)
                matches23 = bf.knnMatch(desc2, desc3, k=2)
            triplet_correspondences = []
            pair_correspondences12 = []
            pair_correspondences13 = []
            for i in range(len(matches12)):
                if crossCheck:
                    try:
                        m12 = matches12[i][0]
                        queryIdx12 = m12.queryIdx # In image 1
                        trainIdx12 = m12.trainIdx # In image 2
                        
                        m13 = matches13[i][0]
                        queryIdx13 = m13.queryIdx # In image 1
                        trainIdx13 = m13.trainIdx # In image 3
                        
                        m23 = matches23[trainIdx12][0]
                        queryIdx23 = m23.queryIdx # In image 2
                        trainIdx23 = m23.trainIdx # In image 3
                    except:
                        continue
                    
                    snn12 = 0
                    snn13 = 0
                    snn23 = 0
                else:
                    m12, n12 = matches12[i]
                    queryIdx12 = m12.queryIdx # In image 1
                    trainIdx12 = m12.trainIdx # In image 2
                    
                    m13, n13 = matches13[i]
                    queryIdx13 = m13.queryIdx # In image 1
                    trainIdx13 = m13.trainIdx # In image 3
                    
                    m23, n23 = matches23[trainIdx12]
                    queryIdx23 = m23.queryIdx # In image 2
                    trainIdx23 = m23.trainIdx # In image 3
                
                    snn12 = m12.distance / n12.distance
                    snn13 = m13.distance / n13.distance
                    snn23 = m23.distance / n23.distance
                
                # Cycle consistency check
                if trainIdx12 == queryIdx23 and trainIdx13 == trainIdx23:
                    kp1 = feat1[m12.queryIdx]
                    kp2 = feat2[m12.trainIdx]
                    kp3 = feat3[m13.trainIdx]
                    triplet_correspondences.append([kp1[0], kp1[1], kp2[0], kp2[1], kp3[0], kp3[1], snn12, snn13, snn23])
                    pair_correspondences12.append([kp1[0], kp1[1], kp2[0], kp2[1], snn12])
                    pair_correspondences13.append([kp1[0], kp1[1], kp3[0], kp3[1], snn13])
            triplet_correspondences = np.array(triplet_correspondences)
            pair_correspondences12 = np.array(pair_correspondences12)
            pair_correspondences13 = np.array(pair_correspondences13)

            database.create_dataset(label, data=triplet_correspondences)
            if label12 not in database.keys():
                database.create_dataset(label12, data=pair_correspondences12)
            if label13 not in database.keys():   
                database.create_dataset(label13, data=pair_correspondences13)
            list_file.write(f"{img1} {img2} {img3}\n")
            
            success += 1
            if success >= 1000:
                break
    database.close()

def ETH3D_pair(data_path, scene, output_path, output_path_list):
    # Loading data
    cameras = open(f'{data_path}/{scene}/dslr_calibration_undistorted/cameras.txt', 'r')
    cam_data = None
    # cam_data: (no.cam) X [width, height, focal_x, focal_y, cx, cy]
    while(True):
        line = cameras.readline()
        if(line == ''):
            break
        if(line[:-3] == '# Number of cameras:'):
            cam_data = np.zeros((int(line[-2]), 6), dtype=np.float32)
        if(line[0] == '#'):
            continue
        if(cam_data is None):
            break
        tok = line.split(' ')
        cam_data[int(tok[0])] = np.array([float(tok[2]), float(tok[3]), float(tok[4]), float(tok[5]), float(tok[6]), float(tok[7])])

    # image_data: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    gt_matrices = {}
    image_data = open(f'{data_path}/{scene}/dslr_calibration_undistorted/images.txt', 'r')
    while(True):
        line = image_data.readline()
        tmp_tok = re.split(', |:',line)
        if(line == ''):
            break
        if(line[0] == '#'):
            continue
        tok = re.split(' |\n',line)
        # Quaternion stored as scalar-last (X, Y, Z, W)
        gt_matrices[tok[-2]] = {'q': [float(tok[2]), float(tok[3]), float(tok[4]), float(tok[1])], 
                                't': [float(tok[5]), float(tok[6]), float(tok[7])], 
                                'cam': int(tok[8])}
        line = image_data.readline()
        index3d = np.array([float(x) for x in re.split(' |\n', line)[:-1]]).reshape(-1, 3)
        gt_matrices[tok[-2]]['index3d'] = set(index3d[:, 2])

    # Write cam matrices on h5 file
    image_names = gt_matrices.keys()
    
    camera_params = h5py.File(f"/datagrid/personal/tzamocha/datasets/ETH3D/{scene}/parameters.h5", 'w')
    for image in image_names:
        # camera index: gt_matrices[image]['cam']
        r = R.from_quat(gt_matrices[image]['q'])
        camera_params.create_dataset(f"{image}-R", data=r.as_matrix())
        camera_params.create_dataset(f"{image}-T", data=np.array(gt_matrices[image]['t']))
        intrinsic = cam_data[gt_matrices[image]['cam']]
        K = np.array([[intrinsic[2], 0, intrinsic[4]], [0, intrinsic[3], intrinsic[5]], [0, 0, 1]])
        camera_params.create_dataset(f"{image}-K", data=K)
        camera_params.create_dataset(f"{image}-width", data=intrinsic[0])
        camera_params.create_dataset(f"{image}-height", data=intrinsic[1])
    camera_params.close()
    
    images = {}
    for img1 in image_names:
        for img2 in image_names:
            pointsInCommon = len(gt_matrices[img1]['index3d'].intersection(gt_matrices[img2]['index3d']))
            if(img1 == img2):
                continue
            if(pointsInCommon < 500):
                print(f"Skipping {img1} {img2}: {pointsInCommon} 3D points in common.")
                continue
            if(img1 in images.keys()):
                images[img1].append(img2)
            else:
                images[img1] = [img2]

    pairs = []
    pair_number = 10000
    for image in images.keys():
        if len(images[image]) < 2:
            continue
        
        for subset in images[image]:
            pairs.append((image, subset))
            if len(pairs) >= pair_number:
                break
        if len(pairs) >= pair_number:
            break
    
    bf = cv2.BFMatcher()
    epsilon = 1
    crossCheck = False
    database = h5py.File(output_path, 'w')
    success = 0

    with open(output_path_list, "w") as list_file:
        for i in tqdm(range(len(pairs))):
            pair = pairs[i]
            img1 = pair[0]
            img2 = pair[1]
            label = f"{img1}-{img2}"

            if label in database.keys():
                continue
            
            # Detect features in the 1st image
            if f"{img1}-feat" in database.keys():
                feat1 = np.array(database[f"{img1}-feat"])
                desc1 = np.array(database[f"{img1}-desc"])
            else:
                image1 = cv2.imread(os.path.join(data_path, scene, "images", img1))
                feat1, desc1 = detect_sift(image1, feature_number=5000)
                database.create_dataset(f"{img1}-feat", data=feat1)
                database.create_dataset(f"{img1}-desc", data=desc1)
            
            # Detect features in the 2nd image
            if f"{img2}-feat" in database.keys():
                feat2 = np.array(database[f"{img2}-feat"]) 
                desc2 = np.array(database[f"{img2}-desc"]) 
            else:
                image2 = cv2.imread(os.path.join(data_path, scene, "images", img2))
                feat2, desc2 = detect_sift(image2, feature_number=5000)
                database.create_dataset(f"{img2}-feat", data=feat2)
                database.create_dataset(f"{img2}-desc", data=desc2)

            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
            if crossCheck:
                matches12 = bf.knnMatch(desc1, desc2, k=1)
            else:
                matches12 = bf.knnMatch(desc1, desc2, k=2)
            pair_correspondences = []

            for i in range(len(matches12)):
                if crossCheck:
                    try:
                        m12 = matches12[i][0]
                        queryIdx12 = m12.queryIdx # In image 1
                        trainIdx12 = m12.trainIdx # In image 2
                    except:
                        continue
                    snn12 = 0
                else:
                    m12, n12 = matches12[i]
                    queryIdx12 = m12.queryIdx # In image 1
                    trainIdx12 = m12.trainIdx # In image 2
                
                    snn12 = m12.distance / n12.distance
                

                kp1 = feat1[m12.queryIdx]
                kp2 = feat2[m12.trainIdx]
                pair_correspondences.append([kp1[0], kp1[1], kp2[0], kp2[1], snn12])
            pair_correspondences = np.array(pair_correspondences)
            
            database.create_dataset(label, data=pair_correspondences)
            list_file.write(f"{img1} {img2}\n")
            
            success += 1
            if success >= 1000:
                break
    database.close()
            
def generateParamsDistortion(data_path, scene):
    # Loading data
    cameras = open(f'{data_path}/{scene}/dslr_calibration_jpg/cameras.txt', 'r')\
    
    # cam_data: (no.cam) X [width, height, focal_x, focal_y, cx, cy]
    cam_data = None
    
    # rd_data: (no.cam) X [k1 k2 p1 p2 k3 k4 sx1 sy1]
    rd_data = None
    while(True):
        line = cameras.readline()
        if(line == ''):
            break
        if(line[:-3] == '# Number of cameras:'):
            cam_data = np.zeros((int(line[-2]), 6), dtype=np.float32)
            rd_data = np.zeros((int(line[-2]), 8), dtype=np.float32)
        if(line[0] == '#'):
            continue
        if(cam_data is None):
            break
        tok = line.split(' ')
        # example from courtyard scene: 
        # 0 THIN_PRISM_FISHEYE 6048 4032 3411.42 3410.02 3041.29 2014.07 0.21047 0.21102 -5.36231e-06 0.00051541 -0.158023 0.406856 -8.46499e-05 0.000861313
        # CAMERA_ID MODEL WIDTH HEIGHT Fx Fy Cx Cy k1 k2 p1 p2 k3 k4 sx1 sy1
        cam_data[int(tok[0])] = np.array([float(tok[2]), float(tok[3]), float(tok[4]), float(tok[5]), float(tok[6]), float(tok[7])])
        rd_data[int(tok[0])] = np.array([float(tok[8]), float(tok[9]), float(tok[10]), float(tok[11]), float(tok[12]), float(tok[13]), float(tok[14]), float(tok[15])])

        #print(f"Camera {int(tok[0])} has parameters: {cam_data[int(tok[0])]} and distortion: {rd_data[int(tok[0])]}")

    # image_data: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    gt_matrices = {}
    image_data = open(f'{data_path}/{scene}/dslr_calibration_jpg/images.txt', 'r')
    while(True):
        line = image_data.readline()
        tmp_tok = re.split(', |:',line)
        if(line == ''):
            break
        if(line[0] == '#'):
            continue
        tok = re.split(' |\n',line)
        # Quaternion stored as scalar-last (X, Y, Z, W)
        gt_matrices[tok[-2]] = {'q': [float(tok[2]), float(tok[3]), float(tok[4]), float(tok[1])], 
                                't': [float(tok[5]), float(tok[6]), float(tok[7])], 
                                'cam': int(tok[8])}
        line = image_data.readline()
        index3d = np.array([float(x) for x in re.split(' |\n', line)[:-1]]).reshape(-1, 3)
        gt_matrices[tok[-2]]['index3d'] = set(index3d[:, 2])

    # Write cam matrices on h5 file
    image_names = gt_matrices.keys()
    
    camera_params = h5py.File(f"/datagrid/personal/tzamocha/datasets/ETH3D/{scene}/parametersDistorted.h5", 'w')
    for image in image_names:
        # camera index: gt_matrices[image]['cam']
        r = R.from_quat(gt_matrices[image]['q'])
        intrinsic = cam_data[gt_matrices[image]['cam']]
        K = np.array([[intrinsic[2], 0, intrinsic[4]], [0, intrinsic[3], intrinsic[5]], [0, 0, 1]])
        rdParams = rd_data[gt_matrices[image]['cam']]

        camera_params.create_dataset(f"{image}-R", data=r.as_matrix())
        camera_params.create_dataset(f"{image}-T", data=np.array(gt_matrices[image]['t']))
        camera_params.create_dataset(f"{image}-D", data=rdParams)
        camera_params.create_dataset(f"{image}-K", data=K)
        camera_params.create_dataset(f"{image}-width", data=intrinsic[0])
        camera_params.create_dataset(f"{image}-height", data=intrinsic[1])
    camera_params.close()

if __name__ == "__main__":
    # for photoTourism
    data_path = "/datagrid/personal/tzamocha/datasets/RANSAC-Tutorial-Data-ValOnly/val"
    scene = "trevi_fountain"
    output_path = f"/datagrid/personal/tzamocha/datasets/RANSAC-Tutorial-Data-ValOnly/val/{scene}/SIFT_triplet_correspondences.h5"
    output_path_list = f"/datagrid/personal/tzamocha/datasets/RANSAC-Tutorial-Data-ValOnly/val/{scene}/SIFT_triplet_correspondences_list.txt"

    PhotoTourism(data_path, scene, output_path, output_path_list)

    # for ETH3D
    #data_path = "/datagrid/personal/tzamocha/datasets/ETH3D"
    #scene = "bridge"
    #output_path = f"/datagrid/personal/tzamocha/datasets/ETH3D/{scene}/SIFT_triplet_correspondences.h5"
    #output_path_list = f"/datagrid/personal/tzamocha/datasets/ETH3D/{scene}/SIFT_triplet_correspondences_list.txt"

    #ETH3D(data_path, scene, output_path, output_path_list)

    # for distortion on ETH3D
    data_path = "/datagrid/personal/tzamocha/datasets/ETH3D"
    scene = "terrains"

    generateParamsDistortion(data_path, scene)