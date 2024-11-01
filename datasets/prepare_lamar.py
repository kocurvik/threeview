import argparse
import itertools
import ntpath
import os
from pathlib import Path
import h5py
import numpy as np
import torch
from tqdm import tqdm
import lightglue
from lightglue.utils import load_image

from utils.matching import get_matcher_string, get_extractor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_samples', type=int, default=None)
    parser.add_argument('-s', '--step', type=int, default=10)
    parser.add_argument('-f', '--features', type=str, default='superpoint')
    parser.add_argument('-mf', '--max_features', type=int, default=2048)
    parser.add_argument('-r', '--resize', type=int, default=None)
    parser.add_argument('--recalc', action='store_true', default=False)
    parser.add_argument('out_path')
    parser.add_argument('dataset_path')

    return parser.parse_args()


def qvec2rotmat(qvec):
    """ Convert from quaternions to rotation matrix. """
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

calib = {
    'hetlf': np.array([[363.32, 0, 235.031], [0, 360.299, 288.286], [0, 0, 1]]),
    'hetrf': np.array([[364.494, 0, 242.239], [0, 362.158, 297.774], [0, 0, 1]]),
    'hetrr': np.array([[365.116, 0, 245.76], [0, 363.651, 299.386], [0, 0, 1]]),
    'hetll': np.array([[363.605, 0, 246.758], [0, 362.364, 303.493], [0, 0, 1]]),
}


def load_lamar_and_extract_intrinsics(root_dir, out_dir):
    exist = [os.path.exists(os.path.join(out_dir, f'{x}.h5')) for x in ['K', 'R', 'T', 'parameters_rd']]
    if not False in exist and not args.recalc:
        print(f"GT info exists in {out_dir} - not creating it anew")
        save_intrinsics = False
    else:
        print(f"Writing GT info to {out_dir}")
        fK = h5py.File(os.path.join(out_dir, 'K.h5'), 'w')
        fR = h5py.File(os.path.join(out_dir, 'R.h5'), 'w')
        fT = h5py.File(os.path.join(out_dir, 'T.h5'), 'w')
        fH = h5py.File(os.path.join(out_dir, 'parameters_rd.h5'), 'w')
        save_intrinsics = True

    img2rig_id = {}
    with open(os.path.join(root_dir, 'images.txt'), 'r') as f:
        lines = [s.strip('\n').split(', ') for s in f.readlines()][1:]
        for l in lines:
            img2rig_id[l[2][-13:]] = l[0]

    # Extract the GT transformation from world to rig
    T_w_rig = {}
    with open(os.path.join(root_dir, 'proc', 'alignment_trajectories.txt'), 'r') as f:
        lines = [s.strip('\n').split(', ') for s in f.readlines()][1:]
        for l in lines:
            R_rig_w = qvec2rotmat(
                np.array(l[2:6]).astype(np.float32))
            t_rig_w = np.array(l[6:9], dtype=np.float32)
            T_w_rig[l[0]] = np.eye(4)
            T_w_rig[l[0]][:3, :3] = R_rig_w.T
            T_w_rig[l[0]][:3, 3] = -R_rig_w.T @ t_rig_w

    # Extract the transformation from rig to cam
    T_rig_cam = {}
    with open(os.path.join(root_dir, 'rigs.txt'), 'r') as f:
        lines = [s.strip('\n').split(', ') for s in f.readlines()][1:]
        for l in lines:
            rig_id = l[0][-9:]
            if rig_id not in T_rig_cam:
                T_rig_cam[rig_id] = {}
            R_cam_rig = qvec2rotmat(np.array(l[2:6]).astype(np.float32))
            t_cam_rig = np.array(l[6:9], dtype=np.float32)
            T_rig_cam[rig_id][l[1][-5:]] = np.eye(4)
            T_rig_cam[rig_id][l[1][-5:]][:3, :3] = R_cam_rig.T
            T_rig_cam[rig_id][l[1][-5:]][:3, 3] = -R_cam_rig.T @ t_cam_rig

    sequences = os.listdir(os.path.join(root_dir, 'raw_data'))

    seq_images = {}
    all_images = []

    for s in sequences:
        seq_path = os.path.join(root_dir, 'raw_data', s)

        seq_images[s] = {}

        # Extract the images
        for cam in ['hetlf', 'hetll', 'hetrf', 'hetrr']:
            seq_images[s][cam] = []
            img_folder = os.path.join(seq_path, 'images', cam)
            images = os.listdir(img_folder)
            images.sort()
            for i in range(len(images)):
                image_name = ntpath.normpath(f'raw_data/{s}/images/{cam}/{images[i]}')
                all_images.append(image_name)
                seq_images[s][cam].append(image_name)
                rig_id = img2rig_id[images[i]]

                if save_intrinsics:
                    T = T_rig_cam[rig_id][cam] @ T_w_rig[rig_id]
                    R = T[:3, :3]
                    t = T[:3, 3]
                    K = calib[cam]

                    fR.create_dataset(image_name, shape=(3, 3), data=R)
                    fT.create_dataset(image_name, shape=(3, 1), data=t.reshape(3, 1))
                    fK.create_dataset(image_name, shape=(3, 3), data=K)

    return seq_images, all_images

def extract_features(all_images, session_path, out_dir, args):
    # extractor = lightglue.SuperPoint(max_num_keypoints=2048).eval().cuda()
    extractor = get_extractor(args)
    out_path = os.path.join(out_dir, f"{get_matcher_string(args)}.pt")

    if os.path.exists(out_path) and not args.recalc:
        print(f"Features already found in {out_path}")
        return

    print("Extracting features")

    feature_dict = {}
    for image_name in tqdm(all_images):
        path = os.path.join(session_path, image_name)
        image_tensor = load_image(path).cuda()

        kp_tensor = extractor.extract(image_tensor, resize=args.resize)
        feature_dict[image_name] = kp_tensor

    torch.save(feature_dict, out_path)
    print("Features saved to: ", out_path)

def create_pairs(seq_images, out_dir, args):
    step = args.step
    features = torch.load(os.path.join(out_dir, f"{get_matcher_string(args)}.pt"))

    matcher = lightglue.LightGlue(features=args.features).eval().cuda()
    h5_path = os.path.join(out_dir, f'pairs-{step}steps-{get_matcher_string(args)}-LG.h5')
    h5_file = h5py.File(h5_path, 'w')

    triplets = []

    print("Writing matches to: ", h5_path)

    for seq in seq_images.keys():
        cams = list(seq_images[seq].keys())
        for cam in cams:
            for k in tqdm(range(len(seq_images[seq][cam]) - step)):
                name_1 = seq_images[seq][cam][k]
                name_2 = seq_images[seq][cam][k + step]

                label = f'{name_1}-{name_2}'

                if label in h5_file:
                    continue

                feats_1 = features[name_1]
                feats_2 = features[name_2]

                out_12 = matcher({'image0': feats_1, 'image1': feats_2})
                scores_12 = out_12['matching_scores0'][0].detach().cpu().numpy()
                matches_12 = out_12['matches0'][0].detach().cpu().numpy()

                idxs = []

                for idx_1, idx_2 in enumerate(matches_12):
                    if idx_2 != -1:
                        idxs.append((idx_1, idx_2))

                if len(idxs) < 5:
                    continue

                out_array = np.empty([len(idxs), 5])

                for i, idx in enumerate(idxs):
                    idx_1, idx_2 = idx
                    point_1 = feats_1['keypoints'][0, idx_1].detach().cpu().numpy()
                    point_2 = feats_2['keypoints'][0, idx_2].detach().cpu().numpy()
                    score_12 = scores_12[idx_1]
                    out_array[i] = np.array([*point_1, *point_2, score_12])

                h5_file.create_dataset(label, shape=out_array.shape, data=out_array)
                triplets.append(f'{name_1} {name_2}')

    pairs_txt_path = os.path.join(out_dir, f'pairs-{step}steps-{get_matcher_string(args)}-LG.txt')
    print("Writing list of pairs to: ", pairs_txt_path)
    with open(pairs_txt_path, 'w') as f:
        f.writelines(line + '\n' for line in triplets)


def create_pairs_intercam(seq_images, out_dir, args):
    step = args.step
    features = torch.load(os.path.join(out_dir, f"{get_matcher_string(args)}.pt"))

    matcher = lightglue.LightGlue(features=args.features).eval().cuda()
    h5_path = os.path.join(out_dir, f'pairs-intercam-{get_matcher_string(args)}-LG.h5')
    h5_file = h5py.File(h5_path, 'w')

    triplets = []

    print("Writing matches to: ", h5_path)

    for seq in seq_images.keys():
        cams = list(seq_images[seq].keys())
        max_idx = min([len(seq_images[seq][cam]) for cam in cams])

        for k in tqdm(range(max_idx)):
            for cam1, cam2 in itertools.combinations(cams, 2):
                name_1 = seq_images[seq][cam1][k]
                name_2 = seq_images[seq][cam2][k]

                label = f'{name_1}-{name_2}'

                if label in h5_file:
                    continue

                feats_1 = features[name_1]
                feats_2 = features[name_2]

                out_12 = matcher({'image0': feats_1, 'image1': feats_2})
                scores_12 = out_12['matching_scores0'][0].detach().cpu().numpy()
                matches_12 = out_12['matches0'][0].detach().cpu().numpy()

                idxs = []

                for idx_1, idx_2 in enumerate(matches_12):
                    if idx_2 != -1:
                        idxs.append((idx_1, idx_2))

                if len(idxs) < 5:
                    continue

                out_array = np.empty([len(idxs), 5])

                for i, idx in enumerate(idxs):
                    idx_1, idx_2 = idx
                    point_1 = feats_1['keypoints'][0, idx_1].detach().cpu().numpy()
                    point_2 = feats_2['keypoints'][0, idx_2].detach().cpu().numpy()
                    score_12 = scores_12[idx_1]
                    out_array[i] = np.array([*point_1, *point_2, score_12])

                h5_file.create_dataset(label, shape=out_array.shape, data=out_array)
                triplets.append(f'{name_1} {name_2}')

    pairs_txt_path = os.path.join(out_dir, f'pairs-intercam-{get_matcher_string(args)}-LG.txt')
    print("Writing list of pairs to: ", pairs_txt_path)
    with open(pairs_txt_path, 'w') as f:
        f.writelines(line + '\n' for line in triplets)


def prepare_single(args, session):
    out_dir = os.path.join(args.out_path, session)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    session_path = os.path.join(args.dataset_path, session)
    seq_images, all_images = load_lamar_and_extract_intrinsics(session_path, out_dir)
    extract_features(all_images, session_path, out_dir, args)
    create_pairs(seq_images, out_dir, args)
    # create_pairs_intercam(seq_images, out_dir, args)

def run_im(args):
    # sessions = [x for x in os.listdir(args.dataset_path) if 'val' in x]
    sessions = ['query_val_hololens']

    for s in sessions:
        prepare_single(args, s)

if __name__ == '__main__':
    args = parse_args()
    run_im(args)