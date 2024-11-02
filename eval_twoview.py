import argparse
import json
import os
from multiprocessing import Pool
from time import perf_counter

import h5py
import numpy as np
import poselib
# import pykitti
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from tqdm import tqdm

from utils.data import err_twoview
from utils.geometry import rotation_angle, angle, get_camera_dicts, force_inliers_twoview
from utils.vis import draw_results_pose_auc_10


# from utils.vis import draw_results_pose_auc_10

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--first', type=int, default=None)
    parser.add_argument('-i', '--force_inliers', type=float, default=None)
    parser.add_argument('-t', '--threshold', type=float, default=1.0)
    parser.add_argument('-nw', '--num_workers', type=int, default=1)
    parser.add_argument('-l', '--load', action='store_true', default=False)
    parser.add_argument('-g', '--graph', action='store_true', default=False)
    parser.add_argument('-s', '--shuffle', type=float, default=0.0)
    parser.add_argument('-a', '--append', action='store_true', default=False)
    parser.add_argument('feature_file')
    parser.add_argument('dataset_path')

    return parser.parse_args()

# def get_pairs(file):
#     return [tuple(x.split('-')) for x in file.keys() if 'feat' not in x and 'desc' not in x]

def get_pairs(file):
    with open(file, 'r') as f:
        pairs = f.readlines()
    return [tuple(x.strip().split(' ')) for x in pairs]

def get_result_dict(info, pose_est, R_gt, t_gt):
    out = {}

    R_est, t_est = pose_est.R, pose_est.t

    out['R_err'] = rotation_angle(R_est.T @ R_gt)
    out['t_err'] = angle(t_est, t_gt)
    out['R'] = R_est.tolist()
    out['R_gt'] = R_gt.tolist()
    out['t'] = R_est.tolist()
    out['t_gt'] = R_gt.tolist()

    out['P_err'] = max(out['R_err'], out['t_err'])

    info['inliers'] = []
    out['info'] = info

    return out


def eval_experiment(x):
    iters, experiment, kp1, kp2, R_gt, t_gt, K1, K2, t = x

    sample_sz = int(experiment[0])

    if iters is None:
        ransac_dict = {'max_iterations': 5000, 'max_epipolar_error': t, 'progressive_sampling': False,
                       'min_iterations': 50, 'lo_iterations': 25, 'sample_sz': sample_sz}
    else:
        ransac_dict = {'max_iterations': iters, 'max_epipolar_error': t, 'progressive_sampling': False,
                       'min_iterations': iters, 'sample_sz': sample_sz}

    ransac_dict['use_homography'] = 'H' in experiment
    ransac_dict['use_affine'] = '(A)' in experiment
    ransac_dict['early_nonminimal'] = '+ ENM' in experiment
    ransac_dict['early_lm'] = '+ ELM' in experiment

    if 'D(' in experiment:
        idx = experiment.find('D(')
        idx_end = experiment[idx+2:].find(')')
        delta = float(experiment[idx+2:idx + 2 + idx_end])
    else:
        delta = 0.0

    ransac_dict['delta'] = delta

    # ransac_dict['nonminimal_refinement'] = 'nonminimal' in experiment
    ransac_dict['lo_iterations'] = 0 if 'nLO' in experiment else 25
    bundle_dict = {'max_iterations': 0 if 'nLO' in experiment else 100}

    camera1 = {'model': 'PINHOLE', 'width': -1, 'height': -1, 'params': [K1[0, 0], K1[1, 1], K1[0, 2], K1[1, 2]]}
    camera2 = {'model': 'PINHOLE', 'width': -1, 'height': -1, 'params': [K2[0, 0], K2[1, 1], K2[0, 2], K2[1, 2]]}

    start = perf_counter()
    pose_est, info = poselib.estimate_relative_pose(kp1, kp2, camera1, camera2, ransac_dict, bundle_dict)
    info['runtime'] = 1000 * (perf_counter() - start)

    result_dict = get_result_dict(info, pose_est, R_gt, t_gt)
    result_dict['experiment'] = experiment

    return result_dict


def print_results(experiments, results, eq_only=False):
    tab = PrettyTable(['solver', 'median pose err', 'mean pose err',
                       'Pose AUC@5', 'Pose AUC@10', 'Pose AUC@20',
                       'median time', 'mean time', 'median inliers', 'mean inliers'])
    tab.align["solver"] = "l"
    tab.float_format = '0.2'

    for exp in experiments:
        exp_results = [x for x in results if x['experiment'] == exp]

        p_errs = np.array([max(r['R_err'], r['t_err']) for r in exp_results])
        p_errs[np.isnan(p_errs)] = 180
        p_res = np.array([np.sum(p_errs < t) / len(p_errs) for t in range(1, 21)])

        times = np.array([x['info']['runtime'] for x in exp_results])
        inliers = np.array([x['info']['inlier_ratio'] for x in exp_results])

        exp_name = exp


        tab.add_row([exp_name, np.median(p_errs), np.mean(p_errs),
                     100*np.mean(p_res[:5]), 100*np.mean(p_res[:10]), 100*np.mean(p_res),
                     np.median(times), np.mean(times),
                     np.median(inliers), np.mean(inliers)])
    print(tab)

    print('latex')

    print(tab.get_formatted_string('latex'))

def draw_cumplots(experiments, results):
    plt.figure()
    plt.xlabel('Pose error')
    plt.ylabel('Portion of samples')

    for exp in experiments:
        exp_results = [x for x in results if x['experiment'] == exp]
        exp_name = exp
        label = f'{exp_name}'

        R_errs = np.array([max(r['R_err'], r['t_err']) for r in exp_results])
        R_res = np.array([np.sum(R_errs < t) / len(R_errs) for t in range(1, 180)])
        plt.plot(np.arange(1, 180), R_res, label = label)

    plt.legend()
    plt.show()

    plt.figure()
    plt.xlabel('k error')
    plt.ylabel('Portion of samples')


def get_im_generator(args, dataset_path):
    R_file = h5py.File(os.path.join(dataset_path, 'R.h5'))
    T_file = h5py.File(os.path.join(dataset_path, 'T.h5'))
    K_file = h5py.File(os.path.join(dataset_path, 'K.h5'))
    C_file = h5py.File(os.path.join(dataset_path, f'{args.feature_file}.h5'))
    # R_dict = {k: np.array(v) for k, v in R_file.items()}
    # t_dict = {k: np.array(v) for k, v in T_file.items()}
    # w_dict = {k.split('-')[0]: v[0, 0] for k, v in P_file.items()}
    # h_dict = {k.split('-')[0]: v[1, 1] for k, v in P_file.items()}
    camera_dicts = get_camera_dicts(os.path.join(dataset_path, 'K.h5'))
    pairs = get_pairs(os.path.join(dataset_path, f'{args.feature_file}.txt'))
    if args.first is not None:
        pairs = pairs[:args.first]

    def gen_data():
        for img_name_1, img_name_2 in pairs:
            R1 = np.array(R_file[img_name_1])
            t1 = np.array(T_file[img_name_1])
            R2 = np.array(R_file[img_name_2])
            t2 = np.array(T_file[img_name_2])
            K1 = np.array(K_file[img_name_1])
            K2 = np.array(K_file[img_name_2])

            R_gt = np.dot(R2, R1.T)
            t_gt = t2 - np.dot(R_gt, t1)

            matches = np.array(C_file[f'{img_name_1}-{img_name_2}'])

            kp1 = matches[:, :2]
            kp2 = matches[:, 2:4]

            if len(kp1) < 5:
                continue

            if args.shuffle > 0.0:
                kp1 = shuffle_portion(kp1, args.shuffle)

            if args.force_inliers is not None:
                kp1, kp2 = force_inliers_twoview(kp1, kp2, R_gt, t_gt, K1, K2,
                                                   args.force_inliers, args.threshold)

            yield np.copy(kp1), np.copy(kp2), R_gt, t_gt, K1, K2, args.threshold

    return gen_data, len(pairs)

def gen_wrapper(gen, experiments, iterations_list):
    for x in gen:
        for experiment in experiments:
            for iterations in iterations_list:
                yield iterations, experiment, *x

def get_kitti_generator(args, dataset_path, sequences='all'):
    if sequences == 'all':
        sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '08',
                     '09', '10']
    elif isinstance(sequences, str):
        sequences = [sequences]
    else:
        assert isinstance(sequences, list), "sequences must be a str or list."

    M_file = h5py.File(os.path.join(dataset_path, 'matches.h5'))
    feat = args.feature_file

    num_pairs = 0
    for s in sequences:
        data = pykitti.odometry(dataset_path, s)
        num_pairs += len(data.poses) - 1

    def gen_data(experiments, iterations_list):
        for s in sequences:
            data = pykitti.odometry(dataset_path, s)
            # num_imgs = len(data.cam0_files)
            # Load the image pairs

            calib_filepath = os.path.join(data.sequence_path, 'calib.txt')
            filedata = pykitti.utils.read_calib_file(calib_filepath)
            K = np.reshape(filedata['P0'], (3, 4))[:3, :3]

            # Load the poses
            for i in range(len(data.poses) - 1):
                T_1_w = data.poses[i]
                T_2_w = data.poses[i + 1]
                T_1_2 = np.linalg.inv(T_2_w) @ T_1_w
                t_gt = T_1_2[:3, 3]
                R_gt = T_1_2[:3, :3]

                label1 = "-".join(data.cam0_files[i].split("/")[-3:])
                label2 = "-".join(data.cam0_files[i+1].split("/")[-3:])
                label_kps = f"{feat.lower()}-{label1}-{label2}"
                label_score = f"{feat.lower()}-{label1}-{label2}-scores"

                kps = np.array(M_file[label_kps])
                s = np.array(M_file[label_score])
                kp1 = kps[:, :2]
                kp2 = kps[:, 2:]

                yield np.copy(kp1), np.copy(kp2), R_gt, t_gt, K, K

    return gen_data, num_pairs


import numpy as np


def shuffle_portion(kp: np.ndarray, s: float) -> np.ndarray:
    num_rows_to_shuffle = int(s * kp.shape[0])
    indices_to_shuffle = np.random.choice(kp.shape[0], num_rows_to_shuffle, replace=False)
    rows_to_shuffle = kp[indices_to_shuffle]
    np.random.shuffle(rows_to_shuffle)
    shuffled_kp = kp.copy()
    shuffled_kp[indices_to_shuffle] = rows_to_shuffle

    return shuffled_kp


def eval(args):
    # experiments = ['5pE', '5pE + ELM', '5pE + ENM',
    #                '4pE(M)', '4pE(M) + ELM', '4pE(M) + ENM',
    #                '4pF(A)', '3pH(A)', '2pE(A)',
    #                '4pF(A) + ENM', '3pH(A) + ENM', '2pE(A) + ENM',
    #                '4pH', '4pH + ENM', '4pH + ELM']

    experiments = ['5pE', '4pE(M)', '4pF(A)', '3pH(A)', '2pE(A)', '4pH']

    dataset_path = args.dataset_path
    basename = os.path.basename(dataset_path)
    # if 'kitti' in dataset_path.lower():
    #     basename = 'kitti'

    if args.force_inliers is not None:
        basename = f'{basename}-{args.force_inliers:.1f}inliers'

    if args.threshold != 1.0:
        basename = f'{basename}-{args.threshold}t'


    matches_basename = os.path.basename(args.feature_file)

    if args.graph:
        basename = f'{basename}-graph'
        # iterations_list = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
        iterations_list = [10, 20, 50, 100, 200, 500, 1000]
    else:
        iterations_list = [None]

    if args.shuffle == 0.0:
        json_string = f'twoview-{basename}-{matches_basename}.json'
    else:
        json_string = f'twoview-{basename}-{matches_basename}-{args.shuffle}s.json'


    if args.load:
        print("Loading: ", json_string)
        with open(os.path.join('results', json_string), 'r') as f:
            results = json.load(f)

    else:
        # if 'kitti' in dataset_path.lower():
        #     gen_data, num_pairs = get_kitti_generator(args, dataset_path)
        # else:
        gen_data, num_pairs = get_im_generator(args, dataset_path)

        total_length = len(experiments) * len(iterations_list) * num_pairs

        print(f"Total runs: {total_length} for {num_pairs} samples")

        if args.num_workers == 1:
            results = [eval_experiment(x) for x in tqdm(gen_wrapper(gen_data(), experiments, iterations_list), total=total_length)]
        else:
            pool = Pool(args.num_workers)
            results = [x for x in pool.imap(eval_experiment, tqdm(gen_wrapper(gen_data(), experiments, iterations_list), total=total_length))]

        os.makedirs('results', exist_ok=True)

        if args.append:
            print(f"Appending from: {os.path.join('results', json_string)}")
            with open(os.path.join('results', json_string), 'r') as f:
                prev_results = json.load(f)
            results.extend(prev_results)

        with open(os.path.join('results', json_string), 'w') as f:
            json.dump(results, f)

        print("Done")

    print_results(experiments, results)
    # draw_cumplots(experiments, results)

    if args.graph:
        draw_results_pose_auc_10(results, experiments, iterations_list, err_fun=err_twoview)


if __name__ == '__main__':
    args = parse_args()
    eval(args)