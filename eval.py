import argparse
import json
import os
from multiprocessing import Pool
from time import perf_counter

import poselib
import h5py
import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from tqdm import tqdm

from theory.lo_verification import skew
from utils.geometry import rotation_angle, angle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--first', type=int, default=None)
    parser.add_argument('-i', '--force_inliers', type=float, default=None)
    parser.add_argument('-nw', '--num_workers', type=int, default=1)
    parser.add_argument('-l', '--load', action='store_true', default=False)
    parser.add_argument('-g', '--graph', action='store_true', default=False)
    parser.add_argument('feature_file')
    parser.add_argument('dataset_path')

    return parser.parse_args()



def get_pose(img1, img2, R_dict, T_dict):
    R1 = np.array(R_dict[img1])
    R2 = np.array(R_dict[img2])
    t1 = np.array(T_dict[img1])
    t2 = np.array(T_dict[img2])

    R = R1.T @ R2
    t = -R @ t1 + t2
    return R, t


def get_camera_dicts(K_file_path):
    K_file = h5py.File(K_file_path)

    d = {}

    # Treat data from Charalambos differently since it is in pairs
    if 'K1_K2' in K_file_path:

        for k, v in K_file.items():
            key1, key2 = k.split('-')
            if key1 not in d.keys():
                K1 = np.array(v)[0, 0]
                d[key1] = {'model': 'SIMPLE_PINHOLE', 'width': int(2 * K1[0, 2]), 'height': int(2 * K1[1,2]), 'params': [K1[0, 0], K1[0, 2], K1[1, 2]]}
            if key2 not in d.keys():
                K2 = np.array(v)[0, 1]
                d[key2] = {'model': 'SIMPLE_PINHOLE', 'width': int(2 * K2[0, 2]), 'height': int(2 * K2[1,2]), 'params': [K2[0, 0], K2[0, 2], K2[1, 2]]}

        return d

    for key, v in K_file.items():
        K = np.array(v)
        d[key] = {'model': 'PINHOLE', 'width': int(2 * K[0, 2]), 'height': int(2 * K[1,2]), 'params': [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]}

    return d

def get_triplets(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    return [line.strip().split(' ') for line in lines]


def get_result_dict(three_view_pose, info, img1, img2, img3, R_file, T_file):
    gt_R_12, gt_t_12 = get_pose(img1, img2, R_file, T_file)
    gt_R_13, gt_t_13 = get_pose(img1, img3, R_file, T_file)
    gt_R_23, gt_t_23 = get_pose(img2, img3, R_file, T_file)

    R_12, t_12 = three_view_pose.pose12.R, three_view_pose.pose12.t
    R_13, t_13 = three_view_pose.pose13.R, three_view_pose.pose13.t
    R_23, t_23 = three_view_pose.pose23().R, three_view_pose.pose23().t

    out = {}
    out['R_12_err'] = rotation_angle(R_12.T @ gt_R_12)
    out['R_13_err'] = rotation_angle(R_13.T @ gt_R_13)
    out['R_23_err'] = rotation_angle(R_23.T @ gt_R_23)

    out['t_12_err'] = angle(t_12, gt_t_12)
    out['t_13_err'] = angle(t_13, gt_t_13)
    out['t_23_err'] = angle(t_23, gt_t_23)

    out['P_12_err'] = max(out['R_12_err'], out['t_12_err'])
    out['P_13_err'] = max(out['R_13_err'], out['t_13_err'])
    out['P_23_err'] = max(out['R_23_err'], out['t_23_err'])

    out['P_err'] = max([v for k, v in out.items()])
    out['info'] = info
    return out


def print_results(results):
    tab = PrettyTable(['metric', 'median', 'mean', 'AUC@5', 'AUC@10', 'AUC@20'])
    tab.align["metric"] = "l"
    tab.float_format = '0.2'
    err_names = ['P_12_err', 'P_13_err', 'P_23_err', 'P_err']
    for err_name in err_names:
        errs = np.array([r[err_name] for r in results])
        errs[np.isnan(errs)] = 180
        res = np.array([np.sum(errs < t) / len(errs) for t in range(1, 21)])
        tab.add_row([err_name, np.median(errs), np.mean(errs), np.mean(res[:5]), np.mean(res[:10]), np.mean(res)])

        # print(f'{err_name}: \t median: {np.median(errs):0.2f} \t mean: {np.mean(errs):0.2f} \t '
        #       f'auc5: {np.mean(res[:5]):0.2f} \t auc10: {np.mean(res[:10]):0.2f} \t auc20: {np.mean(res):0.2f}')

    for field in ['inlier_ratio', 'iterations', 'runtime', 'refinements']:
        xs = [r['info'][field] for r in results]
        tab.add_row([field, np.median(xs), np.mean(xs), '-', '-', '-'])
        # print(f'{field}: \t median: {np.median(xs):0.02f} \t mean: {np.mean(xs):0.02f}')

    print(tab)


def get_gt_F(img1, img2, R_dict, T_dict, camera_dicts):
    R1, t1 = np.array(R_dict[img1]), np.array(T_dict[img1])
    R2, t2 = np.array(R_dict[img2]), np.array(T_dict[img2])
    R = R2 @ R1.T
    t = (t2 - R @ t1).ravel()
    cam1 = camera_dicts[img1]
    cam2 = camera_dicts[img2]
    K1 = np.array([[cam1['params'][0], 0.0, cam1['params'][-2]], [0.0, cam1['params'][0], cam1['params'][-1]], [0, 0, 1.0]])
    K2 = np.array([[cam2['params'][0], 0.0, cam2['params'][-2]], [0.0, cam2['params'][0], cam2['params'][-1]], [0, 0, 1.0]])

    return np.linalg.inv(K2.T) @ skew(t) @ R @ np.linalg.inv(K1)


def get_inliers(F, x1, x2, threshold = 2.0):
    pts1 = np.column_stack([x1, np.ones(len(x1))])
    pts2 = np.column_stack([x2, np.ones(len(x2))])
    F_t = F.T
    line1_in_2 = pts1 @ F_t
    line2_in_1 = pts2 @ F

    # numerator = (x'^T F x) ** 2
    numerator = np.sum(pts2 * line1_in_2, axis=1) ** 2

    # denominator = (((Fx)_1**2) + (Fx)_2**2)) +  (((F^Tx')_1**2) + (F^Tx')_2**2))
    denominator = line1_in_2[:, 0] ** 2 + line1_in_2[:, 1] ** 2 + line2_in_1[:, 0] ** 2 + line2_in_1[:, 1] ** 2
    out = numerator / denominator
    return out < threshold ** 2


def add_rand_pts(x, cam_dict, multiplier):
    x_new = np.random.rand(int(multiplier * len(x)), 2)
    x_new[:, 0] *= cam_dict['width']
    x_new[:, 1] *= cam_dict['height']
    return np.row_stack([x, x_new])

def force_inliers(x1, x2, x3, img1, img2, img3, R_dict, T_dict, camera_dicts, ratio):
    F12 = get_gt_F(img1, img2, R_dict, T_dict, camera_dicts)
    F13 = get_gt_F(img1, img3, R_dict, T_dict, camera_dicts)
    F23 = get_gt_F(img2, img3, R_dict, T_dict, camera_dicts)

    inliers_12 = get_inliers(F12, x1, x2, 2.0)
    inliers_13 = get_inliers(F13, x1, x3, 2.0)
    inliers_23 = get_inliers(F23, x2, x3, 2.0)

    l = np.logical_and(np.logical_and(inliers_12, inliers_13), inliers_23)

    # print(np.sum(inliers_12), np.sum(inliers_13), np.sum(inliers_23), np.sum(l), len(x1))

    multiplier = (1 - ratio) / ratio

    x1, x2, x3 = x1[l], x2[l], x3[l]

    x1 = add_rand_pts(x1, camera_dicts[img1], multiplier)
    x2 = add_rand_pts(x2, camera_dicts[img2], multiplier)
    x3 = add_rand_pts(x3, camera_dicts[img3], multiplier)

    return x1, x2, x3

def eval_experiment(x):
    experiment, iterations, img1, img2, img3, x1, x2, x3, R_dict, T_dict, camera_dicts = x

    inner_refine = 100 if 'R' in experiment else 0
    delta = 0.1 if 'D' in experiment else 0.0
    num_pts = int(experiment[0])
    if iterations is not None:
        ransac_dict = {'max_epipolar_error': 2.0, 'progressive_sampling': False,
                       'min_iterations': iterations,'max_iterations': iterations, 'lo_iterations': 25 if 'LO' in experiment else 0,
                       'inner_refine': inner_refine, 'threeview_check': 'C' in experiment, 'sample_sz': num_pts, 'delta': delta}
    else:
        ransac_dict = {'max_epipolar_error': 2.0, 'progressive_sampling': False,
                       'min_iterations': 100,'max_iterations': 10000, 'lo_iterations': 25 if 'LO' in experiment else 0,
                       'inner_refine': inner_refine, 'threeview_check': 'C' in experiment, 'sample_sz': num_pts, 'delta': delta}

    bundle_dict = {'verbose': False, 'max_iterations': 100 if 'LO' in experiment else 0}
    start = perf_counter()
    three_view_pose, info = poselib.estimate_three_view_relative_pose(x1, x2, x3, camera_dicts[img1], camera_dicts[img2], camera_dicts[img3], ransac_dict, bundle_dict)
    info['runtime'] = 1000 * (perf_counter() - start)
    result_dict = get_result_dict(three_view_pose, info, img1, img2, img3, R_dict, T_dict)
    result_dict['experiment'] = experiment
    result_dict['img1'] = img1
    result_dict['img2'] = img2
    result_dict['img3'] = img3

    # with open(f'results/{experiment}-{img1}-{img2}-{img3}.json', 'w') as f:
    #     json.dump(result_dict, f)

    return result_dict


def draw_results(results, experiments, iterations_list):
    plt.figure()

    for experiment in experiments:
        experiment_results = [x for x in results if x['experiment'] == experiment]

        xs = []
        ys = []

        for iterations in iterations_list:
            iter_results = [x for x in experiment_results if x['info']['iterations'] == iterations]
            mean_runtime = np.mean([x['info']['runtime'] for x in iter_results])
            errs = np.array([r['P_err'] for r in iter_results])
            errs[np.isnan(errs)] = 180
            AUC10 = np.mean(np.array([np.sum(errs < t) / len(errs) for t in range(1, 11)]))

            xs.append(mean_runtime)
            ys.append(AUC10)

        plt.semilogx(xs, ys, label=experiment, marker='*')

    plt.xlabel('Mean runtime (ms)')
    plt.ylabel('AUC@10$\\deg$')
    plt.legend()
    plt.show()

def eval(args):
    dataset_path = args.dataset_path
    matches_basename = os.path.basename(args.feature_file)
    basename = os.path.basename(dataset_path)
    if args.graph:
        basename = f'{basename}-graph'
        # iterations_list = [100, 200, 500, 1000, 2000, 5000, 10000]
        iterations_list = [1000, 2000, 5000, 10000, 20000, 50000]
        # iterations_list = [100, 200, 500, 1000, 2000]
        # iterations_list = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
        # iterations_list = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    else:
        iterations_list = [None]

    # experiments = ['4p3v + LO', '4p3v + LO + C',  '4p3v + R + LO + C', '5p3v + LO', '5p3v + LO + C']
    # experiments = ['4p3v + LO', '4p3v + LO + D', '5p3v + LO', '4p3v + LO + C', '4p3v + LO + D + C', '5p3v + LO + C']
    experiments = ['4p3v + LO', '4p3v + LO + C', '4p3v + LO + D', '4p3v + LO + D + C', '5p3v + LO', '5p3v + LO + C']
    # experiments = ['4p3v + LO']

    json_path = os.path.join('results', f'{basename}-{matches_basename}.json')
    print(f'json_path: {json_path}')

    if args.load:
        with open(json_path, 'r') as f:
            results = json.load(f)
    else:
        R_file = h5py.File(os.path.join(dataset_path, 'R.h5'))
        T_file = h5py.File(os.path.join(dataset_path, 'T.h5'))
        C_file = h5py.File(os.path.join(dataset_path, f'{args.feature_file}.h5'))
        triplets = get_triplets(os.path.join(dataset_path, f'{args.feature_file}.txt'))

        R_dict = {k: np.array(v) for k, v in R_file.items()}
        T_dict = {k: np.array(v) for k, v in T_file.items()}
        camera_dicts = get_camera_dicts(os.path.join(dataset_path, 'K.h5'))

        if args.first is not None:
            triplets = triplets[:args.first]

        def gen_data():
            for triplet in triplets:
                img1, img2, img3 = triplet
                label = f"{img1}-{img2}-{img3}"

                pts = np.array(C_file[label])
                # we only check the first two snns to be consistent with Charalambos's eval code
                l = np.all(pts[:, 6:8] > 0.0, axis=1)

                x1 = pts[l, 0:2]
                x2 = pts[l, 2:4]
                x3 = pts[l, 4:6]

                if args.force_inliers is not None:
                    x1, x2, x3 = force_inliers(x1, x2, x3, img1, img2, img3, R_dict, T_dict, camera_dicts, args.force_inliers)
                    if len(x1) < 25:
                        continue

                for iterations in iterations_list:
                    for experiment in experiments:
                        # yield experiment, img1, img2, img3, x1, x2, x3, RR_dict, TT_dict, cam_dicts
                        yield experiment, iterations, img1, img2, img3, x1, x2, x3, R_dict, T_dict, camera_dicts


        total_length = len(experiments) * len(triplets) * len(iterations_list)
        print(f"Total runs: {total_length} for {len(triplets)} samples")

        if args.num_workers == 1:
            results = [eval_experiment(x) for x in tqdm(gen_data(), total=total_length)]
        else:
            pool = Pool(args.num_workers)
            results = [x for x in pool.imap(eval_experiment, tqdm(gen_data(), total=total_length))]

        print("Done")

    # draw_results(results, experiments, iterations_list)

    for experiment in experiments:
        print(50 * '*')
        print(f'Results for: {experiment}:')
        print_results([r for r in results if r['experiment'] == experiment])

    os.makedirs('results', exist_ok=True)

    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    args = parse_args()
    eval(args)