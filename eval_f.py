import argparse
import json
import os
# from multiprocessing.pool import ThreadPool as Pool
from multiprocessing import Pool
from time import perf_counter

import poselib
import h5py
import numpy as np
import torch
from prettytable import PrettyTable
from tqdm import tqdm

from utils.geometry import rotation_angle, angle, get_pose, get_gt_E, force_inliers
from utils.vis import draw_results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--first', type=int, default=None)
    parser.add_argument('-i', '--force_inliers', type=float, default=None)
    parser.add_argument('-nw', '--num_workers', type=int, default=1)
    parser.add_argument('-l', '--load', action='store_true', default=False)
    parser.add_argument('-fd', '--fix_delta', action='store_true', default=False)
    parser.add_argument('-s', '--synth', action='store_true', default=False)
    parser.add_argument('-g', '--graph', action='store_true', default=False)
    parser.add_argument('-d', '--delta', action='store_true', default=False)
    parser.add_argument('-a', '--append', action='store_true', default=False)
    parser.add_argument('-o', '--oracles', action='store_true', default=False)
    parser.add_argument('-r', '--refine', action='store_true', default=False)
    parser.add_argument('--all', action='store_true', default=False)
    parser.add_argument('feature_file')
    parser.add_argument('dataset_path')

    return parser.parse_args()


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
        d[key.replace('\\', '/')] = {'model': 'SIMPLE_PINHOLE', 'width': int(2 * K[0, 2]), 'height': int(2 * K[1,2]), 'params': [(K[0, 0] + K[1, 1]) * 0.5, K[0, 2], K[1, 2]]}

    return d

def get_triplets(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    return [line.strip().split(' ') for line in lines]


def get_result_dict(out, info, img1, img2, img3, R_dict, T_dict, K_dict):
    gt_R_12, gt_t_12 = get_pose(img1, img2, R_dict, T_dict)
    gt_R_13, gt_t_13 = get_pose(img1, img3, R_dict, T_dict)
    gt_R_23, gt_t_23 = get_pose(img2, img3, R_dict, T_dict)

    R_12, t_12 = out.poses.pose12.R, out.poses.pose12.t
    R_13, t_13 = out.poses.pose13.R, out.poses.pose13.t
    R_23, t_23 = out.poses.pose23().R, out.poses.pose23().t

    d = {}
    d['R_12_err'] = rotation_angle(R_12.T @ gt_R_12)
    d['R_13_err'] = rotation_angle(R_13.T @ gt_R_13)
    d['R_23_err'] = rotation_angle(R_23.T @ gt_R_23)

    d['t_12_err'] = angle(t_12, gt_t_12)
    d['t_13_err'] = angle(t_13, gt_t_13)
    d['t_23_err'] = angle(t_23, gt_t_23)

    d['P_12_err'] = max(d['R_12_err'], d['t_12_err'])
    d['P_13_err'] = max(d['R_13_err'], d['t_13_err'])
    d['P_23_err'] = max(d['R_23_err'], d['t_23_err'])

    d['P_err'] = max([v for k, v in d.items()])
    d['Charalambos_P_err'] = max(0.5 * (d['R_12_err'] + d['R_13_err']), 0.5 * (d['t_12_err'] + d['t_13_err']))
    d['1/3 P_err'] = max((d['R_12_err'] + d['R_13_err'] + d['R_23_err']) / 3, (d['t_12_err'] + d['t_13_err'] + d['t_23_err']) / 3)

    gt_focal_1 = K_dict[img1]['params'][0]
    gt_focal_2 = K_dict[img1]['params'][0]
    gt_focal_3 = K_dict[img1]['params'][0]

    focal = out.camera.focal()

    d['f1_err'] = np.abs(gt_focal_1 - focal) / gt_focal_1
    d['f2_err'] = np.abs(gt_focal_2 - focal) / gt_focal_2
    d['f3_err'] = np.abs(gt_focal_3 - focal) / gt_focal_3

    mean_gt_focal = (gt_focal_1 + gt_focal_2 + gt_focal_3) / 3
    d['f_err'] = np.abs(mean_gt_focal - focal) / mean_gt_focal

    info['inliers'] = []
    d['info'] = info
    return d


def print_results(results):
    tab = PrettyTable(['metric', 'median', 'mean', 'AUC@5', 'AUC@10', 'AUC@20'])
    tab.align["metric"] = "l"
    tab.float_format = '0.2'
    err_names = ['P_12_err', 'P_13_err', 'P_23_err', 'P_err', 'Charalambos_P_err', 'f_err']
    for err_name in err_names:
        errs = np.array([r[err_name] for r in results])
        errs[np.isnan(errs)] = 1.0 if err_name == 'f_err' else 180
        res = np.array([np.sum(errs < t) / len(errs) for t in range(1, 21)])
        tab.add_row([err_name, np.median(errs), np.mean(errs), np.mean(res[:5]), np.mean(res[:10]), np.mean(res)])

        # print(f'{err_name}: \t median: {np.median(errs):0.2f} \t mean: {np.mean(errs):0.2f} \t '
        #       f'auc5: {np.mean(res[:5]):0.2f} \t auc10: {np.mean(res[:10]):0.2f} \t auc20: {np.mean(res):0.2f}')

    for field in ['inlier_ratio', 'iterations', 'runtime', 'refinements']:
        xs = [r['info'][field] for r in results]
        tab.add_row([field, np.median(xs), np.mean(xs), '-', '-', '-'])
        # print(f'{field}: \t median: {np.median(xs):0.02f} \t mean: {np.mean(xs):0.02f}')

    print(tab)

def print_results_summary(results, experiments):
    tab = PrettyTable(['experiment', 'median', 'mean', 'AUC@5', 'AUC@10', 'AUC@20', 'Mean runtime', 'Med runtime'])
    tab.float_format = '0.2'

    for experiment in experiments:
        exp_results = [r for r in results if r['experiment'] == experiment]
        errs = np.array([r['Charalambos_P_err'] for r in exp_results])
        # errs = np.array([r['P_err'] for r in exp_results])
        errs[np.isnan(errs)] = 180
        res = np.array([np.sum(errs < t) / len(errs) for t in range(1, 21)])
        runtime = [r['info']['runtime'] for r in exp_results]
        tab.add_row([experiment, np.median(errs), np.mean(errs),
                     100 * np.mean(res[:5]), 100 * np.mean(res[:10]), 100 * np.mean(res),
                     np.mean(runtime), np.median(runtime)])

    print(tab)


def eval_experiment(x):
    torch.set_num_threads(1)
    experiment, iterations, img1, img2, img3, x1, x2, x3, R_dict, T_dict, camera_dicts = x

    use_net = '(L)' in experiment or '(L+D)' in experiment
    init_net = '(L-ID)' in experiment
    use_hc = 'HC' in experiment
    threeview_check = '+ C' in experiment
    oracle = '(O)' in experiment

    # using R
    inner_refine = 2 if '+ R' in experiment else 0
    if '+ R(' in experiment:
        idx = experiment.find('R(')
        idx_end = experiment[idx+2:].find(')')
        inner_refine = int(experiment[idx+2:idx + 2 + idx_end])
        # print(inner_refine)


    lo_iterations = 0 if '+ nLO' in experiment else 25

    # using delta
    if 'D' in experiment:
        if use_net or init_net:
            delta = 0.08
        else:
            delta = 0.08
    else:
        delta = 0

    if 'D(' in experiment:
        idx = experiment.find('D(')
        idx_end = experiment[idx+2:].find(')')
        delta = float(experiment[idx+2:idx + 2 + idx_end])

    num_pts = int(experiment[0])
    ransac_dict = {'max_epipolar_error': 5.0, 'progressive_sampling': False,
                   'min_iterations': 50, 'max_iterations': 5000, 'lo_iterations': lo_iterations,
                   'inner_refine': inner_refine, 'threeview_check': threeview_check, 'sample_sz': num_pts,
                   'delta': delta, 'use_hc': use_hc, 'use_net': use_net, 'init_net': init_net, 'oracle': oracle}

    if iterations is not None:
        ransac_dict['min_iterations'] = iterations
        ransac_dict['max_iterations'] = iterations

    bundle_dict = {'verbose': False, 'max_iterations': 0 if ' + nLO' in experiment else 100}
    pp = np.array(camera_dicts[img1]['params'][-2:])

    if oracle:
        focal = np.array(camera_dicts[img1]['params'][0])
        K = np.diag([focal, focal, 1])
        K[:2,2] = pp
        K_inv = np.linalg.inv(K)
        gt_E = get_gt_E(img1, img2, R_dict, T_dict, camera_dicts)
        gt_F = K_inv.T @ gt_E @ K_inv
        ransac_dict['gt_E'] = gt_F

    start = perf_counter()
    out, info = poselib.estimate_three_view_shared_focal_relative_pose(x1, x2, x3, pp, ransac_dict, bundle_dict)
    info['runtime'] = 1000 * (perf_counter() - start)
    result_dict = get_result_dict(out, info, img1, img2, img3, R_dict, T_dict, camera_dicts)
    result_dict['experiment'] = experiment
    result_dict['img1'] = img1
    result_dict['img2'] = img2
    result_dict['img3'] = img3

    # with open(f'results/{experiment}-{img1}-{img2}-{img3}.json', 'w') as f:
    #     json.dump(result_dict, f)

    return result_dict


def get_K(camera_dicts, img1):
    pp = np.array(camera_dicts[img1]['params'][-2:])
    focal = np.array(camera_dicts[img1]['params'][0])
    K = np.diag([focal, focal, 1])
    K[:2, 2] = pp
    return K


def eval(args):
    dataset_path = args.dataset_path
    matches_basename = os.path.basename(args.feature_file)
    basename = os.path.basename(dataset_path)
    if args.graph:
        basename = f'{basename}-graph'
        iterations_list = [100, 200, 500, 1000, 2000, 5000, 10000]
        # iterations_list = [20000, 50000]
    else:
        iterations_list = [None]

    if args.force_inliers is not None:
        basename = f'{basename}-{args.force_inliers:.1f}inliers'


    # experiments = ['4p3v(M)', '4p3v(M+D)', '4p3v(L)', '4p3v(L+D)', '4p3v(L+ID)', '4p3v(O)', '4p(HC)', '5p3v']
    # experiments = ['4p3v(M)', '4p3v(M+D)', '4p3v(M) + C', '4p3v(M+D) + C', '5p3v']
    if args.all:
        experiments = ['4p3v(M)', '4p3v(M) + R', '4p3v(M) + R + C', '4p3v(M) + C',
                       '4p3v(M-D)', '4p3v(M-D) + R', '4p3v(M-D) + R + C', '4p3v(M-D) + C',
                       '4p3v(L)', '4p3v(L) + R', '4p3v(L) + R + C', '4p3v(L) + C',
                       '4p3v(L-D)', '4p3v(L-D) + R', '4p3v(L-D) + R + C', '4p3v(L-D) + C',
                       '4p3v(L-ID)', '4p3v(L-ID) + R', '4p3v(L-ID) + R + C', '4p3v(L-ID) + C',
                       '6p3v', '4p3v(O)', '4p3v(O) + R', '4p3v(O) + R + C']
    else:
        experiments = ['4p3v(M)', '4p3v(M) + R', '4p3v(M) + R + C', '4p3v(M) + C',
                       '4p3v(M-D)', '4p3v(M-D) + R', '4p3v(M-D) + R + C', '4p3v(M-D) + C',
                       '6p3v']

    if args.refine:
        experiments = [f'4p3v(M) + R({x}) + C' for x in [20, 30, 40, 50, 100, 200]]

    if args.oracles:
        experiments = ['4p3v(O) + R', '4p3v(O) + R + C']

    if args.delta:
        samples = [0.2, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.005, 0.001]
        experiments = [f'4p3v(M-D({x}))' for x in samples]
        experiments.extend([f'4p3v(M-D({x})) + R' for x in samples])
        experiments.extend([f'4p3v(M-D({x})) + R + C' for x in samples])

    if args.fix_delta:
        if args.all:
            experiments = ['4p3v(M-D)', '4p3v(M-D) + R', '4p3v(M-D) + R + C', '4p3v(M-D) + C',
                           '4p3v(L-D)', '4p3v(L-D) + R', '4p3v(L-D) + R + C', '4p3v(L-D) + C',
                           '4p3v(L-ID)', '4p3v(L-ID) + R', '4p3v(L-ID) + R + C', '4p3v(L-ID) + C']
        else:
            experiments = ['4p3v(M-D)', '4p3v(M-D) + R', '4p3v(M-D) + R + C', '4p3v(M-D) + C']

    # experiments.extend([x + ' + C' for x in experiments])
    # experiments.extend([x + ' + R' for x in experiments])

    json_path = os.path.join('results', f'focal_{basename}-{matches_basename}.json')
    print(f'json_path: {json_path}')

    if args.load:
        with open(json_path, 'r') as f:
            results = json.load(f)

    else:
        R_file = h5py.File(os.path.join(dataset_path, 'R.h5'))
        T_file = h5py.File(os.path.join(dataset_path, 'T.h5'))
        C_file = h5py.File(os.path.join(dataset_path, f'{args.feature_file}.h5'))
        triplets = get_triplets(os.path.join(dataset_path, f'{args.feature_file}.txt'))

        R_dict = {k.replace('\\', '/'): np.array(v) for k, v in R_file.items()}
        T_dict = {k.replace('\\', '/'): np.array(v) for k, v in T_file.items()}
        camera_dicts = get_camera_dicts(os.path.join(dataset_path, 'K.h5'))

        if args.first is not None:
            triplets = triplets[:args.first]

        def gen_data():
            for triplet in triplets:
                img1, img2, img3 = triplet
                label = f"{img1}-{img2}-{img3}"

                pts = np.array(C_file[label])
                # we only check the first two snns to be consistent with Charalambos's eval code
                if 'SIFT_triplet_correspondences' in matches_basename:
                    l = np.all(pts[:, 6:8] <= 0.9, axis=1)
                else:
                    l = np.all(pts[:, 6:8] >= 0.5, axis=1)

                x1 = pts[l, 0:2]
                x2 = pts[l, 2:4]
                x3 = pts[l, 4:6]

                if args.synth:
                    K1 = get_K(camera_dicts, img1)
                    K2_inv = np.linalg.inv(get_K(camera_dicts, img2))
                    K3_inv = np.linalg.inv(get_K(camera_dicts, img3))

                    xx2 = np.column_stack([x2, np.ones(len(x2))])
                    xx3 = np.column_stack([x3, np.ones(len(x3))])

                    xx2 = ((K1 @ K2_inv) @ xx2.T).T
                    x2 = xx2[:, :2] / xx2[:, 2:]

                    xx3 = ((K1 @ K3_inv) @ xx3.T).T
                    x3 = xx3[:, :2] / xx3[:, 2:]

                R_dict_l = {x: R_dict[x] for x in [img1, img2, img3]}
                T_dict_l = {x: T_dict[x] for x in [img1, img2, img3]}
                camera_dicts_l = {x: camera_dicts[x] for x in [img1, img2, img3]}


                if args.force_inliers is not None:
                    x1, x2, x3 = force_inliers(x1, x2, x3, img1, img2, img3, R_dict_l, T_dict_l, camera_dicts_l,
                                               args.force_inliers)
                    if len(x1) < 25:
                        continue

                for iterations in iterations_list:
                    for experiment in experiments:
                        # yield experiment, img1, img2, img3, x1, x2, x3, RR_dict, TT_dict, cam_dicts
                        yield experiment, iterations, img1, img2, img3, x1, x2, x3, R_dict_l, T_dict_l, camera_dicts_l


        total_length = len(experiments) * len(triplets) * len(iterations_list)
        print(f"Total runs: {total_length} for {len(triplets)} samples")

        if args.num_workers == 1:
            results = [eval_experiment(x) for x in tqdm(gen_data(), total=total_length)]
        else:
            pool = Pool(args.num_workers)
            results = [x for x in pool.imap(eval_experiment, tqdm(gen_data(), total=total_length))]

        print("Done")

    if args.append:
        print(f"Appending from: {json_path}")
        with open(json_path, 'r') as f:
            prev_results = json.load(f)
        results.extend(prev_results)

    for experiment in experiments:
        print(50 * '*')
        print(f'Results for: {experiment}:')
        print_results([r for r in results if r['experiment'] == experiment])

    print(50 * '*')
    print(50 * '*')
    print(50 * '*')
    print_results_summary(results, experiments)

    os.makedirs('results', exist_ok=True)

    if not args.load:
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=4)

    title = f'Scene: {os.path.basename(dataset_path)} \n'
    title += f'Matches: {matches_basename}\n'

    draw_results(results, experiments, iterations_list, title=title)

if __name__ == '__main__':
    args = parse_args()
    eval(args)