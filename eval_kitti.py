import argparse
import json
import os
# from multiprocessing.pool import ThreadPool as Pool
from multiprocessing import Pool
from time import perf_counter

import poselib
import h5py
import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from tqdm import tqdm

from theory.lo_verification import skew
from utils.geometry import rotation_angle, angle #, get_pose, get_gt_E


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--first', type=int, default=None)
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
    t = R1.T @ (t2 - t1)
    return R, t



def get_result_dict(image_triplet, info, img1, img2, img3, R_file, T_file):
    gt_R_12 = np.eye(3)
    gt_t_12 = np.array([-3.861448000000e+02, 0.0, 0.0])

    gt_R_13, gt_t_13 = get_pose(img1, img3, R_file, T_file)
    # gt_R_23, gt_t_23 = get_pose(img2, img3, R_file, T_file)

    R_12, t_12 = image_triplet.poses.pose12.R, image_triplet.poses.pose12.t
    R_13, t_13 = image_triplet.poses.pose13.R, image_triplet.poses.pose13.t
    # R_23, t_23 = three_view_pose.pose23().R, three_view_pose.pose23().t

    out = {}
    out['R_12_err'] = rotation_angle(R_12.T @ gt_R_12)
    out['R_13_err'] = rotation_angle(R_13.T @ gt_R_13)
    # out['R_23_err'] = rotation_angle(R_23.T @ gt_R_23)

    out['t_12_err'] = angle(t_12, gt_t_12)
    out['t_13_err'] = angle(t_13, gt_t_13)
    # out['t_23_err'] = angle(t_23, gt_t_23)

    out['P_12_err'] = max(out['R_12_err'], out['t_12_err'])
    out['P_13_err'] = max(out['R_13_err'], out['t_13_err'])
    # out['P_23_err'] = max(out['R_23_err'], out['t_23_err'])

    out['Charalambos_P_err'] = max(0.5 * (out['R_12_err'] + out['R_13_err']), 0.5 * (out['t_12_err'] + out['t_13_err']))

    out['gt_f'] = 7.188560000000e+02
    out['f_est'] = image_triplet.camera.params[0]
    out['f_err'] = np.abs(out['gt_f'] - out['f_est']) / out['gt_f']
    # out['P_err'] = max([v for k, v in out.items()])
    out['info'] = info
    return out


def print_results(results):
    tab = PrettyTable(['metric', 'median', 'mean', 'AUC@5', 'AUC@10', 'AUC@20'])
    tab.align["metric"] = "l"
    tab.float_format = '0.2'
    err_names = ['R_12_err', 'R_13_err', 't_12_err', 't_13_err', 'P_12_err', 'P_13_err', 'Charalambos_P_err', 'f_err']
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

def print_results_summary(results, experiments):
    tab = PrettyTable(['experiment', 'median', 'mean', 'AUC@5', 'AUC@10', 'AUC@20', 'Mean runtime', 'Med runtime'])
    tab.float_format = '0.2'

    for experiment in experiments:
        exp_results = [r for r in results if r['experiment'] == experiment]
        errs = np.array([r['Charalambos_P_err'] for r in exp_results])
        errs[np.isnan(errs)] = 180
        res = np.array([np.sum(errs < t) / len(errs) for t in range(1, 21)])
        runtime = [r['info']['runtime'] for r in exp_results]
        tab.add_row([experiment, np.median(errs), np.mean(errs),
                     np.mean(res[:5]), np.mean(res[:10]), np.mean(res),
                     np.mean(runtime), np.median(runtime)])

    print(tab)


def eval_experiment(x):
    experiment, iterations, img1, img2, img3, x1, x2, x3, R_dict, T_dict, camera_dict = x

    use_net = '(L)' in experiment or '(L+D)' in experiment
    init_net = '(L+ID)' in experiment
    use_hc = 'HC' in experiment
    threeview_check = '+ C' in experiment
    oracle = '(O)' in experiment

    # using R
    inner_refine = 10 if '+ R' in experiment else 0
    lo_iterations = 0 if '+ nLO' in experiment else 25

    # using delta
    if 'D' in experiment:
        if use_net or init_net:
            delta = 0.0125
        else:
            delta = 0.025
    else:
        delta = 0

    num_pts = int(experiment[0])
    ransac_dict = {'max_epipolar_error': 1.0, 'progressive_sampling': False,
                   'min_iterations': 50, 'max_iterations': 5000, 'lo_iterations': lo_iterations,
                   'inner_refine': inner_refine, 'threeview_check': threeview_check, 'sample_sz': num_pts,
                   'delta': delta, 'use_hc': use_hc, 'use_net': use_net, 'init_net': init_net, 'oracle': oracle}

    if iterations is not None:
        ransac_dict['min_iterations'] = iterations
        ransac_dict['max_iterations'] = iterations

    if oracle:
        gt_t_12 = np.array([-3.861448000000e+02, 0.0, 0.0])
        K = np.array([[7.188560000000e+02, 0.0, 6.071928000000e+02], [0.0, 7.188560000000e+02, 1.852157000000e+02], [0.0, 0.0, 1.0]])
        K_inv = np.linalg.inv(K)
        E = K_inv.T @ (skew(-gt_t_12) @ K_inv)
        ransac_dict['gt_E'] = E

    bundle_dict = {'verbose': False, 'max_iterations': 0 if ' + nLO' in experiment else 100}
    start = perf_counter()
    pp = np.array(camera_dict['params'][-2:])
    image_triplet, info = poselib.estimate_three_view_shared_focal_relative_pose(x1, x2, x3, pp, ransac_dict, bundle_dict)
    info['runtime'] = 1000 * (perf_counter() - start)
    result_dict = get_result_dict(image_triplet, info, img1, img2, img3, R_dict, T_dict)
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
            errs = np.array([r['Charalambos_P_err'] for r in iter_results])
            errs[np.isnan(errs)] = 180
            AUC10 = np.mean(np.array([np.sum(errs < t) / len(errs) for t in range(1, 11)]))

            xs.append(mean_runtime)
            ys.append(AUC10)

        plt.semilogx(xs, ys, label=experiment, marker='*')

    plt.xlabel('Mean runtime (ms)')
    plt.ylabel('AUC@10$\\deg$')
    plt.legend()
    plt.show()


def get_pose_dicts(path):
    R_dict = {}
    T_dict = {}

    arr = np.loadtxt(path)
    for i, row in enumerate(arr):
        T = np.reshape(row, (3, 4))
        R_dict[i] = T[:, :3]
        # These are not translations but something else
        T_dict[i] = - T[:, 3]

    return R_dict, T_dict


def eval(args):
    dataset_path = args.dataset_path
    matches_basename = os.path.basename(args.feature_file)
    basename = os.path.basename(dataset_path)
    if args.graph:
        basename = f'{basename}-graph'
        iterations_list = [100, 200, 500, 1000, 2000, 5000, 10000]
    else:
        iterations_list = [None]

    experiments = ['4p3vf(M)', '4p3vf(M+D)', '4p3vf(L)', '4p3vf(L+D)', '4p3vf(L+ID)', '4p3vf(O)', '6p3vf']
    experiments.extend([x + ' + C' for x in experiments])
    # experiments.extend([x + ' + R' for x in experiments])


    json_path = os.path.join('results', f'{basename}-{matches_basename}.json')
    print(f'json_path: {json_path}')

    if args.load:
        with open(json_path, 'r') as f:
            results = json.load(f)
    else:
        C_file = h5py.File(os.path.join(dataset_path, f'{args.feature_file}.h5'))
        R_dict, T_dict = get_pose_dicts(os.path.join(dataset_path, 'poses', '00.txt'))
        camera_dict = {'model': 'SIMPLE_PINHOLE', 'width': -1, 'height': -1,
                       'params': [7.188560000000e+02, 6.071928000000e+02, 1.852157000000e+02]}

        labels = [x for x in C_file.keys() if len(x.split('-')) == 3]

        if args.first is not None:
            labels = labels[:args.first]

        def gen_data():
            for label in labels:
                img1, img2, img3 = label.split('-')

                try:
                    img1 = int(img1.split('_')[1].split('.')[0])
                    img3 = int(img3.split('.')[0])
                except Exception:
                    continue

                pts = np.array(C_file[label])
                # we only check the first two snns to be consistent with Charalambos's eval code
                # if 'SIFT_triplet_correspondences' in matches_basename:
                l = np.all(pts[:, 6:8] <= 0.9, axis=1)

                x1 = pts[l, 0:2]
                x2 = pts[l, 2:4]
                x3 = pts[l, 4:6]

                R_dict_l = {x: R_dict[x] for x in [img1, img3]}
                T_dict_l = {x: T_dict[x] for x in [img1, img3]}

                for iterations in iterations_list:
                    for experiment in experiments:
                        # yield experiment, img1, img2, img3, x1, x2, x3, RR_dict, TT_dict, cam_dicts
                        yield experiment, iterations, img1, img2, img3, x1, x2, x3, R_dict_l, T_dict_l, camera_dict


        total_length = len(experiments) * len(labels) * len(iterations_list)
        print(f"Total runs: {total_length} for {len(labels)} samples")

        if args.num_workers == 1:
            results = [eval_experiment(x) for x in tqdm(gen_data(), total=total_length)]
        else:
            pool = Pool(args.num_workers)
            results = [x for x in pool.imap(eval_experiment, tqdm(gen_data(), total=total_length))]

        print("Done")

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

    draw_results(results, experiments, iterations_list)

if __name__ == '__main__':
    args = parse_args()
    eval(args)