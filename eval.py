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

from utils.geometry import rotation_angle, angle, get_pose, get_gt_E, force_inliers, get_camera_dicts
from utils.vis import draw_results, draw_results_pose_portion, draw_results_pose_auc_10


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--first', type=int, default=None)
    parser.add_argument('--iters', type=int, default=None)
    parser.add_argument('-i', '--force_inliers', type=float, default=None)
    parser.add_argument('-t', '--threshold', type=float, default=1.0)
    parser.add_argument('-nw', '--num_workers', type=int, default=1)
    parser.add_argument('-l', '--load', action='store_true', default=False)
    parser.add_argument('-g', '--graph', action='store_true', default=False)
    parser.add_argument('-p', '--para', action='store_true', default=False)
    parser.add_argument('-fd', '--fix_delta', action='store_true', default=False)
    parser.add_argument('-d', '--delta', action='store_true', default=False)
    parser.add_argument('-a', '--append', action='store_true', default=False)
    parser.add_argument('-o', '--overwrite', action='store_true', default=False)
    parser.add_argument('-cp', '--check_previous', action='store_true', default=False)
    parser.add_argument('-e', '--early', action='store_true', default=False)
    parser.add_argument('--affine', action='store_true', default=False)
    parser.add_argument('--oracles', action='store_true', default=False)
    parser.add_argument('--rebuttal', action='store_true', default=False)
    parser.add_argument('--table', action='store_true', default=False)
    parser.add_argument('--learning', action='store_true', default=False)
    parser.add_argument('--toptim', action='store_true', default=False)
    parser.add_argument('--final', action='store_true', default=False)
    parser.add_argument('--nister', action='store_true', default=False)
    parser.add_argument('--rc', action='store_true', default=False)
    parser.add_argument('-r', '--refine', action='store_true', default=False)
    parser.add_argument('--all', action='store_true', default=False)
    parser.add_argument('feature_file')
    parser.add_argument('dataset_path')

    return parser.parse_args()


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
    out['Charalambos_P_err'] = max(0.5 *(out['R_12_err'] + out['R_13_err']), 0.5 * (out['t_12_err'] + out['t_13_err']))
    out['1/3 P_err'] = max((out['R_12_err'] + out['R_13_err'] + out['R_23_err'])/3, (out['t_12_err'] + out['t_13_err'] + out['t_23_err'])/3)

    info['inliers'] = []
    out['info'] = info
    return out


def print_results(results):
    tab = PrettyTable(['metric', 'median', 'mean', 'AUC@5', 'AUC@10', 'AUC@20'])
    tab.align["metric"] = "l"
    tab.float_format = '0.2'
    err_names = ['P_12_err', 'P_13_err', 'P_23_err', 'P_err', 'Charalambos_P_err']
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
    tab = PrettyTable(['experiment', 'median', 'mean', 'AUC@5', 'AUC@10', 'AUC@20', 'Mean runtime', 'Med runtime', 'Mean inliers', 'Med inliers'])
    tab.float_format = '0.2'

    for experiment in experiments:
        exp_results = [r for r in results if r['experiment'] == experiment]
        errs = np.array([r['Charalambos_P_err'] for r in exp_results])
        # errs = np.array([r['P_err'] for r in exp_results])
        errs[np.isnan(errs)] = 180
        res = np.array([np.sum(errs < t) / len(errs) for t in range(1, 21)])
        runtime = [r['info']['runtime'] for r in exp_results]
        inlier_ratios = [r['info']['inlier_ratio'] for r in exp_results]
        tab.add_row([experiment, np.median(errs), np.mean(errs),
                     100 * np.mean(res[:5]), 100 * np.mean(res[:10]), 100 * np.mean(res),
                     np.mean(runtime), np.median(runtime), np.mean(inlier_ratios), np.median(inlier_ratios)])

    print(tab)


def eval_experiment(x):
    torch.set_num_threads(1)
    experiment, iterations, img1, img2, img3, x1, x2, x3, R_dict, T_dict, camera_dicts, t = x

    use_net = '(L)' in experiment or '(L-D)' in experiment
    init_net = '(L-ID)' in experiment
    use_hc = 'HC' in experiment
    threeview_check = '+ C' in experiment
    oracle = '(O)' in experiment
    affine = '(A)' in experiment
    use_para = '(P)' in experiment
    early_lm = '+ ELM' in experiment
    early_nm = '+ ENM' in experiment

    if 'N1' in experiment:
        nister = 1
    elif 'N3' in experiment:
        nister = 2
    else:
        nister = 0

    # using R
    inner_refine = 2 if '+ R' in experiment else 0
    if '+ R(' in experiment:
        idx = experiment.find('R(')
        idx_end = experiment[idx+2:].find(')')
        inner_refine = int(experiment[idx+2:idx + 2 + idx_end])

    inner_refine_extra = 1 if '+ RR' in experiment else 0
    threeview_check_extra = '+ CC' in experiment


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
    ransac_dict = {'max_epipolar_error': t, 'progressive_sampling': False,
                   'min_iterations': 100, 'max_iterations': 1000, 'lo_iterations': lo_iterations,
                   'inner_refine': inner_refine, 'threeview_check': threeview_check, 'sample_sz': num_pts,
                   'delta': delta, 'use_hc': use_hc, 'use_net': use_net, 'init_net': init_net, 'oracle': oracle,
                   'use_affine': affine, 'early_lm': early_lm, 'early_nonminimal': early_nm, 'use_para': use_para,
                   'use_nister': nister, 'inner_refine_extra': inner_refine_extra,
                   'threeview_check_extra': threeview_check_extra}

    if iterations is not None:
        ransac_dict['min_iterations'] = iterations
        ransac_dict['max_iterations'] = iterations

    if oracle or nister > 0:
        gt_E = get_gt_E(img1, img2, R_dict, T_dict, camera_dicts)
        ransac_dict['gt_E'] = gt_E

    bundle_dict = {'verbose': False, 'max_iterations': 0 if ' + nLO' in experiment else 100}
    start = perf_counter()
    three_view_pose, info = poselib.estimate_three_view_relative_pose(x1, x2, x3, camera_dicts[img1],
                                                                      camera_dicts[img2], camera_dicts[img3],
                                                                      ransac_dict, bundle_dict)
    info['runtime'] = 1000 * (perf_counter() - start)
    result_dict = get_result_dict(three_view_pose, info, img1, img2, img3, R_dict, T_dict)
    result_dict['experiment'] = experiment
    result_dict['img1'] = img1
    result_dict['img2'] = img2
    result_dict['img3'] = img3

    # with open(f'results/{experiment}-{img1}-{img2}-{img3}.json', 'w') as f:
    #     json.dump(result_dict, f)

    return result_dict


def fix_ch_err(results):
    for out in results:
        out['Charalambos P_err'] = max(0.5 * (out['R_12_err'] + out['R_13_err']), 0.5 * (out['t_12_err'] + out['t_13_err']))

def eval(args):
    dataset_path = args.dataset_path
    matches_basename = os.path.basename(args.feature_file)
    basename = os.path.basename(dataset_path)
    if args.graph:
        basename = f'{basename}-graph'
        iterations_list = [100, 200, 500, 1000, 2000, 5000, 10000]
        # iterations_list = [20000, 50000]
    else:
        iterations_list = [args.iters]

    if args.force_inliers is not None:
        basename = f'{basename}-{args.force_inliers:.1f}inliers'

    if args.threshold != 1.0:
        basename = f'{basename}-{args.threshold}t'


    # experiments = ['4p3v(M)', '4p3v(M+D)', '4p3v(L)', '4p3v(L+D)', '4p3v(L+ID)', '4p3v(O)', '4p(HC)', '5p3v']
    # experiments = ['4p3v(M)', '4p3v(M+D)', '4p3v(M) + C', '4p3v(M+D) + C', '5p3v']
    if args.all:
        experiments = ['4p3v(M)', '4p3v(M) + R', '4p3v(M) + R + C', '4p3v(M) + C',
                       '4p3v(M-D)', '4p3v(M-D) + R', '4p3v(M-D) + R + C', '4p3v(M-D) + C',
                       '4p3v(L)', '4p3v(L) + R', '4p3v(L) + R + C', '4p3v(L) + C',
                       '4p3v(L-D)', '4p3v(L-D) + R', '4p3v(L-D) + R + C', '4p3v(L-D) + C',
                       '4p3v(L--ID)', '4p3v(L--ID) + R', '4p3v(L--ID) + R + C', '4p3v(L--ID) + C',
                       '4p(HC)', '5p3v', '4p3v(O)', '4p3v(O) + R', '4p3v(O) + R + C',
                       '4p3v(A)', '3p3v(A)', '2p3v(A)', '4p3v(A) + nLO', '3p3v(A) + nLO', '2p3v(A) + nLO']

    else:
        experiments = ['4p3v(M)', '4p3v(M) + R', '4p3v(M) + R + C',
                       '4p3v(M-D)', '4p3v(M-D) + R', '4p3v(M-D) + R + C',
                       '4p(HC)', '5p3v',
                       '4p3v(O)', '4p3v(O) + R', '4p3v(O) + R + C',
                       '4p3v(A)', '4p3v(A) + R', '4p3v(A) + R + C',
                       '3p3v(A)', '2p3v(A)',
                       '4p3v(A) + ENM', '4p3v(A) + R + ENM', '4p3v(A) + R + C + ENM',
                       '3p3v(A) + ENM', '2p3v(A) + ENM',
                       '4p3v(N1)', '4p3v(N3)', '4p3v(N1) + ENM', '4p3v(N3) + ENM']

    if args.learning:
        experiments = ['4p3v(L) + R + C', '4p3v(L-D) + R + C', '4p3v(L-ID) + R + C',
                       '4p3v(L) + R', '4p3v(L-D) + R', '4p3v(L-ID) + R',
                       '4p3v(L)', '4p3v(L-D)', '4p3v(L-ID)']

    if args.table:
        experiments = [
            '5p3v', '5p3v + ENM', '4p(HC)',
            '4p3v(A)', '4p3v(A) + ENM', '4p3v(A) + R + ENM', '4p3v(A) + R + C + ENM',
            '4p3v(A) + R', '4p3v(A) + R + C',
            '4p3v(M)', '4p3v(M) + ENM', '4p3v(M) + R + ENM', '4p3v(M) + R + C + ENM',
            '4p3v(M) + R', '4p3v(M) + R + C',
            '4p3v(M-D)', '4p3v(M-D) + ENM', '4p3v(M-D) + R + ENM', '4p3v(M-D) + R + C + ENM',
            '4p3v(M-D) + R', '4p3v(M-D) + R + C']

    if args.oracles:
        experiments = ['4p3v(O)', '4p3v(O) + R']

    if args.fix_delta:
        if args.all:
            experiments = ['4p3v(L--ID)', '4p3v(L--ID) + R', '4p3v(L--ID) + R + C', '4p3v(L--ID) + C']
        else:
            experiments = []

    if args.refine:
        experiments = [f'4p3v(M-D) + R({x}) + C' for x in [1, 2, 3, 5, 10]]

    if args.delta:
        # samples = [0.2, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.005, 0.001]
        samples = [0.2, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.01, 0.005, 0.001]
        experiments = [f'4p3v(M-D({x}))' for x in samples]
        experiments.extend([f'4p3v(M-D({x})) + R' for x in samples])
        experiments.extend([f'4p3v(M-D({x})) + R + C' for x in samples])

    if args.affine:
        experiments = ['4p3v(A)', '4p3v(A) + R + C', '3p3v(A)', '2p3v(A)']

    if args.early:
        # experiments = ['4p3v(M) + ELM', '4p3v(M) + R + C + ELM', '4p3v(M) + ENM', '4p3v(M) + R + C + ENM']
        experiments = ['5p3v + ENM', '4p3v(M) + R + C + ENM', '4p3v(M-D) + R + C + ENM', '4p3v(A) + R + C + ENM', '3p3v(A) + ENM', '2p3v(A) + ENM']

    if args.para:
        experiments = ['4p3v(P)']

    if args.toptim:
        experiments = ['5p3v', '4p3v(M-D) + R + C',  '4p3v(M) + R + C', '4p(HC)']

    if args.rc:
        experiments = ['4p3v(M)', '4p3v(M) + R', '4p3v(M-D)', '4p3v(M-D) + R', '4p3v(A)', '4p3v(A) + R']

    if args.final:
        experiments = ['4p3v(M) + R + C', '4p3v(M-D) + R + C', '5p3v', '4p(HC)',
                       '4p3v(M) + R + C + ENM', '4p3v(M-D) + R + C + ENM', '5p3v + ENM',
                       '4p3v(A) + R + C + ENM', '4p3v(A) + R + C', '3p3v(A) + ENM', '3p3v(A)']
    if args.rebuttal:
        experiments = ['5p3v + ENM + RR + CC', '5p3v + RR + CC']

    if args.nister:
        experiments = ['4p3v(N1)', '4p3v(N3)', '4p3v(N1) + ENM', '4p3v(N3) + ENM']

    # experiments.extend([x + ' + C' for x in experiments])
    # experiments.extend([x + ' + R' for x in experiments])

    json_path = os.path.join('results', f'{basename}-{matches_basename}.json')
    print(f'json_path: {json_path}')

    if args.check_previous:
        print("Checking previous!")
        if os.path.exists(json_path):
            if not args.append:
                raise ValueError("Ran check previous without append when the results file already exists! Aborting!")

            with open(json_path, 'r') as f:
                prev_results = json.load(f)

            prev_experiments = set([x['experiment'] for x in prev_results])

            experiments = list(set(experiments).difference(prev_experiments))

            print("Some experiments already found. Only performing experiments: ", experiments)
        else:
            print("Prev file not found")

    print("Experiments: ", experiments)

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

                R_dict_l = {x: R_dict[x] for x in [img1, img2, img3]}
                T_dict_l = {x: T_dict[x] for x in [img1, img2, img3]}
                camera_dicts_l = {x: camera_dicts[x] for x in [img1, img2, img3]}


                if args.force_inliers is not None:
                    x1, x2, x3 = force_inliers(x1, x2, x3, img1, img2, img3, R_dict_l, T_dict_l, camera_dicts_l,
                                               args.force_inliers, args.threshold)
                    if len(x1) < 25:
                        continue

                for iterations in iterations_list:
                    for experiment in experiments:
                        # yield experiment, img1, img2, img3, x1, x2, x3, RR_dict, TT_dict, cam_dicts
                        yield experiment, iterations, img1, img2, img3, x1, x2, x3, R_dict_l, T_dict_l, camera_dicts_l, args.threshold


        total_length = len(experiments) * len(triplets) * len(iterations_list)
        print(f"Total runs: {total_length} for {len(triplets)} samples")

        if args.num_workers == 1:
            results = [eval_experiment(x) for x in tqdm(gen_data(), total=total_length)]
        else:
            pool = Pool(args.num_workers)
            results = [x for x in pool.imap(eval_experiment, tqdm(gen_data(), total=total_length))]

        print("Done")

        if args.append:
            if os.path.exists(json_path):
                print(f"Appending from: {json_path}")
                with open(json_path, 'r') as f:
                    prev_results = json.load(f)
            else:
                print("Prev file not found!")
                prev_results = []

            if args.overwrite:
                prev_results = [x for x in prev_results if x['experiment'] not in experiments]

            results.extend(prev_results)

    fix_ch_err(results)

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

    # draw_results_pose_auc_10(results, experiments, iterations_list, title=json_path)
    # draw_results(results, experiments, iterations_list, title=title)
    # draw_results_pose_portion(results, experiments, iterations_list)

if __name__ == '__main__':
    args = parse_args()
    eval(args)