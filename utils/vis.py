import json
import os

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils.data import colors, experiments, iterations_list, get_basenames, styles

large_size = 24
small_size = 20

def draw_results(results, experiments, iterations_list, title=''):
    plt.figure()

    for experiment in tqdm(experiments):
        experiment_results = [x for x in results if x['experiment'] == experiment]

        xs = []
        ys = []

        for iterations in iterations_list:
            iter_results = [x for x in experiment_results if x['info']['iterations'] == iterations]
            mean_runtime = np.mean([x['info']['runtime'] for x in iter_results])
            # errs = np.array([max(1/3 *(out['R_12_err'] + out['R_13_err'] + out['R_23_err']), 1/3 * (out['t_12_err'] + out['t_13_err'] + out['t_23_err'])) for out in iter_results])
            errs = np.array([max(0.5 * (out['R_12_err'] + out['R_13_err']), 0.5 * (out['t_12_err'] + out['t_13_err'])) for out in iter_results])
            # errs = np.array([r['P_err'] for r in iter_results])
            errs[np.isnan(errs)] = 180
            AUC10 = np.mean(np.array([np.sum(errs < t) / len(errs) for t in range(1, 11)]))

            xs.append(mean_runtime)
            ys.append(AUC10)

        plt.semilogx(xs, ys, label=experiment, marker='*')

    title += f"Error: max(0.5 * (out['R_12_err'] + out['R_13_err']), 0.5 * (out['t_12_err'] + out['t_13_err']))"

    plt.title(title, fontsize=8)


    plt.xlabel('Mean runtime (ms)')
    plt.ylabel('AUC@10$\\deg$')
    plt.legend()
    plt.show()


def draw_results_pose_auc_10(results, experiments, iterations_list, title=None):
    plt.figure(frameon=False)

    for experiment in tqdm(experiments):
        experiment_results = [x for x in results if x['experiment'] == experiment]

        xs = []
        ys = []

        for iterations in iterations_list:
            iter_results = [x for x in experiment_results if x['info']['iterations'] == iterations]
            mean_runtime = np.mean([x['info']['runtime'] for x in iter_results])
            errs = np.array([max(0.5 * (out['R_12_err'] + out['R_13_err']), 0.5 * (out['t_12_err'] + out['t_13_err'])) for out in iter_results])
            # errs = np.array([0.5 * (out['t_12_err'] + out['t_13_err']) for out in iter_results])
            errs[np.isnan(errs)] = 180
            AUC10 = np.mean(np.array([np.sum(errs < t) / len(errs) for t in range(1, 11)]))
            # AUC10 = np.mean([x['info']['inlier_ratio'] for x in iter_results])

            xs.append(mean_runtime)
            ys.append(AUC10)

        plt.semilogx(xs, ys, label=experiment, marker='*', color=colors[experiment], linestyle=styles[experiment])

    plt.xlim([5.0, 1.9e4])
    plt.xlabel('Mean runtime (ms)', fontsize=large_size)
    plt.ylabel('AUC@10$^\\circ$', fontsize=large_size)
    plt.tick_params(axis='x', which='major', labelsize=small_size)
    plt.tick_params(axis='y', which='major', labelsize=small_size)
    if title is not None:
        # plt.legend()
        plt.savefig(f'figs/{title}_pose.pdf', bbox_inches='tight', pad_inches=0)
        print(f'saved pose: {title}')

    else:
        plt.legend()
        plt.show()


def draw_results_pose_portion(results, experiments, iterations_list, title=None):
    plt.figure(frameon=False)

    for experiment in tqdm(experiments):
        experiment_results = [x for x in results if x['experiment'] == experiment]

        xs = []
        ys = []

        iter_results = experiment_results
        mean_runtime = np.mean([x['info']['runtime'] for x in iter_results])
        # errs = np.array([max(0.5 * (out['R_12_err'] + out['R_13_err']), 0.5 * (out['t_12_err'] + out['t_13_err'])) for out in iter_results])
        errs = np.array([out['R_23_err'] for out in iter_results])
        # errs = np.array([0.5 * (out['t_12_err'] + out['t_13_err']) for out in iter_results])
        errs[np.isnan(errs)] = 180
        cum_err = np.array([np.sum(errs < t) / len(errs) for t in range(1, 181)])

        # AUC10 = np.mean([x['info']['inlier_ratio'] for x in iter_results])

        xs = np.arange(1, 181)
        ys = cum_err

        plt.plot(xs, ys, label=experiment, marker='*', color=colors[experiment])

    # plt.xlim([5.0, 1.9e4])
    plt.xlabel('Mean runtime (ms)', fontsize=large_size)
    plt.ylabel('AUC@10$^\\circ$', fontsize=large_size)
    plt.tick_params(axis='x', which='major', labelsize=small_size)
    plt.tick_params(axis='y', which='major', labelsize=small_size)
    if title is not None:
        # plt.legend()
        plt.savefig(f'figs/{title}_cumpose.pdf', bbox_inches='tight', pad_inches=0)
        print(f'saved pose: {title}')

    else:
        plt.legend()
        plt.show()

def generate_graphs(dataset, results_type, all=True):
    basenames = get_basenames(dataset)

    # results_type = 'graph-SIFT_triplet_correspondences'

    results = []
    for basename in basenames:
        json_path = os.path.join('results', f'{basename}-{results_type}.json')
        print(f'json_path: {json_path}')
        with open(json_path, 'r') as f:
            if all:
                results.extend(json.load(f))
            else:
                results = json.load(f)
                draw_results_pose_auc_10(results, experiments, iterations_list, f'{dataset}_{basename}_{results_type}')

    if all:
        title = f'{dataset}_{results_type}'
        draw_results_pose_auc_10(results, experiments, iterations_list, title)
    # draw_results_pose_portion(results, experiments, iterations_list, title)


if __name__ == '__main__':
    # generate_graphs('cambridge', 'graph-triplets-features_superpoint_noresize_2048-LG', all=False)
    # generate_graphs('pt', 'graph-0.4inliers-triplets-features_superpoint_noresize_2048-LG', all=False)
    # generate_graphs('pt', 'graph-triplets-features_superpoint_noresize_2048-LG', all=False)
    # generate_graphs('urban', 'graph-triplets-features_superpoint_noresize_2048-LG')
    # generate_graphs('pt', 'graph-triplets-features_loftr_1024_0')
    # generate_graphs('eth3d', 'graph-triplets-features_superpoint_1600_2048-LG')
    # generate_graphs('eth3d', 'graph-triplets-a0.4-features_superpoint_noresize_2048-LG')
    generate_graphs('aachen', 'graph-triplets-a0.4-features_superpoint_noresize_2048-LG')
