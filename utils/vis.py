import json
import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import seaborn as sns
from tqdm import tqdm

from utils.data import experiments, iterations_list, get_basenames, styles, err_fun_main, err_fun_max, \
    err_twoview, twoview_experiments

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


def draw_results_pose_auc_10(results, experiments, iterations_list, title=None, xlim=(5.0, 1.9e4), err_fun=err_fun_main):
    plt.figure(frameon=False)

    for experiment in tqdm(experiments):
        experiment_results = [x for x in results if x['experiment'] == experiment]

        xs = []
        ys = []

        for iterations in iterations_list:
            iter_results = [x for x in experiment_results if x['info']['iterations'] == iterations]
            mean_runtime = np.mean([x['info']['runtime'] for x in iter_results])
            errs = np.array([err_fun(out) for out in iter_results])
            # errs = np.array([0.5 * (out['t_12_err'] + out['t_13_err']) for out in iter_results])
            errs[np.isnan(errs)] = 180
            AUC10 = np.mean(np.array([np.sum(errs < t) / len(errs) for t in range(1, 11)]))
            # AUC10 = np.mean([x['info']['inlier_ratio'] for x in iter_results])

            xs.append(mean_runtime)
            ys.append(AUC10)

        if colors is None:
            colors = {exp: sns.color_palette("hls", len(experiments))[i] for i, exp in enumerate(experiments)}

        plt.semilogx(xs, ys, label=experiment, marker='*', color=colors[experiment], linestyle=styles[experiment])

    plt.xlim(xlim)
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
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
        errs = np.array([out['t_13_err'] for out in iter_results])
        # errs = np.array([0.5 * (out['t_12_err'] + out['t_13_err']) for out in iter_results])
        errs[np.isnan(errs)] = 180
        cum_err = np.array([np.sum(errs < t) / len(errs) for t in range(1, 181)])

        # AUC10 = np.mean([x['info']['inlier_ratio'] for x in iter_results])

        xs = np.arange(1, 181)
        ys = cum_err

        # plt.plot(xs, ys, label=experiment, marker='*', color=colors[experiment])
        plt.plot(xs, ys, label=experiment, marker='*')

    # plt.xlim([5.0, 1.9e4])
    plt.xlabel('Pose error', fontsize=large_size)
    plt.ylabel('Portion of samples', fontsize=large_size)
    plt.tick_params(axis='x', which='major', labelsize=small_size)
    plt.tick_params(axis='y', which='major', labelsize=small_size)
    if title is not None:
        # plt.legend()
        plt.savefig(f'figs/{title}_cumpose.pdf', bbox_inches='tight', pad_inches=0)
        print(f'saved pose: {title}')

    else:
        plt.legend()
        plt.show()

def generate_graphs(dataset, results_type, all=True, use_max_err=False):
    basenames = get_basenames(dataset)

    # results_type = 'graph-SIFT_triplet_correspondences'
    if use_max_err:
        err_fun = err_fun_max
        err_str = 'maxerr_'
    else:
        err_fun = err_fun_main
        err_str = ''

    results = []
    for basename in basenames:
        json_path = os.path.join('results', f'{basename}-{results_type}.json')
        print(f'json_path: {json_path}')
        with open(json_path, 'r') as f:
            if all:
                results.extend([x for x in json.load(f) if x['experiment'] in experiments])
            else:
                results = [x for x in json.load(f) if x['experiment'] in experiments]
                draw_results_pose_auc_10(results, experiments, iterations_list,
                                         f'{err_str}{dataset}_{basename}_{results_type}', err_fun=err_fun)

    if all:
        title = f'{err_str}{dataset}_{results_type}'
        draw_results_pose_auc_10(results, experiments, iterations_list, title, err_fun=err_fun)
    # draw_results_pose_portion(results, experiments, iterations_list, title)

def generate_graphs_twoview(dataset, results_type, all=True, use_max_err=False):
    basenames = get_basenames(dataset)

    # results_type = 'graph-SIFT_triplet_correspondences'
    err_fun = err_twoview
    err_str = 'twoview'

    results = []
    for basename in basenames:
        json_path = os.path.join('results', f'twoview-{basename}-{results_type}.json')
        print(f'json_path: {json_path}')
        with open(json_path, 'r') as f:
            if all:
                results.extend([x for x in json.load(f) if x['experiment'] in twoview_experiments])
            else:
                results = [x for x in json.load(f) if x['experiment'] in twoview_experiments]
                draw_results_pose_auc_10(results, twoview_experiments, iterations_list,
                                         f'{err_str}{dataset}_{basename}_{results_type}', err_fun=err_fun)

    if all:
        title = f'{err_str}{dataset}_{results_type}'
        draw_results_pose_auc_10(results, twoview_experiments, iterations_list, title, err_fun=err_fun)
    # draw_results_pose_portion(results, experiments, iterations_list, title)

def generate_outliers():
    basenames = ['StMarysChurch', 'sacre_coeur']


    results_types = [f'graph-{x}inliers-triplets-features_superpoint_noresize_2048-LG' for x in ['0.2', '0.4', '0.6']]
    results_types.append('graph-triplets-features_superpoint_noresize_2048-LG')

    for basename in basenames:
        for results_type in results_types:
            json_path = os.path.join('results', f'{basename}-{results_type}.json')
            print(f'json_path: {json_path}')
            with open(json_path, 'r') as f:
                results = [x for x in json.load(f) if x['experiment'] in experiments]
                draw_results_pose_auc_10(results, experiments, iterations_list, f'{basename}_{results_type}', xlim=(2.0, 7.9e3))

def generate_refinement_graph():
    json_path = os.path.join('results', 'st_peters_square-graph-triplets-features_superpoint_noresize_2048-LG.json')
    print(f'json_path: {json_path}')
    with open(json_path, 'r') as f:
        results = ([x for x in json.load(f) if 'R(' in x['experiment']])

    plt.figure(frameon=False)

    samples = [1, 2, 3, 5, 10]

    for steps in tqdm(samples):
        experiment = f'4p3v(M) + R({steps}) + C'
        experiment_results = [x for x in results if x['experiment'] == experiment]

        xs = []
        ys = []

        for iterations in iterations_list:
            iter_results = [x for x in experiment_results if x['info']['iterations'] == iterations]
            mean_runtime = np.mean([x['info']['runtime'] for x in iter_results])
            errs = np.array([err_fun_main(out) for out in iter_results])
            # errs = np.array([0.5 * (out['t_12_err'] + out['t_13_err']) for out in iter_results])
            errs[np.isnan(errs)] = 180
            AUC10 = np.mean(np.array([np.sum(errs < t) / len(errs) for t in range(1, 11)]))
            # AUC10 = np.mean([x['info']['inlier_ratio'] for x in iter_results])

            xs.append(mean_runtime)
            ys.append(AUC10)

        plt.semilogx(xs, ys, label=steps, marker='*')

    small_size = 10
    large_size = 12


    plt.xlim([8.0, 1.9e3])
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    plt.xlabel('Mean runtime (ms)', fontsize=large_size)
    plt.ylabel('AUC@10$^\\circ$', fontsize=large_size)
    plt.tick_params(axis='x', which='major', labelsize=small_size)
    plt.tick_params(axis='y', which='major', labelsize=small_size)
    plt.legend(title='LM Iterations',loc=4, prop={'size': small_size}, title_fontproperties={'size': small_size})
    plt.savefig(f'figs/st_peters_square_refinement_validation.pdf', bbox_inches='tight', pad_inches=0)

if __name__ == '__main__':
    # generate_outliers()
    # generate_refinement_graph()
    generate_graphs_twoview('pt', '5.0t-graph-pairs-features_superpoint_noresize_2048-LG', all=True, use_max_err=True)
    # generate_graphs('cambridge', 'graph-triplets-features_superpoint_noresize_2048-LG', all=True, use_max_err=True)
    # generate_graphs('pt', 'graph-triplets-features_superpoint_noresize_2048-LG', all=True, use_max_err=True)
    # generate_graphs('cambridge', 'graph-triplets-features_superpoint_noresize_2048-LG', all=False)
    # generate_graphs('pt', 'graph-0.4inliers-triplets-features_superpoint_noresize_2048-LG', all=False)
    # generate_graphs('pt', 'graph-triplets-features_superpoint_noresize_2048-LG', all=True, use_max_err=True)
    # generate_graphs('pt', 'graph-triplets-features_superpoint_noresize_2048-LG', all=False)
    # generate_graphs('urban', 'graph-triplets-features_superpoint_noresize_2048-LG')
    # generate_graphs('pt', 'graph-triplets-features_loftr_1024_0')
    # generate_graphs('eth3d', 'graph-triplets-features_superpoint_1600_2048-LG')
    # generate_graphs('eth3d', 'graph-triplets-a0.4-features_superpoint_noresize_2048-LG')
    # generate_graphs('aachen', 'graph-triplets-a0.4-features_superpoint_noresize_2048-LG')
