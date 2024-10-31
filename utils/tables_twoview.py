import json
import os

import numpy as np

from eval import print_results_summary
from utils.data import basenames_pt, basenames_eth, basenames_cambridge, get_basenames, err_fun_max, err_fun_main, \
    err_twoview

incdec = [1, 1, -1, -1, -1, 1]

experiments = ['5pE', '5pE + ELM', '5pE + ENM',
               '4pE(M)', '4pE(M) + ELM', '4pE(M) + ENM',
               '4pF(A)', '3pH(A)',
               '4pH', '4pH + ENM', '4pH + ELM']

names = { x: x for x in experiments}

def print_table_text(experiments, rows):

    print(
        f'\\begin{{tabular}}{{ l | c c | c c c | c}}\n'
        f'    \\toprule\n'
        f'    Estimator & AVG $(^\\circ)$ $\\downarrow$ & MED $(^\\circ)$ $\\downarrow$ & AUC@5 $\\uparrow$ & @10 $\\uparrow$ & @20 $\\uparrow$ & Runtime (ms) $\\downarrow$\\\\\n'
        f'    \\midrule\n')

    for experiment, row in zip(experiments, rows):
        print(f'{names[experiment]} & {row} \\\\')

    print('\\midrule')

def get_rows(results, order, err_fun, runtime=True):
    num_rows = []

    l = np.array(['(O)' in exp for exp in order])

    for experiment in order:
        exp_results = [x for x in results if x['experiment'] == experiment]

        p_errs = np.array([err_fun(out) for out in exp_results])
        p_errs[np.isnan(p_errs)] = 180
        p_res = np.array([np.sum(p_errs < t) / len(p_errs) for t in range(1, 21)])
        p_auc_5 = np.mean(p_res[:5])
        p_auc_10 = np.mean(p_res[:10])
        p_auc_20 = np.mean(p_res)
        p_avg = np.mean(p_errs)
        p_med = np.median(p_errs)

        times = [r['info']['runtime'] for r in exp_results]
        time_avg = np.mean(times)

        num_rows.append([p_avg, p_med, 100 * p_auc_5, 100 * p_auc_10, 100 * p_auc_20, time_avg])

    text_rows = [[f'{x:0.2f}' for x in y] for y in num_rows]
    lens = np.array([[len(x) for x in y] for y in text_rows])
    arr = np.array(num_rows)
    arr[l, :] = np.inf * np.array([incdec])
    for j in range(len(text_rows[0])):
        idxs = np.argsort(incdec[j] * arr[:, j])
        text_rows[idxs[0]][j] = '\\textbf{' + text_rows[idxs[0]][j] + '}'
        text_rows[idxs[1]][j] = '\\underline{' + text_rows[idxs[1]][j] + '}'

    max_len = np.max(lens, axis=0)
    phantoms = max_len - lens
    for i in range(len(text_rows)):
        for j in range(len(text_rows[0])):
            if phantoms[i, j] > 0:
                text_rows[i][j] = '\\phantom{' + (phantoms[i, j] * '1') + '}' + text_rows[i][j]

    if runtime:
        return [' & '.join(row) for row in text_rows]
    else:
        return [' & '.join(row[:-1]) for row in text_rows]

def generate_table(dataset, feat, all_experiments=False, use_max_err=False):
    basenames = get_basenames(dataset)

    err_fun = err_twoview


    if all_experiments:
        l_experiments = names.keys()
    else:
        l_experiments = experiments

    results_type = f'5.0t-pairs-features_{feat}_noresize_2048-LG'

    results = []
    for basename in basenames:
        json_path = os.path.join('results', f'twoview-{basename}-{results_type}.json')
        print(f'json_path: {json_path}')
        with open(json_path, 'r') as f:
            results.extend([x for x in json.load(f) if x['experiment'] in l_experiments])

    print("Data loaded")

    print(30 * '*')
    print(30 * '*')
    print(30 * '*')
    print("Printing: ", dataset)
    print(30 * '*')

    rows = get_rows(results, experiments, err_fun)
    print_table_text(experiments, rows)


def print_delta_rows(name, rows, samples):
    print('\\multirow{' + str(len(samples)) + '}{*}{' + name + '}')
    for row, sample in zip(rows, samples):
        print(f' & {sample} & {row} \\\\')


def generate_delta_table():
    json_path = os.path.join('results', 'st_peters_square-triplets-features_superpoint_noresize_2048-LG.json')
    print(f'json_path: {json_path}')
    with open(json_path, 'r') as f:
        results = ([x for x in json.load(f) if 'D(' in x['experiment']])

    samples = [0.2, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.005, 0.001]
    experiments_M = [f'4p3v(M+D({x}))' for x in samples]
    experiments_R = [f'4p3v(M+D({x})) + R' for x in samples]
    experiments_RC = [f'4p3v(M+D({x})) + R + C' for x in samples]

    rows_M = get_rows(results, experiments_M, err_fun=err_fun_main, runtime=False)
    rows_R = get_rows(results, experiments_R, err_fun=err_fun_main, runtime=False)
    rows_RC = get_rows(results, experiments_RC, err_fun=err_fun_main, runtime=False)

    print(
        '\\begin{tabular}{ l | l | c c | c c c}\n'
        '\\toprule\n'
        'Estimator & \\multicolumn{1}{|c|}{$\\delta$} & AVG $(^\\circ)$ $\\downarrow$ & MED $(^\\circ)$ $\\downarrow$ & AUC@5 $\\uparrow$ & @10 $\\uparrow$ & @20 $\\uparrow$ \\\\\n'
        '\\midrule\n')

    print_delta_rows('\\sftmd', rows_M, samples)
    print('\\midrule')
    print_delta_rows('\\sftmdR', rows_R, samples)
    print('\\midrule')
    print_delta_rows('\\sftmdRC', rows_RC, samples)
    print('\\midrule')


if __name__ == '__main__':
    generate_table('pt', 'superpoint', all_experiments=False)
    # generate_table('cambridge', 'superpoint', all_experiments=False, use_max_err=True)
    # generate_delta_table()

