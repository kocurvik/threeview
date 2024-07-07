import json
import os

from utils.vis import draw_results


def draw_pt():
    # results_type = 'graph-triplets-features_loftr_1024_0'
    # results_type = 'graph-SIFT_triplet_correspondences'
    results_type = 'graph-triplets-features_superpoint_noresize_2048-LG'

    iterations_list = [100, 200, 500, 1000, 2000, 5000, 10000]
    experiments = ['4p3v(M)', '4p3v(M) + R', '4p3v(M) + R + C', '4p3v(M) + C',
                   '4p3v(M+D)', '4p3v(M+D) + R', '4p3v(M+D) + R + C', '4p3v(M+D) + C',
                   '4p(HC)', '5p3v', '4p3v(O)']

    basenames = ['brandenburg_gate', 'buckingham_palace', 'colosseum_exterior', 'grand_place_brussels',
                 'notre_dame_front_facade', 'palace_of_westminster', 'pantheon_exterior', 'reichstag',
                 'sacre_coeur', 'st_peters_square', 'taj_mahal', 'temple_nara_japan', 'trevi_fountain']
    # basenames = ['sacre_coeur', 'st_peters_square']
    results = []
    for basename in basenames:
        json_path = os.path.join('results', f'{basename}-{results_type}.json')
        print(f'json_path: {json_path}')
        with open(json_path, 'r') as f:
            results.extend(json.load(f))

    print("Data loaded")

    draw_results(results, experiments, iterations_list, title='Phototourism All - SPLG 1px thresh\n')
    # draw_results(results, experiments, iterations_list, title='Phototourism All - LoFTR 3px thresh\n')



if __name__ == '__main__':
    draw_pt()
