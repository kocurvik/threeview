import seaborn as sns
#
experiments = ['4p3v(M)', '4p3v(M) + R', '4p3v(M) + R + C', '4p3v(M) + C',
               '4p3v(M) + ENM', '4p3v(M) + R + C + ENM',
               '4p3v(M-D)', '4p3v(M-D) + R', '4p3v(M-D) + R + C', '4p3v(M-D) + C',
               '4p(HC)', '5p3v', '5p3v + ENM', '4p3v(O)', '4p3v(O) + R', '4p3v(O) + R + C',
               '4p3v(A)', '4p3v(A) + R + C', '3p3v(A)', '2p3v(A)',
               '4p3v(A) + ENM', '4p3v(A) + R + C + ENM', '3p3v(A) + ENM', '2p3v(A) + ENM']

# experiments = ['4p3v(M) + R + C',
#                '4p3v(M-D) + R + C',
#                '4p(HC)', '5p3v', '4p3v(O) + R + C',
#                '4p3v(A) + R + C', '3p3v(A)', '2p3v(A)']

iterations_list = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
# colors = {exp: sns.color_palette("tab10")[i] for i, exp in enumerate(experiments)}
colors = {exp: sns.color_palette("hls", len(experiments))[i] for i, exp in enumerate(experiments)}
print('******')
print('Color palette:')
print(sns.color_palette("hls", len(experiments)).as_hex())


# twoview_experiments = ['5pE', '5pE + ELM', '5pE + ENM',
#                    '4pE(M)', '4pE(M) + ELM', '4pE(M) + ENM',
#                    '4pF(A)', '3pH(A)', '2pE(A)',
#                    '4pF(A) + ENM', '3pH(A) + ENM', '2pE(A) + ENM',
#                    '4pH', '4pH + ENM', '4pH + ELM']

# twoview_experiments = ['5pE', '5pE', '5pE + ENM',
#                    '4pE(M)', '4pE(M)', '4pE(M) + ENM',
#                    '4pF(A)', '3pH(A)', '2pE(A)',
#                    '4pF(A) + ENM', '3pH(A) + ENM', '2pE(A) + ENM',
#                    '4pH', '4pH + ENM']

twoview_experiments = ['5pE', '4pE(M)', '4pE(M-D)', '4pF(A)', '3pH(A)', '2pE(A)', '4pH']

styles = {exp: 'dashed' if 'O' in exp else 'solid' for exp in experiments}
# colors[experiments[7]] = sns.color_palette("tab10")[9]

basenames_pt = ['brandenburg_gate', 'buckingham_palace', 'colosseum_exterior', 'grand_place_brussels',
                'notre_dame_front_facade', 'palace_of_westminster', 'pantheon_exterior', 'reichstag', #'st_peters_square',
                'sacre_coeur', 'taj_mahal', 'temple_nara_japan', 'trevi_fountain']
# basenames_pt = ['sacre_coeur']
# basenames_eth = ['courtyard', 'delivery_area', 'electro', 'facade', 'kicker', 'list.py', 'meadow', 'office', 'pipes', 'playground', 'relief', 'relief_2', 'terrace', 'terrains']
basenames_eth = ['courtyard', 'delivery_area', 'electro', 'kicker', 'meadow', 'office', 'pipes', 'playground', 'relief', 'relief_2', 'terrace', 'terrains']
basenames_eth_test = ['botanical_garden', 'boulders', 'bridge', 'door', 'exhibition_hall', 'lecture_room', 'living_room', 'lounge', 'observatory', 'old_computer', 'statue', 'terrace_2']
basenames_cambridge = ['GreatCourt', 'KingsCollege', 'ShopFacade', 'StMarysChurch', 'OldHospital']


basenames_aachen = ['aachen_v1.1']


def err_fun_main(out):
    return max(0.5 * (out['R_12_err'] + out['R_13_err']), 0.5 * (out['t_12_err'] + out['t_13_err']))

def err_fun_max(out):
    return max([out['R_12_err'], out['R_13_err'], out['R_23_err'], out['t_12_err'], out['t_13_err'], out['t_23_err']])

def err_twoview(out):
    return max([out['R_err'], out['t_err']])

def get_basenames(dataset):
    if dataset == 'pt':
        basenames = basenames_pt
        name = '\\Phototourism'
    elif dataset == 'eth3d':
        basenames = basenames_eth
        name = '\\ETH'
    elif dataset == 'aachen':
        basenames = basenames_aachen
    elif dataset == 'urban':
        name = 'Urban'
        basenames = ['kyiv-puppet-theater']
    elif dataset == 'cambridge':
        basenames = basenames_cambridge
    elif dataset == 'eth3d_test':
        basenames = basenames_eth_test
    elif dataset == 'indoor6':
        basenames = ['scene1', 'scene2a', 'scene3', 'scene4a', 'scene5', 'scene6']
    else:
        raise ValueError
    return basenames
