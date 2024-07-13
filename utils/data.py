import seaborn as sns

experiments = ['4p3v(M)', '4p3v(M) + R', '4p3v(M) + R + C',
               '4p3v(M+D)', '4p3v(M+D) + R', '4p3v(M+D) + R + C',
               '4p(HC)', '5p3v', '4p3v(O) + R + C']

iterations_list = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
colors = {exp: sns.color_palette("tab10")[i] for i, exp in enumerate(experiments)}
styles = {exp: 'dashed' if 'O' in exp else 'solid' for exp in experiments}
colors[experiments[7]] = sns.color_palette("tab10")[9]

basenames_pt = ['brandenburg_gate', 'buckingham_palace', 'colosseum_exterior', 'grand_place_brussels',
                'notre_dame_front_facade', 'palace_of_westminster', 'pantheon_exterior', 'reichstag',
                'sacre_coeur', 'st_peters_square', 'taj_mahal', 'temple_nara_japan', 'trevi_fountain']
# basenames_pt = ['sacre_coeur']
# basenames_eth = ['courtyard', 'delivery_area', 'electro', 'facade', 'kicker', 'list.py', 'meadow', 'office', 'pipes', 'playground', 'relief', 'relief_2', 'terrace', 'terrains']
basenames_eth = ['courtyard', 'delivery_area', 'electro', 'kicker', 'meadow', 'office', 'pipes', 'playground', 'relief', 'relief_2', 'terrace', 'terrains']
basenames_cambridge = ['GreatCourt', 'KingsCollege', 'ShopFacade', 'StMarysChurch', 'OldHospital']


basenames_aachen = ['aachen_v1.1']


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
    else:
        raise ValueError
    return basenames
