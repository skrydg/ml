import numpy as np

COEF = 30

MAGIC_COEFF = 0.78

bins = ['6', '7', '8', '10', '12', '14', '16', '18', '20', '25', '30', '35', '40', '45', '50', '60', '70', '80', '100']
bins_mm = [
    3.35,
    2.8,
    2.36,
    2.,
    1.7,
    1.4,
    1.18,
    1.,
    0.85,
    0.71,
    0.6,
    0.5,
    0.425,
    0.355,
    0.3,
    0.25,
    0.212,
    0.18,
    0.15
]
bins_pixel = [b * COEF for b in bins_mm]
bins_dict = {key: value for key, value in zip(bins, bins_pixel)}
bins2mm = {key: value for key, value in zip(bins, bins_mm)}

INNER_SHAPE = (143.5, 86.5)
OUTER_SHAPE = (147.5, 90.5)
TARGET_SHAPE = (round(INNER_SHAPE[0] * COEF), round(INNER_SHAPE[1] * COEF))

def r2prop_size(r):
    return 2 * r / COEF * MAGIC_COEFF

def prop_size2r(prop_size):
    return prop_size * COEF / 2 / MAGIC_COEFF

def sizes_to_sieves(sizes, sive_diam, sieves_names):
    """
    Распределяет предикты по ситам
    """
    sizes_ = np.sort(sizes)
    sieve_bins = np.zeros_like(sizes_)

    for diam, name in zip(sive_diam, sieves_names):
        sieve_bins[sizes_<= diam] = name

    return sizes_, sieve_bins


def generate_low_high(bins_name):
    bin2low = {b: [] for b in bins_name}
    bin2high = {b: [] for b in bins_name}
    for r in range(1, 100):
        sieves, names = sizes_to_sieves([r2prop_size(r)], [bins2mm[b] for b in bins_name], bins_name)
        assert (len(names) == 1)
        if int(names[0]) == 0:
            continue
        bin2low[str(int(names[0]))].append(r)
        bin2high[str(int(names[0]))].append(r)


    bin2low = {k: min(v if len(v) else [None]) for k, v in bin2low.items()}
    bin2high = {k: max(v if len(v) else [None]) for k, v in bin2high.items()}

    return bin2low, bin2high

bin2low, bin2high = generate_low_high(bins)
