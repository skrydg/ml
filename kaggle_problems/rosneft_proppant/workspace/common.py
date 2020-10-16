COEF = 30

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

INNER_SHAPE = (143.5, 86.5)
OUTER_SHAPE = (147.5, 90.5)
TARGET_SHAPE = (round(INNER_SHAPE[0] * COEF), round(INNER_SHAPE[1] * COEF))
