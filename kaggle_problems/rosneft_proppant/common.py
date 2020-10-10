DATA_DIR = "kaggle_problems/rosneft_proppant/data"
ORIGINAL_IMG_DIR = "kaggle_problems/rosneft_proppant/data/train"
DEBUG_IMG_DIR = "kaggle_problems/rosneft_proppant/data/debug"

TRAIN_DIR = "kaggle_problems/rosneft_proppant/data/processing/main_area"
MODEL_DIR = "kaggle_problems/rosneft_proppant/models"

bins = ['6', '7', '8', '10', '12', '14', '16', '18', '20', '25', '30', '35', '40', '45', '50', '60', '70', '80', '100']
DF_RATE = 1.

INNER_SHAPE = (143.5, 86.5)
OUTER_SHAPE = (147.5, 90.5)
TARGET_SHAPE = (round(INNER_SHAPE[0] * 30), round(INNER_SHAPE[1] * 30))
