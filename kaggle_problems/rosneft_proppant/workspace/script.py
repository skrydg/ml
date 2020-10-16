import cv2
import os
from pathlib import Path
import datetime
import helpers

import preprocessing
import bw_processing
import submition

DEBUG = False
DEBUG_IMG_DIR = './data/debug'
ORIGINAL_IMG_DIR = './data/test'
TRAIN_DIR = './data/main_area'
FILE_TO_SUBMIT = "answers.csv"

all_img = [img for img in os.listdir(ORIGINAL_IMG_DIR) if img.endswith('.jpg')]
start = datetime.datetime.now()

try:
    os.remove(FILE_TO_SUBMIT)
except:
    pass

for img_name in all_img:
    print(img_name)
    ############################ Img reading ############################
    img_number, _ = img_name.split('.')
    debug_dir = "{}/{}".format(DEBUG_IMG_DIR, img_number)

    img = cv2.imread("{}/{}".format(ORIGINAL_IMG_DIR, img_name))

    ############################ Preprocessing ############################
    pr = preprocessing.Processor(img, img_name)
    if DEBUG:
        pr.with_debug(debug_dir + "/preprocessing")

    main_area = pr.process()
    if main_area is None:
        continue

    if DEBUG:
        Path(TRAIN_DIR).mkdir(exist_ok=True, parents=True)
        cv2.imwrite("{}/{}".format(TRAIN_DIR, img_name), main_area)

    ############################ Is gray ############################
    is_gray = helpers.is_grey_img(main_area)

    result_circles = []
    ############################ BW processing ############################
    if is_gray:
        pr = bw_processing.Processor(main_area, img_number)
        if DEBUG:
            pr.with_debug(debug_dir + "/bw_processing")
        result_circles = pr.process()

    ############################ submit ############################
    sb = submition.Submition(img_number, FILE_TO_SUBMIT)
    sb.submit(result_circles)


finish = datetime.datetime.now()

print("Total time: {}".format(finish - start))
print("For one img: {}".format((finish - start).total_seconds() / len(all_img)))
