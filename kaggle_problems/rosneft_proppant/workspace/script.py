import cv2
import os
from pathlib import Path
import datetime
import helpers

import preprocessing
import bw_processing
import colored_processing
import submition

DEBUG = False
DEBUG_IMG_DIR = './data/debug'
ORIGINAL_IMG_DIR = './data/test'
TRAIN_DIR = './data/main_area'
DATA_DIR = './data'
FILE_TO_SUBMIT = "answers.csv"
MODEL_DIR = './models'

all_img = [img for img in os.listdir(ORIGINAL_IMG_DIR) if img.endswith('.jpg')]
start = datetime.datetime.now()
all_img = all_img[:10]
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

    if DEBUG:
        if is_gray:
            dir = "{}/bw_main_area".format(DATA_DIR)
            Path(dir).mkdir(exist_ok=True, parents=True)
            cv2.imwrite(dir + "/" + img_name, main_area)

            gray = cv2.cvtColor(main_area, cv2.COLOR_RGB2GRAY)
            _, threshed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            dir = "{}/threshed_main_area".format(DATA_DIR)
            Path(dir).mkdir(exist_ok=True, parents=True)
            cv2.imwrite(dir + "/" + img_name, threshed)
        else:
            dir = "{}/colored_main_area".format(DATA_DIR)
            Path(dir).mkdir(exist_ok=True, parents=True)
            cv2.imwrite(dir + "/" + img_name, main_area)

    result_circles = []
    ############################ BW processing ############################
    if is_gray:
        pr = bw_processing.Processor(main_area, img_number)
        if DEBUG:
            pr.with_debug(debug_dir + "/bw_processing")
        result_circles = pr.process()

    ############################ COLORED processing ############################
    if not is_gray:
        pr = colored_processing.Processor(main_area, img_number)
        if DEBUG:
            pr.with_debug(debug_dir + "/colored_processing")
        result_circles = pr.process()

    ############################ submit ############################
    sb = submition.Submition(img_number, FILE_TO_SUBMIT)
    sb.submit(result_circles)


finish = datetime.datetime.now()

print("Total time: {}".format(finish - start))
print("For one img: {}".format((finish - start).total_seconds() / len(all_img)))
