import copy
from pathlib import Path
from helpers import *
from common import *
import tensorflow as tf
import json
import math

BINS_NAME = ['16', '18', '20', '25', '30', '40']
BINS_SIZE = [1.18, 1, 0.85, 0.71, 0.6, 0.5]

mean_value = {
    "16/20": {
        "16": 0.008445,
        "18": 0.345696,
        "20": 0.608227,
        "25": 0.035619,
        "30": 0.001163,
        "40": 0.000142
    },
    "20/40_pdcpd_bash_lab": {
        "16": 0.004605,
        "18": 0.007023,
        "20": 0.077243,
        "25": 0.412786,
        "30": 0.253526,
        "40": 0.200304,
    },
}
def decrease_brightness(img, value):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v < value] = 0
    v[v >= value] -= value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

class GrayCircleContour:
    def __init__(self):
        self.msk = []
        self.msk_sum = []
        for r in np.arange(0, 100):
            img = np.zeros(shape=(2 * r + 1, 2 * r + 1))
            cv2.circle(img, (r, r), r, 1, -1)

            self.msk.append(img)
            self.msk_sum.append(np.sum(img))


    def get_msk(self, r):
        return self.msk[r]

    def get_msk_sum(self, r):
        return self.msk_sum[r]

circleContour = GrayCircleContour()

def in_range(l, s, r):
    return l <= s and s < r

def get_masked_img(img, x, y, r):
    x_min = x - r
    x_max = x + r + 1
    y_min = y - r
    y_max = y + r + 1
    if (not in_range(0, x_min, img.shape[0])) or \
            (not in_range(0, x_max, img.shape[0])) or \
            (not in_range(0, y_min, img.shape[1])) or \
            (not in_range(0, y_max, img.shape[1])):
        return None

    msk = circleContour.get_msk(r)
    sub_img = img[x_min:x_max, y_min:y_max]

    sub_img = (sub_img * msk).astype(dtype=int)
    return sub_img

def get_black_circle(th, x, y, r):
    x, y = y, x
    sub_img = get_masked_img(th, x, y, r)
    if (sub_img is None):
        return False
    sub_img = sub_img.astype(float)
    msk_sum = circleContour.get_msk_sum(r)

    sub_img2 = get_masked_img(th, x, y, r - 2)
    if (sub_img2 is None):
        return False
    sub_img2 = sub_img2.astype(float)
    msk_sum2 = circleContour.get_msk_sum(r - 2)

    return (np.sum(sub_img == 0) - np.sum(sub_img2 == 0)) / (msk_sum - msk_sum2)


def is_black_circle(th, x, y, r):
    return get_black_circle(th, x, y, r) > 0.9


def get_flare_circle(th, x, y, r):
    x, y = y, x
    sub_img2 = get_masked_img(th, x, y, r - 2)
    if (sub_img2 is None):
        return False
    sub_img2 = sub_img2.astype(float)
    msk_sum2 = circleContour.get_msk_sum(r - 2)

    return np.sum(sub_img2 == 255) / msk_sum2


def is_flare_circle(th, x, y, r):
    return get_flare_circle(th, x, y, r) > 0.4


def get_not_black_circle(th, x, y, r):
    x, y = y, x
    sub_img = get_masked_img(th, x, y, r)
    if (sub_img is None):
        return False
    sub_img = sub_img.astype(float)
    msk_sum = circleContour.get_msk_sum(r)

    sub_img2 = get_masked_img(th, x, y, r + 2)
    if (sub_img2 is None):
        return False
    sub_img2 = sub_img2.astype(float)
    msk_sum2 = circleContour.get_msk_sum(r + 2)

    return (np.sum(sub_img2 == 0) - np.sum(sub_img == 0)) / (msk_sum2 - msk_sum)


def is_not_black_around_circle(th, x, y, r):
    return get_not_black_circle(th, x, y, r) < 0.5


class Processor:
    def __init__(self, img, img_number):
        self.debug = False
        self.img_number = img_number
        self.img = cv2.resize(img, (int(TARGET_SHAPE[1] / 1.5), int(TARGET_SHAPE[0] / 1.5)))
        self.img = decrease_brightness(self.img, 100)

    def with_debug(self, debug_dir):
        self.debug = True
        self.debug_dir = debug_dir
        Path(self.debug_dir).mkdir(exist_ok=True, parents=True)

    def get_circles(self, gray, min_r, max_r):
        _, th = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 5, minDist=min_r,
                                   param1=200, param2=20, minRadius=min_r, maxRadius=max_r)
        circles = circles[0]
        print("Circle count: {}. r: {}-{}".format(len(circles), min_r, max_r))

        filtered_circle = [(int(circle[0]), int(circle[1]), int(circle[2])) for circle in circles]

        filtered_circle = [circle for circle in filtered_circle if is_black_circle(th, circle[0], circle[1], circle[2])]
        print("After first filter: {}.".format(len(filtered_circle)))

        filtered_circle = [circle for circle in filtered_circle if
                           is_flare_circle(th, circle[0], circle[1], circle[2]) or
                           is_not_black_around_circle(th, circle[0], circle[1], circle[2])]

        print("After second filter: {}.".format(len(filtered_circle)))

        return filtered_circle

    def get_fraction(self):
        gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)

        big_circles = self.get_circles(gray, bin2low['20'] + 1, bin2high['16'])
        small_circles = self.get_circles(gray, bin2low['40'], bin2high['25'] - 1)

        #print("big_circles: {}, small_circles: {}".format(len(big_circles), len(small_circles)))

        if (len(big_circles) > len(small_circles)):
            return '16/20'
        return '20/40_pdcpd_bash_lab'

    def get_prop_count(self, img, bins):
        pred_s = 0

        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        s = np.sum(hsv_img[:, :, 1] > 170)

        for b, diam, persent in zip(BINS_NAME, BINS_SIZE, bins):
            if persent > 1e-5:
                r = (bin2low[b] + bin2high[b]) / 2
                pred_s += (r ** 2) * math.pi * persent

        return s / pred_s

    def process(self):
        fraction = self.get_fraction()
        print("fraction: {}".format(fraction))
        # self.model_img = cv2.resize(self.img, (TARGET_SHAPE[1], TARGET_SHAPE[0]))
        bins = mean_value[fraction].values()
        #self.model.predict(np.array([self.model_img.astype(np.float64)]))[0]

        prop_count = self.get_prop_count(self.img, bins)
        if self.debug:
            with open("{}/predicted_bins".format(self.debug_dir), 'w') as f:
                json.dump([float(i) for i in bins], f)

        assert(len(bins) == len(BINS_NAME))
        result_circles = []
        for b, size in zip(bins, BINS_SIZE):
            cnt = int(round(b * prop_count))
            result_circles.extend([(None, None, (size - 1e-5))] * cnt)
        return result_circles