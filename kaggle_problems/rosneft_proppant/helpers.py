import numpy as np
import copy
import cv2
import random

def dist(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def in_range(l, s, r):
    return l <= s and s < r

class CircleContour:
    def __init__(self):
        self.msk = []
        for r in np.arange(0, 20):
            img = np.zeros(shape=(2 * r + 1, 2 * r + 1))
            cv2.circle(img, (r, r), r, 1, -1)

            self.msk.append(img)


    def get_msk(self, r):
        assert(r >= 1 and r < 20)

        return self.msk[r]


circleContour = CircleContour()

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

def get_count_pixel_in_circle(img, x, y, r, pixel):
    sub_img = get_masked_img(img, x, y, r)
    if sub_img is None:
        return None
    msk = circleContour.get_msk(r)

    cnt = np.sum(np.logical_and(sub_img == pixel, msk == 1))

    return cnt


def is_white_circle(threshed, x, y, r):
    x, y = y, x
    cnt_black = get_count_pixel_in_circle(threshed, x, y, r, 0)
    cnt_white = get_count_pixel_in_circle(threshed, x, y, r, 255)
    # print(cnt_black, cnt_white)
    if (cnt_black is None) or (cnt_white is None):
        return False
    cnt_all = cnt_black + cnt_white

    if (cnt_white / cnt_all >= 0.8):
        return True
    return False


def is_convex_in_circle(threshed, x, y, r):
    x, y = y, x
    sub_img = get_masked_img(threshed, x, y, r)

    if sub_img is None:
        return 0

    x, y = np.where(sub_img == 0)

    x -= r
    y -= r

    min_di = np.min(np.sqrt(x * x + y * y))

    return min_di / r >= 0.8


def is_border_white_circle(threshed, x, y, r):
    x, y = y, x
    cnt_black = get_count_pixel_in_circle(threshed, x, y, r, 0)
    cnt_white = get_count_pixel_in_circle(threshed, x, y, r, 255)
    if (cnt_black is None) or (cnt_white is None):
        return False
        # return None, None

    cnt_all = cnt_black + cnt_white

    cnt_black1 = get_count_pixel_in_circle(threshed, x, y, r + 2, 0)
    cnt_white1 = get_count_pixel_in_circle(threshed, x, y, r + 2, 255)
    if (cnt_black1 is None) or (cnt_white1 is None):
        return False
        # return None, None
    cnt_all1 = cnt_black + cnt_white

    diff_black = cnt_black1 - cnt_black
    diff_white = cnt_white1 - cnt_white

    return (diff_black / (diff_black + diff_white)) >= 0.2
    # return diff_black, diff_white

def get_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def is_grey_img(image):
    gr = np.absolute(np.array(image[:, :, 0], dtype=int) - np.array(image[:, :, 1], dtype=int))
    gr += np.absolute(np.array(image[:, :, 1], dtype=int) - np.array(image[:, :, 2], dtype=int))
    gr += np.absolute(np.array(image[:, :, 2], dtype=int) - np.array(image[:, :, 0], dtype=int))
    return gr.max() < 300