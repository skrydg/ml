import numpy as np
import copy
import cv2
import random

def dist(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def in_range(l, s, r):
    return l <= s and s < r

class ColoredCircleContour:
    def __init__(self):
        self.msk = []
        for r in np.arange(0, 100):
            img = np.zeros(shape=(2 * r + 1, 2 * r + 1, 3))
            cv2.circle(img, (r, r), r, (1, 1, 1), -1)

            self.msk.append(img)


    def get_msk(self, r):
        assert(r >= 1 and r < 100)

        return self.msk[r]



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

def get_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def is_grey_img(image):
    gr = np.absolute(np.array(image[:, :, 0], dtype=int) - np.array(image[:, :, 1], dtype=int))
    gr += np.absolute(np.array(image[:, :, 1], dtype=int) - np.array(image[:, :, 2], dtype=int))
    gr += np.absolute(np.array(image[:, :, 2], dtype=int) - np.array(image[:, :, 0], dtype=int))
    return gr.max() < 300