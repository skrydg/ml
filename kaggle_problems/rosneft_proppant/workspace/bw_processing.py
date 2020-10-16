import copy
from pathlib import Path
from helpers import *
from common import *

#bw_bins = ['16', '18', '20', '25', '30', '35', '40', '50']
bw_bins = ['20', '25', '30', '35', '40', '50']


class GrayCircleContour:
    def __init__(self):
        self.msk = []
        for r in np.arange(0, 100):
            img = np.zeros(shape=(2 * r + 1, 2 * r + 1))
            cv2.circle(img, (r, r), r, 1, -1)

            self.msk.append(img)


    def get_msk(self, r):
        assert(r >= 1 and r < 100)

        return self.msk[r]

class Processor:
    circleContour = GrayCircleContour()
    def __init__(self, img, img_number):
        self.debug = False
        self.img_number = img_number
        self.img = img
        self.original_img = copy.deepcopy(img)
        self.img_with_circle = copy.deepcopy(img)

    def with_debug(self, debug_dir):
        self.debug = True
        self.debug_dir = debug_dir
        Path(self.debug_dir).mkdir(exist_ok=True, parents=True)

    def get_masked_img(self, img, x, y, r):
        x_min = x - r
        x_max = x + r + 1
        y_min = y - r
        y_max = y + r + 1
        if (not in_range(0, x_min, img.shape[0])) or \
                (not in_range(0, x_max, img.shape[0])) or \
                (not in_range(0, y_min, img.shape[1])) or \
                (not in_range(0, y_max, img.shape[1])):
            return None

        msk = self.circleContour.get_msk(r)
        sub_img = img[x_min:x_max, y_min:y_max]

        sub_img = (sub_img * msk).astype(dtype=int)
        return sub_img

    def is_convex_in_circle(self, threshed, x, y, r):
        x, y = y, x
        sub_img = self.get_masked_img(threshed, x, y, r)

        if sub_img is None:
            return 0

        x, y = np.where(sub_img == 0)

        x -= r
        y -= r

        min_di = np.min(np.sqrt(x * x + y * y))

        return min_di / r >= 0.8

    def get_count_pixel_in_circle(self, img, x, y, r, pixel):
        sub_img = self.get_masked_img(img, x, y, r)
        if sub_img is None:
            return None
        msk = self.circleContour.get_msk(r)

        cnt = np.sum(np.logical_and(sub_img == pixel, msk == 1))

        return cnt

    def is_border_white_circle(self, threshed, x, y, r):
        x, y = y, x
        cnt_black = self.get_count_pixel_in_circle(threshed, x, y, r, 0)
        cnt_white = self.get_count_pixel_in_circle(threshed, x, y, r, 255)
        if (cnt_black is None) or (cnt_white is None):
            return False
            # return None, None

        cnt_all = cnt_black + cnt_white

        cnt_black1 = self.get_count_pixel_in_circle(threshed, x, y, r + 2, 0)
        cnt_white1 = self.get_count_pixel_in_circle(threshed, x, y, r + 2, 255)
        if (cnt_black1 is None) or (cnt_white1 is None):
            return False
            # return None, None
        cnt_all1 = cnt_black + cnt_white

        diff_black = cnt_black1 - cnt_black
        diff_white = cnt_white1 - cnt_white

        return (diff_black / (diff_black + diff_white)) >= 0.2

    def process(self):
        gaussian_3 = cv2.GaussianBlur(self.img, (0, 0), 2.0)
        self.img = cv2.addWeighted(self.img, 1.5, gaussian_3, -0.5, 0, self.img)
        if self.debug:
            cv2.imwrite("{}/img_after_bluer.jpg".format(self.debug_dir), self.img)

        self.gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        _, self.threshed = cv2.threshold(self.gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        self.threshed_with_circle = cv2.cvtColor(self.threshed, cv2.COLOR_GRAY2RGB)

        if self.debug:
            cv2.imwrite("{}/gray.jpg".format(self.debug_dir), self.gray)
            cv2.imwrite("{}/threshed.jpg".format(self.debug_dir), self.threshed)


        # bw_pixels = sorted([bins_dict[bin_key] for bin_key in bw_bins], reverse=True)
        # bw_pixels = [100] + bw_pixels

        result_circles = []
        for i in range(50, 10, -1):
            l = i - 1
            r = i

            circles = cv2.HoughCircles(self.threshed, cv2.HOUGH_GRADIENT, 0.5, minDist=1.7 * l,
                                       param1=200, param2=1, minRadius=l, maxRadius=r)
            circles = circles[0]

            print("Circle count: {}. r: {}-{}".format(len(circles), l, r))
            filtered_circle = [(int(circle[0]), int(circle[1]), int(circle[2])) for circle in circles]

            filtered_circle = [circle for circle in filtered_circle \
                               if self.is_convex_in_circle(self.threshed, circle[0], circle[1], circle[2])]

            print("After first filter: {}. r: {}-{}".format(len(filtered_circle), l, r))

            filtered_circle = [circle for circle in filtered_circle \
                               if self.is_border_white_circle(self.threshed, circle[0], circle[1], circle[2])]

            print("After second filter: {}. r: {}-{}".format(len(filtered_circle), l, r))

            color = get_random_color()
            for circle in filtered_circle:
                cv2.circle(self.threshed, (circle[0], circle[1]), circle[2], 0, -1)  # fill with 0 because of THRESH_BINARY_INV
                cv2.circle(self.original_img, (circle[0], circle[1]), circle[2], (255, 255, 255), -1)
                cv2.circle(self.img_with_circle, (circle[0], circle[1]), circle[2], color, 1)
                cv2.circle(self.threshed_with_circle, (circle[0], circle[1]), circle[2], color, 1)

            if self.debug:
                cv2.imwrite("{}/original_img_{}.jpg".format(self.debug_dir, i), self.original_img)
                cv2.imwrite("{}/img_with_circle_{}.jpg".format(self.debug_dir, i), self.img_with_circle)
                cv2.imwrite("{}/threshed_with_circle_{}.jpg".format(self.debug_dir, i), self.threshed_with_circle)
                cv2.imwrite("{}/threshed_{}.jpg".format(self.debug_dir, i), self.threshed)

            result_circles.extend(filtered_circle)

        return result_circles