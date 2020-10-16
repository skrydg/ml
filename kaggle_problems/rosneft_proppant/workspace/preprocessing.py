import copy
from pathlib import Path
from helpers import *
from common import *

class Processor():
    def __init__(self, img, img_number):
        self.debug = False
        self.img_number = img_number
        self.img = img
        self.origin_img = copy.deepcopy(img)

    def with_debug(self, debug_dir):
        self.debug = True
        self.debug_dir = debug_dir
        Path(self.debug_dir).mkdir(exist_ok=True, parents=True)

    def increase_brightness(self, img, value):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img

    def check_size_contour(self, img, contour, down, up):
        area = cv2.contourArea(contour)
        return area > 1000 and \
               area / (img.shape[0] * img.shape[1]) >= down and \
               area / (img.shape[0] * img.shape[1]) <= up

    def is_rectange_area(self, contour):
        epsilon = 0.01 * cv2.arcLength(contour, True)
        res = cv2.approxPolyDP(contour, epsilon, True)
        return len(res) == 4

    def is_contours_simular(self, a, b, eps=0.1):
        return (abs(cv2.contourArea(a) - cv2.contourArea(b)) / (cv2.contourArea(a) + cv2.contourArea(b)) < eps) and \
               (abs(cv2.arcLength(a, True) - cv2.arcLength(b, True)) / (
                           cv2.arcLength(a, True) + cv2.arcLength(b, True)) < eps)

    def get_main_contour(self, thresh, find_inner):
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        hierarchy = hierarchy[0]
        origin_contours = copy.deepcopy(contours)
        contours = [(contour, i) for contour, i in zip(contours, range(len(contours)))]

        # filters
        contours = [(contour, i) for (contour, i) in contours if self.check_size_contour(thresh, contour, 0.1, 0.5)]
        contours = [(contour, i) for (contour, i) in contours if self.is_rectange_area(contour)]

        if (find_inner):
            set_of_contours = set([i for c, i in contours])
            contours = [(contour, i) for (contour, i) in contours if
                        hierarchy[i][-1] in set_of_contours]  # stay only if parent is rectangle
            contours = [(contour, i) for (contour, i) in contours if
                        self.is_contours_simular(contour, origin_contours[hierarchy[i][-1]])]

        if (len(contours) != 1):
            return None

        return contours[0][0]

    def transform(self, img, contour):
        epsilon = 0.01 * cv2.arcLength(contour, True)
        contour = cv2.approxPolyDP(contour, epsilon, True)

        contour = np.float32([[i[0][0], i[0][1]] for i in contour])

        l = round(dist(contour[0], contour[1]))
        h = round(dist(contour[1], contour[2]))

        needed_contour = np.float32([[0, 0], [0, l], [h, l], [h, 0]])

        M = cv2.getPerspectiveTransform(contour, needed_contour)
        transformed = cv2.warpPerspective(img, M, (h, l))

        if transformed.shape[0] < transformed.shape[1]:
            transformed = cv2.rotate(transformed, cv2.ROTATE_90_CLOCKWISE)

        assert (transformed.shape[0] >= transformed.shape[1])

        return transformed

    def resize(self, img):
        # изменение пропорций картинки
        return cv2.resize(img, (TARGET_SHAPE[1], TARGET_SHAPE[0]))

    def process(self):
        #############################################################################################
        self.img = self.increase_brightness(self.img, 100)
        if self.debug:
            cv2.imwrite("{}/img_with_brightness.jpg".format(self.debug_dir), self.img)
        #############################################################################################
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        if self.debug:
            cv2.imwrite("{}/gray.jpg".format(self.debug_dir), self.gray)

        _, self.thresh = cv2.threshold(self.gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        if self.debug:
            cv2.imwrite("{}/thresh.jpg".format(self.debug_dir), self.thresh)
        #############################################################################################
        is_inner = True
        contour = self.get_main_contour(self.thresh, find_inner=is_inner)
        if contour is None:
            is_inner = False
            contour = self.get_main_contour(self.thresh, find_inner=is_inner)
            print("find outer contour for img_name: {}", self.img_number)

        if contour is None:
            print("ERROR: {}", self.img_number)
            return None
        #############################################################################################

        if self.debug:
            img_with_main_contour = copy.deepcopy(self.img)
            epsilon = 0.01 * cv2.arcLength(contour, True)
            res = cv2.approxPolyDP(contour, epsilon, True)
            cv2.drawContours(img_with_main_contour, [res], -1, get_random_color(), 10)
            cv2.imwrite("{}/img_with_main_contour.jpg".format(self.debug_dir), img_with_main_contour)

        #############################################################################################

        self.img = self.transform(self.origin_img, contour)
        if self.debug:
            cv2.imwrite("{}/after_transorm.jpg".format(self.debug_dir), self.img)
        #############################################################################################

        if not is_inner:
            x_diff = round((OUTER_SHAPE[0] - INNER_SHAPE[0]) / OUTER_SHAPE[0] * self.img.shape[0] / 2)
            y_diff = round((OUTER_SHAPE[1] - INNER_SHAPE[1]) / OUTER_SHAPE[1] * self.img.shape[1] / 2)

            self.img = self.img[x_diff:-x_diff, y_diff:-y_diff]

        #############################################################################################
        print(self.img.shape)
        self.img = self.resize(self.img)
        print(self.img.shape)
        if self.debug:
            cv2.imwrite("{}/main_area.jpg".format(self.debug_dir), self.img)
        return self.img