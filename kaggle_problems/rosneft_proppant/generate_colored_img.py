#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
while not os.getcwd().endswith('ml'):
    os.chdir('..')
sys.path.insert(0, os.getcwd())


# In[2]:


from kaggle_problems.rosneft_proppant.workspace.common import TARGET_SHAPE
from kaggle_problems.rosneft_proppant.workspace.helpers import get_random_color
from kaggle_problems.rosneft_proppant.workspace.common import r2prop_size, bins2mm, generate_low_high
from kaggle_problems.rosneft_proppant.RPCC_metric_utils_for_participants import sizes_to_sieves, get_bins_from_granules, sizes_to_sieve_hist
from pathlib import Path
import random
import numpy as np
from matplotlib import pyplot as plt
import cv2
import collections
import pandas as pd
import math


# In[3]:


GENERATED_DIR = "kaggle_problems/rosneft_proppant/data/generated/"
GENERATED_IMG_DIR = GENERATED_DIR + "colored_img"
GENERATED_LABELS_DIR = GENERATED_DIR + "labels"
Path(GENERATED_IMG_DIR).mkdir(exist_ok=True, parents=True)
Path(GENERATED_LABELS_DIR).mkdir(exist_ok=True, parents=True)


# In[4]:


# img = np.zeros(shape=(11, 11, 3))

# cv2.imwrite("{}/{}.jpg".format("kaggle_problems/rosneft_proppant/data/colored_circles", '100500_5.jpg'), img)


# In[5]:


class CircleGetter:
    CIRCLE_DIR = "kaggle_problems/rosneft_proppant/data/colored_circles"
    def __init__(self):
        random.seed()
        self.r2circle = collections.defaultdict(list)
        all_img = [img for img in os.listdir(self.CIRCLE_DIR) if img.endswith('.jpg')]
        all_img = [cv2.imread("{}/{}".format(self.CIRCLE_DIR, img_name)) for img_name in all_img]
        for img in all_img:
            if self.is_gray_circle(img) and ( not img.shape[0] <= 10):
                continue
            assert(img.shape[0] == img.shape[1])
            r = (img.shape[0] - 1) // 2
            self.r2circle[r].append(img)
        
        for key in sorted(self.r2circle.keys()):
            print(key, len(self.r2circle[key]))
        
    def is_gray_circle(self, image):
        gr = np.absolute(np.array(image[:, :, 0], dtype=int) - np.array(image[:, :, 1], dtype=int))
        gr += np.absolute(np.array(image[:, :, 1], dtype=int) - np.array(image[:, :, 2], dtype=int))
        gr += np.absolute(np.array(image[:, :, 2], dtype=int) - np.array(image[:, :, 0], dtype=int))
        
        return np.percentile(gr, 70) < 100
    
    def get_circle(self, r):
        cnt = len(self.r2circle[r])
        assert(cnt > 0)
        i = random.randint(0, cnt - 1)
        return self.r2circle[r][i]
    
    def get_r(self, r_mi, r_ma):
        r_ma += 1
        r_mi = math.floor(r_mi)
        r_ma = math.ceil(r_ma)
        
        total = np.sum([len(self.r2circle[r]) for r in range(r_mi, r_ma)])
       # print([len(self.r2circle[r]) / total for r in range(r_mi, r_ma)])
        res = np.random.choice([r for r in range(r_mi, r_ma)], 1,
              p=[len(self.r2circle[r]) / total for r in range(r_mi, r_ma)])[0]
        return res
    
circleGetter = CircleGetter()


# In[6]:


def in_range(l, s, r):
    return l <= s and s < r

class GrayCircleContour:
    def __init__(self):
        self.msk = []
        for r in np.arange(0, 100):
            img = np.zeros(shape=(2 * r + 1, 2 * r + 1), dtype=np.uint8)
            cv2.circle(img, (r, r), r, 1, -1)

            self.msk.append(img)


    def get_msk(self, r):
        assert(r >= 1 and r < 100)

        return self.msk[r]

circleContour = GrayCircleContour()

def get_masked_img(img, x, y, r):
    x_min = x - r
    x_max = x + r + 1
    y_min = y - r
    y_max = y + r + 1
    if (not in_range(0, x_min, img.shape[0])) or             (not in_range(0, x_max, img.shape[0])) or             (not in_range(0, y_min, img.shape[1])) or             (not in_range(0, y_max, img.shape[1])):
        return None

    msk = circleContour.get_msk(r)
    sub_img = img[x_min:x_max, y_min:y_max]

    sub_img = (sub_img * msk).astype(dtype=int)
    return sub_img


# In[7]:


def get_empty_img():
    img = np.empty(shape=(int(TARGET_SHAPE[0] / 1.5), int(TARGET_SHAPE[1] / 1.5), 3), dtype=int)
    img[:, :, :] = 0
    return img


def draw_background(img, msk):
    x_background, y_background = np.where(msk == 0)
    grays = np.random.uniform(low=[170, 170, 170], high=[200, 200, 200], size=(len(x_background), 3))
    img[x_background, y_background] = grays
    return img


# In[8]:


def generate_low_high(bins_name):
    bin2low = {b: [] for b in bins_name}
    bin2high = {b: [] for b in bins_name}
    for r in range(1, 100):
        sieves, names = sizes_to_sieves([r2prop_size(r)], [bins2mm[b] for b in bins_name], bins_name)
        assert(len(names) == 1)
        if int(names[0]) == 0:
            continue
        bin2low[str(int(names[0]))].append(r)
        bin2high[str(int(names[0]))].append(r)
       
    bin2low = {k: min(v) for k,v in bin2low.items()}
    bin2high = {k: max(v) for k,v in bin2high.items()}
    
    return bin2low, bin2high


# In[ ]:


bin2mean = {
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

bin2std = {
    "16/20": {
"16":              0.000700,
"18":              0.009249,
"20":              0.008222,
"25":              0.002565,
"30":              0.000234,
"40":              0.000143,
    },
    "20/40_pdcpd_bash_lab": {
"16":              0.001335,
"18":              0.001203,
"20":              0.014613,
"25":              0.050904,
"30":              0.007153,
"40":              0.034531,
    },
}

bin2low, bin2high = generate_low_high(bin2mean["16/20"].keys())


# In[ ]:





# In[ ]:


print(bin2low, bin2high)


# In[ ]:


def generate_img(_mean, _std):
    bin2normal = {key: (lambda key: float(np.random.normal(_mean[key], 2 * _std[key], 1)[0])) for key in _mean.keys()}

    PERSENT_FREE = random.randint(20, 80) / 100
    n = random.randint(2, 10)
    xy_min = [TARGET_SHAPE[0] * 0.2, TARGET_SHAPE[1] * 0.2]
    xy_max = [TARGET_SHAPE[0] * 0.8, TARGET_SHAPE[1] * 0.8]
    means = np.random.uniform(low=xy_min, high=xy_max, size=(n,2))
    
    all_circles = []
    img = get_empty_img()
    for (x, y) in means:
        CNT_CIRCLES = random.randint(1000, 5000)
        color = get_random_color()
        centers = np.random.multivariate_normal([x, y], [[TARGET_SHAPE[0] ** 2 / 100, 0], [0, TARGET_SHAPE[1] ** 2 / 100]], size=CNT_CIRCLES)

        radius = []
        for b, pers in bin2normal.items():
            cnt_circle = max(0, int(round(CNT_CIRCLES * pers(b))))
            radius.extend([(bin2low[b], bin2high[b])] * cnt_circle)
            
        for center, r in zip(centers, radius):
            all_circles.append((int(center[0]), int(center[1]), r))
    random.shuffle(all_circles)

    msk = np.zeros(shape=img.shape[0:2], dtype=np.uint8)
    img = get_empty_img()
    filtered_circles = []
    
    for circle in all_circles:
        x, y, (r_mi, r_ma) = circle
        
        r = circleGetter.get_r(r_mi, r_ma)
        
        sub_img = get_masked_img(msk, x, y, r)
        
        circle_msk = circleContour.get_msk(r)

        if sub_img is None:
            continue

        cnt_free = np.sum(np.logical_and(sub_img == 0, circle_msk == 1))
        s = np.sum(circle_msk == 1)

        if (cnt_free / s < PERSENT_FREE):
            continue

        circle = circleGetter.get_circle(r)
        circle *= circle_msk[:, :, np.newaxis]
        
        msk[x - r: x + r + 1, y - r: y + r + 1] += circle_msk
        img[x - r: x + r + 1, y - r: y + r + 1] *= (1 - circle_msk[:, :, np.newaxis])
        img[x - r: x + r + 1, y - r: y + r + 1] += circle

        filtered_circles.append((x, y, r))

    img = draw_background(img, msk).astype(np.uint8)
    
    img = cv2.resize(img, (TARGET_SHAPE[1] // 4, TARGET_SHAPE[0] // 4))
    img = cv2.resize(img, (int(TARGET_SHAPE[1] / 1.5), int(TARGET_SHAPE[0] / 1.5)))
    return img, filtered_circles


# In[ ]:


result_bins = []
fraction_it = 0
CNT_IMG = 100
for fraction in ["16/20", "20/40_pdcpd_bash_lab"]:
    for i in range(CNT_IMG):
        print(i)
        img, circles = generate_img(bin2mean[fraction], bin2std[fraction])
        prop_sizes = [r2prop_size(r) for (_, _, r) in circles]

        bins_names = [i for i in bin2low.keys()]
        bins_mm = [bins2mm[key] for key in bins_names]
        bins_names += [0]
        bins_mm += [0]
        cur_bins = sizes_to_sieve_hist(pd.DataFrame({"prop_size": prop_sizes}), bins_mm, bins_names)
        cur_bins['ImageId'] = i + 1 + fraction_it * CNT_IMG
        cur_bins['fraction'] = fraction

        result_bins.append(cur_bins)
        cv2.imwrite("{}/{}.jpg".format(GENERATED_IMG_DIR, i + 1 + fraction_it * CNT_IMG), img)

        generated_train = pd.DataFrame(data=result_bins)
        generated_train.to_csv(GENERATED_LABELS_DIR + "/generated_colored_train.csv")
    fraction_it += 1


# In[ ]:





# In[ ]:


generated_train.describe()


# In[ ]:


get_ipython().system('jupyter nbconvert --to script kaggle_problems/rosneft_proppant/generate_colored_img.ipynb')


# In[ ]:




