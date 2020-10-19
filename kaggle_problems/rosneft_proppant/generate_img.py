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
from kaggle_problems.rosneft_proppant.workspace.common import r2prop_size, bins2mm
from kaggle_problems.rosneft_proppant.RPCC_metric_utils_for_participants import get_bins_from_granules, sizes_to_sieve_hist
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


class CircleGetter:
    CIRCLE_DIR = "kaggle_problems/rosneft_proppant/data/circles"
    def __init__(self):
        random.seed()
        self.r2circle = collections.defaultdict(list)
        all_img = [img for img in os.listdir(self.CIRCLE_DIR) if img.endswith('.jpg')]
        all_img = [cv2.imread("{}/{}".format(self.CIRCLE_DIR, img_name)) for img_name in all_img]
        for img in all_img:
            if self.is_gray_circle(img):
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
        
        return np.percentile(gr, 70) < 200
    
    def get_circle(self, r):
        cnt = len(self.r2circle[r])
        assert(cnt > 0)
        i = random.randint(0, cnt - 1)
        return self.r2circle[r][i]
    
    def get_r(self, r_mi, r_ma):
        r_mi = math.floor(r_mi)
        r_ma = math.ceil(r_ma)
        total = np.sum([len(self.r2circle[r]) for r in range(r_mi, r_ma)])
        res = np.random.choice([r for r in range(r_mi, r_ma)], 1,
              p=[len(self.r2circle[r]) / total for r in range(r_mi, r_ma)])[0]
        return res
    
circleGetter = CircleGetter()


# In[5]:


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


# In[6]:


def get_empty_img():
    img = np.empty(shape=(TARGET_SHAPE[0], TARGET_SHAPE[1], 3), dtype=int)
    img[:, :, :] = 0
    return img


def draw_background(img, msk):
    x_background, y_background = np.where(msk == 0)
    grays = np.random.uniform(low=[170, 170, 170], high=[200, 200, 200], size=(len(x_background), 3))
    img[x_background, y_background] = grays
    return img


# In[7]:


bin2low = {'16': 30.0 / 2 + 1e-5, '18': 25.5 / 2 + 1e-5, '20': 21.3 / 2 + 1e-5, '25': 18.0 / 2 + 1e-5, '30': 15.0 / 2, '40': 12.75 / 2}
bin2high = {'16': 35.5 / 2,       '18': 30.0 / 2,        '20': 25.5 / 2,        '25': 21.3 / 2,        '30': 18.0 / 2, '40': 15.0 / 2}

bin2mean = {'16': 0.005181, '18': 0.057824, '20': 0.156891, '25': 0.356211, '30': 0.215671, '40': 0.170280}
bin2std = {'16': 0.001865, '18': 0.121420, '20': 0.190781, '25': 0.143069, '30': 0.090675, '40': 0.078468}

bin2normal = {key: (lambda key: float(np.random.normal(bin2mean[key], bin2std[key] ** 2, 1)[0])) for key in bin2mean.keys()}


# In[8]:


def generate_img():
    PERSENT_FREE = random.randint(20, 80) / 100
    n = random.randint(2, 15)
    xy_min = [TARGET_SHAPE[0] * 0.2, TARGET_SHAPE[1] * 0.2]
    xy_max = [TARGET_SHAPE[0] * 0.8, TARGET_SHAPE[1] * 0.8]
    means = np.random.uniform(low=xy_min, high=xy_max, size=(n,2))
    
    all_circles = []
    img = get_empty_img()
    for (x, y) in means:
        CNT_CIRCLES = random.randint(1000, 10000)
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
    img = cv2.resize(img, (TARGET_SHAPE[1], TARGET_SHAPE[0]))
    return img, filtered_circles


# In[9]:


result_bins = []
for i in range(2000):
    img, circles = generate_img()
    prop_sizes = [r2prop_size(r) for (_, _, r) in circles]
    
    bins_names = [i for i in bin2low.keys()]
    bins_mm = [bins2mm[key] for key in bins_names]
    bins_names += [0]
    bins_mm += [0]
    cur_bins = sizes_to_sieve_hist(pd.DataFrame({"prop_size": prop_sizes}), bins_mm, bins_names)
    cur_bins['ImageId'] = i + 1
    
    result_bins.append(cur_bins)
    cv2.imwrite("{}/{}.jpg".format(GENERATED_IMG_DIR, i + 1), img)


# In[10]:


generated_train = pd.DataFrame(data=result_bins)
generated_train.to_csv(GENERATED_LABELS_DIR + "/generated_train.csv")


# In[11]:


generated_train.describe()


# In[12]:


get_ipython().system('jupyter nbconvert --to script kaggle_problems/rosneft_proppant/generate_img.ipynb')


# In[ ]:




