#!/usr/bin/env python
# coding: utf-8

# In[64]:


import os
import sys
while not os.getcwd().endswith('ml'):
    os.chdir('..')
sys.path.insert(0, os.getcwd())


# In[65]:


import math
import copy
import cv2
import pandas as pd
import numpy as np
import random
from shutil import copyfile
from pathlib import Path
from matplotlib import pyplot as plt

from kaggle_problems.rosneft_proppant.helpers import *

import tensorflow as tf
from tensorflow.keras import Model
from  tensorflow.keras.preprocessing.image import ImageDataGenerator

from ml_helpers.image_helpers import display_images

from kaggle_problems.rosneft_proppant.common import *
from kaggle_problems.rosneft_proppant.RPCC_metric_utils_for_participants import fraction_sievs
from sklearn.model_selection import train_test_split
import pickle


# In[ ]:





# In[66]:


train = pd.read_csv("{}/RPCC_labels.csv".format(DATA_DIR))
train.describe()


# In[ ]:





# In[68]:


fractions = train.fraction.unique()
fractions = fractions[:-1] # delete nana
print(fractions)


# In[69]:


for i in bins:
    train = train[~train[i].isnull()]

train["filename"] = train['ImageId'].astype(str) + '.jpg'


# In[70]:


train['y'] = train.apply(lambda x: np.array([x[i] for i in bins]), axis=1)


# ### Model

# In[47]:


class BinsExtraction(Model):
    def __init__(self, fraction):
        super(BinsExtraction, self).__init__()
        self.FilterSize1 = 32
        self.FilterSize2 = 16
        self.FilterSize3 = 8
        
        self.model_layers = [
            tf.keras.layers.Conv2D(filters=self.FilterSize1, kernel_size=(5, 5), strides=(5, 5)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(rate=0.3),

            tf.keras.layers.Conv2D(filters=self.FilterSize2, kernel_size=(3, 3), strides=(3, 3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(rate=0.3),
            
            tf.keras.layers.Conv2D(filters=self.FilterSize3, kernel_size=(3, 3), strides=(3, 3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(rate=0.3),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(rate=0.5),
            tf.keras.layers.Dense(len(fraction_sievs[fraction]['all']), activation='softmax'),
        ]

    def call(self, x, *args, **kwargs):
        for model_layer in self.model_layers:
            x = model_layer(x, *args, **kwargs)
        return x


# #### Train Input Generator

# In[48]:


def get_train_val_datagen(train, fraction):
    train_fraction = train[train['fraction'] == fraction]
    
    train_fraction, val_fraction = train_test_split(train_fraction, train_size=0.8)
    
    bins_fraction = fraction_sievs[fraction]['all']
    
    datagen = ImageDataGenerator()

    train_generator = datagen.flow_from_dataframe(
            train_fraction.sample(n=int(len(train_fraction) * DF_RATE)),
            directory=TRAIN_DIR,
            x_col='filename', 
            y_col=bins_fraction,
            target_size=TARGET_SHAPE,
            batch_size=16,
            class_mode='other')
    
    val_generator = datagen.flow_from_dataframe(
        val_fraction.sample(n=int(len(val_fraction) * DF_RATE)),
        directory=TRAIN_DIR,
        x_col='filename', 
        y_col=bins_fraction,
        target_size=TARGET_SHAPE,
        batch_size=16,
        class_mode='other')
    return train_generator, val_generator


# #### Input generator checking

# In[49]:


img, labels = get_train_val_datagen(train, '20/40')[0].next()
display_images(img[0:8].astype(int), 4)


# In[50]:


print(labels)


# #### Callbacks

# In[51]:


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# In[52]:


learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

earlystop = EarlyStopping(patience=10)

callbacks = [earlystop, learning_rate_reduction]


# In[75]:


def metric(true, predicted):
    return tf.keras.backend.mean(tf.math.reduce_sum((true - predicted) ** 2 / (true + predicted), axis=1))


# In[76]:


for fraction, i in zip(fractions, range(len(fractions))):
    print(fraction)
    model = BinsExtraction(fraction)
    model.compile(
        loss=metric,
        optimizer='rmsprop',
        metrics=['mse']
    )

    train_datagen, val_datagen = get_train_val_datagen(train, fraction)

    history = model.fit(
        x=train_datagen,
        epochs=30,
        validation_data=val_datagen,
        callbacks=callbacks
    )
    
    with open(MODEL_DIR + "/history_model_benchmark_{}.pickle".format(i), 'wb') as f:
        pickle.dump(history.history, f)
        
    model.save(MODEL_DIR + "/model_benchmark_{}".format(i))


# In[14]:


# model.save(MODEL_DIR + "/cnn_benchmark")


# In[15]:


#model = tf.keras.models.load_model(MODEL_DIR + "/cnn_benchmark")


# In[16]:


#model.summary()


# In[ ]:


#predicted_labels = model.predict(img)


# In[ ]:


# def get_bins_metric(predicted, true):
#     print(predicted.shape)
#     return 0.5 * np.sum((predicted - true) ** 2 / (predicted + true)) / predicted.shape[0]
    


# In[ ]:


# print(get_bins_metric(predicted_labels, labels))


# In[78]:


get_ipython().system('jupyter nbconvert --to script kaggle_problems/rosneft_proppant/cnn_try.ipynb')


# In[ ]:




