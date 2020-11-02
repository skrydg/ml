#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
while not os.getcwd().endswith('ml'):
    os.chdir('..')
sys.path.insert(0, os.getcwd())

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# In[2]:


import math
import copy
import cv2
import pandas as pd
import numpy as np
import random
from shutil import copyfile
from pathlib import Path
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from ml_helpers.image_helpers import display_images
from ml_helpers.common_helpers import display_history_metrics

from sklearn.model_selection import train_test_split
import pickle

from kaggle_problems.rosneft_proppant.workspace.common import bins, bin2high, TARGET_SHAPE, SUB_IMG_CNT


# In[3]:


DATA_DIR = "kaggle_problems/rosneft_proppant/data/"
MODEL_DIR = "kaggle_problems/rosneft_proppant/workspace/models"
GENERATED_DIR = "kaggle_problems/rosneft_proppant/data/generated/"
GENERATED_LABELS_DIR = GENERATED_DIR + "labels"
DF_RATE = 1

sources = ['bw'] #'colored']
source_to_fraction = {
    'bw': 'bw',
    'colored': 'colored',
    'threshed': 'bw'
}

fraction_sievs = {
    'bw': ['16', '18', '20', '25', '30', '35', '40']
}

COEF_COMPRESS = 4


# In[4]:


def enrich_fraction(train):
    for fraction in source_to_fraction.values():
        img_numbers = [int(img[0:-len(".jpg")]) for img in os.listdir(DATA_DIR + fraction + "_main_area") if img.endswith('.jpg')]
        train.loc[train.ImageId.isin(img_numbers), 'fraction'] = fraction
    return train

def get_fraction_sievs(data, fraction):
    data_fraction = data[data.fraction == fraction]
    result_bins = []
    for b in bins:
        if data_fraction[b].sum() > 1e-5:
            result_bins.append(b)
    return result_bins
        

def common_df_processing(data):
    for i in bins:
        if i in data.columns:
            data = data[~data[i].isnull()]

    data["filename"] = data['ImageId'].astype(str) + '.jpg'
    return data

def get_validation(source):
    validation = pd.read_csv("{}labels/train.csv".format(DATA_DIR))
    
    validation.fraction = None
    validation = enrich_fraction(validation)

    validation = validation[~validation.fraction.isnull()]
    
    fraction = source_to_fraction[source]
    validation = validation[validation['fraction'] == source_to_fraction[source]]

    validation = common_df_processing(validation)
    
    return validation

def get_train(source):
    train = pd.read_csv("{}/generated_{}_train.csv".format(GENERATED_LABELS_DIR, source))
    train.prop_count = train.prop_count.astype(np.float64)
    
    train = common_df_processing(train)
    return train


# ### Model

# In[5]:


from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dropout, MaxPooling2D, Flatten, Dense
from tensorflow import concat as concatenate

def dense_layer(x, layer_configs):
    layers = []
    for i in range(2):
        if layer_configs[i]["layer_type"] == "Conv2D":
            layer = Conv2D(layer_configs[i]["filters"], layer_configs[i]["kernel_size"], strides = layer_configs[i]["strides"], padding = layer_configs[i]["padding"], activation = layer_configs[i]["activation"])(x)
        layers.append(layer)
            
    for n in range(2, len(layer_configs)):
        if layer_configs[n]["layer_type"] == "Conv2D":
            layer = Conv2D(layer_configs[n]["filters"], layer_configs[n]["kernel_size"], strides = layer_configs[n]["strides"], padding = layer_configs[n]["padding"], activation = layer_configs[n]["activation"])(concatenate(layers, axis = 3))
        layers.append(layer)
    return layer
layer_f8 = [
    {
        "layer_type" : "Conv2D", "filters" : 8, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
    },{
        "layer_type" : "Conv2D", "filters" : 8, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
    },{
        "layer_type" : "Conv2D", "filters" : 8, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
    },{
        "layer_type" : "Conv2D", "filters" : 8, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
    },{
        "layer_type" : "Conv2D", "filters" : 8, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
    }
]

layer_f16 = [
    {
        "layer_type" : "Conv2D", "filters" : 16, "kernel_size" : (3, 3), "strides" : 2, "padding" : "same", "activation" : "relu"
    },{
        "layer_type" : "Conv2D", "filters" : 16, "kernel_size" : (3, 3), "strides" : 2, "padding" : "same", "activation" : "relu"
    },{
        "layer_type" : "Conv2D", "filters" : 16, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
    },{
        "layer_type" : "Conv2D", "filters" : 16, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
    },{
        "layer_type" : "Conv2D", "filters" : 16, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
    }
]

layer_f32 = [
    {
        "layer_type" : "Conv2D", "filters" : 32, "kernel_size" : (3, 3), "strides" : 2, "padding" : "same", "activation" : "relu"
    },{
        "layer_type" : "Conv2D", "filters" : 32, "kernel_size" : (3, 3), "strides" : 2, "padding" : "same", "activation" : "relu"
    },{
        "layer_type" : "Conv2D", "filters" : 32, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
    },{
        "layer_type" : "Conv2D", "filters" : 32, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
    },{
        "layer_type" : "Conv2D", "filters" : 32, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
    }
]

layer_f64 = [
    {
        "layer_type" : "Conv2D", "filters" : 64, "kernel_size" : (3, 3), "strides" : 2, "padding" : "same", "activation" : "relu"
    },{
        "layer_type" : "Conv2D", "filters" : 64, "kernel_size" : (3, 3), "strides" : 2, "padding" : "same", "activation" : "relu"
    },{
        "layer_type" : "Conv2D", "filters" : 64, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
    },{
        "layer_type" : "Conv2D", "filters" : 64, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
    },{
        "layer_type" : "Conv2D", "filters" : 64, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
    }
]

inp = Input(shape = (TARGET_SHAPE[0] // COEF_COMPRESS // SUB_IMG_CNT, TARGET_SHAPE[1] // COEF_COMPRESS // SUB_IMG_CNT, 3))
x = inp
x = Conv2D(4, (3, 3), strides = 3, padding = "same", activation = "relu")(x)
x = dense_layer(x, layer_f8)
x = Dropout(0.5)(x)

x = BatchNormalization()(x)
x = dense_layer(x, layer_f16)
x = Dropout(0.5)(x)

# x = BatchNormalization(axis = 3)(x)
# x = dense_layer(x, layer_f32)
# x = Dropout(0.8)(x)

# x = BatchNormalization(axis = 3)(x)
# x = dense_layer(x, layer_f64)
# x = Dropout(0.8)(x)

x = MaxPooling2D((2, 2))(x)
x = BatchNormalization()(x)
x = Conv2D(96, (1, 1), activation = "relu")(x)
x = BatchNormalization()(x)

x = MaxPooling2D((2, 2))(x)
x = BatchNormalization()(x)
x = Flatten()(x)

x = Dropout(0.4)(x)
x = Dense(len(['16', '18', '20', '25', '30', '35', '40']), activation = "softmax")(x)

dense_net = Model(inp, x)
dense_net.summary()


# In[6]:


# class BinsExtraction(Model):
#     def __init__(self, fraction):
#         super(BinsExtraction, self).__init__()
#         self.FilterSize1 = 10
        
#         self.fraction_sievs = fraction_sievs[fraction]
        
#         self.pipe_for_size = {}
        
#         for b in self.fraction_sievs:
#             d = round(2 * bin2high[b] / COEF_COMPRESS)
            
#             self.pipe_for_size[b] = [
#                 [
#                     tf.keras.layers.Conv2D(filters=self.FilterSize1, kernel_size=(3, 3), strides=(2, 2)),
#                     tf.keras.layers.BatchNormalization(),
#                     tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
#                 ],
#                 [
#                     tf.keras.layers.Conv2D(filters=self.FilterSize1, kernel_size=(3, 3), strides=(2, 2)),
#                     tf.keras.layers.BatchNormalization(),
#                     tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
#                 ],
#                 [
#                     tf.keras.layers.Conv2D(filters=self.FilterSize1, kernel_size=(3, 3), strides=(2, 2)),
#                     tf.keras.layers.BatchNormalization(),
#                     tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
#                 ]
#             ]
        
#         self.cnn_afterword = [
#             tf.keras.layers.Dropout(rate=0.3),
#             tf.keras.layers.Flatten(),
#             lambda x: tf.reduce_sum(x, axis=1, keepdims=True)
#         ]
        
#         self.fraction_agregate = [
#             tf.keras.layers.Dense(1, activation='softmax')
#         ]
        
#         self.afterword_pipes = [
#             tf.keras.layers.Dense(len(self.fraction_sievs), activation='softmax')
#         ]

#     def call(self, x, *args, **kwargs):
#         res = []
#         for b in self.fraction_sievs:
#             cnn_res = []
#             copy_x = x
#             cnn_pipes = self.pipe_for_size[b]
#             for cnn in cnn_pipes:
#                 for pipe in cnn:
#                     copy_x = pipe(copy_x)
                
#                 result_after_cnn = copy_x
#                 for layer in self.cnn_afterword:
#                     result_after_cnn = layer(result_after_cnn)
#                 cnn_res.append(result_after_cnn)
            

#             cnn_res = tf.concat(cnn_res, axis=1)
#            # print("cnn_res: ", cnn_res)
#             for layer in self.fraction_agregate:
#                 cnn_res = layer(cnn_res)
            
#             res.append(cnn_res)
        

#         res = tf.concat(res, axis=1)
#         #print("res: ", res)
        
#         for pipe in self.afterword_pipes:
#             res = pipe(res)
        
        
#         return res
    


# In[7]:


# model = BinsExtraction('bw')


# #### Train Input Generator

# In[8]:


def get_train_val_datagen(train, source, train_size=0.8):
    train_fraction, val_fraction = train_test_split(train, train_size=train_size, random_state=123)
    
    bins_fraction = fraction_sievs[fraction]
    
    datagen = ImageDataGenerator()

    train_generator = datagen.flow_from_dataframe(
            train_fraction.sample(n=int(len(train_fraction) * DF_RATE)),
            directory="kaggle_problems/rosneft_proppant/data/generated/{}_img".format(source),
            x_col='filename', 
            y_col=bins_fraction,
            target_size=(TARGET_SHAPE[0] // COEF_COMPRESS // SUB_IMG_CNT, TARGET_SHAPE[1] // COEF_COMPRESS // SUB_IMG_CNT),
            batch_size=512,
            class_mode='other',
    )
    
    val_generator = datagen.flow_from_dataframe(
        val_fraction.sample(n=int(len(val_fraction) * DF_RATE)),
        directory="kaggle_problems/rosneft_proppant/data/generated/{}_img".format(source),
        x_col='filename', 
        y_col=bins_fraction,
        target_size=(TARGET_SHAPE[0] // COEF_COMPRESS // SUB_IMG_CNT, TARGET_SHAPE[1] // COEF_COMPRESS // SUB_IMG_CNT),
        batch_size=512,
        class_mode='other')
    return train_generator, val_generator


# In[9]:


# def get_train_val_datagen(train, validation, source):
#     fraction = source_to_fraction[source]
#     bins_fraction = fraction_sievs[fraction]
    
#     datagen = ImageDataGenerator()

#     train_generator = datagen.flow_from_dataframe(
#             train.sample(n=int(len(train) * DF_RATE)),
#             directory="kaggle_problems/rosneft_proppant/data/generated/{}_img".format(source),
#             x_col='filename', 
#             y_col=bins_fraction,
#             target_size=(TARGET_SHAPE[0] // COEF_COMPRESS, TARGET_SHAPE[1] // COEF_COMPRESS),
#             batch_size=64,
#             class_mode='other',
#     )
    
#     validation_generator = datagen.flow_from_dataframe(
#             validation.sample(n=int(len(validation) * DF_RATE)),
#             directory="kaggle_problems/rosneft_proppant/data/{}_main_area".format(source),
#             x_col='filename', 
#             y_col=bins_fraction,
#             target_size=(TARGET_SHAPE[0] // COEF_COMPRESS, TARGET_SHAPE[1] // COEF_COMPRESS),
#             batch_size=64,
#             class_mode='other',
#     )
    
#     return train_generator, validation_generator


# #### Input generator checking

# In[10]:


# img, labels = get_train_val_datagen(train, validation, 'bw')[0].next()
# display_images(img[0:8].astype(int), 4)


# #### Callbacks

# In[11]:


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# In[12]:


learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                            patience=5, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

earlystop = EarlyStopping(patience=10)

callbacks = [earlystop, learning_rate_reduction]


# In[13]:


EPS = 1e-5
def metric(true, predicted):
    true = tf.math.maximum(true, EPS * tf.ones_like( true ))
    predicted = tf.math.maximum(predicted, EPS * tf.ones_like( predicted ))
    
    sum_true = tf.math.reduce_sum(true, axis=1, keepdims=True)
    true /= sum_true
    
    sum_predicted = tf.math.reduce_sum(predicted, axis = 1, keepdims=True)
    predicted /= sum_predicted

    return tf.keras.backend.mean(tf.math.reduce_sum((true - predicted) ** 2 / (true + predicted), axis=1))


# In[14]:


for source, i in zip(sources, range(len(sources))):
    validation = get_validation(source)
    train = get_train(source)
    fraction = source_to_fraction[source]
    
    
    model = dense_net #BinsExtraction(fraction)
    model.compile(
        loss=metric,
        optimizer='rmsprop',
    )

    train_datagen, val_datagen = get_train_val_datagen(train, source)
    
    history = model.fit(
        x=train_datagen,
        epochs=30,
        validation_data=val_datagen,
        callbacks=callbacks
    )
    
    Path(MODEL_DIR).mkdir(exist_ok=True, parents=True)
    
    with open(MODEL_DIR + "/history_model_benchmark_{}.pickle".format(source), 'wb') as f:
        pickle.dump(history.history, f)
        
    model.save(MODEL_DIR + "/model_benchmark_{}".format(source))


# In[ ]:


res = model.predict(x=val_datagen)


# In[ ]:


print(res)


# In[ ]:


model.summary()


# In[ ]:


def get_bins_metric(predicted, true):
    return 0.5 * np.sum((predicted - true) ** 2 / (predicted + true)) / predicted.shape[0]

def get_bins_metric_by_image(predicted, true):
    return np.sum(0.5 * (predicted - true) ** 2 / (predicted + true), axis=1)

def get_bins_metric_by_bins(predicted, true):
    return np.sum(0.5 * (predicted - true) ** 2 / (predicted + true), axis=0)


# In[ ]:


#print("Total bin loss: {}".format(get_bins_metric(predicted_labels, all_labels)))


# In[ ]:


# for source, i in zip(sources, range(len(sources))):
#     fraction = source_to_fraction[source]
#     print(source + "-" * 100)
#     from keras.utils.generic_utils import get_custom_objects

#     get_custom_objects().update({'metric': metric})

#     with open(MODEL_DIR + "/history_model_benchmark_{}.pickle".format(source), 'rb') as f:
#         history = pickle.load(f)

#     model = tf.keras.models.load_model(MODEL_DIR + "/model_benchmark_{}".format(source), 
#                                        compile=False)
#     model.compile(
#         loss=metric,
#         optimizer='rmsprop',
#        # metrics=['mse']
#     )                                                                               

#     display_history_metrics(history, source)
#     print(source + '-' * 100)
    
#     train_datagen = get_train_val_datagen(train, fraction)[0]

#     predicted_labels = []
#     all_labels = []
#     train_fraction = train[train['fraction'] == fraction]

#     for i in range(int(train_fraction.shape[0]) // 16):
#         imgs, labels = train_datagen.next()
#         predicted_labels.extend(model.predict(imgs))
#         all_labels.extend(labels)
#     predicted_labels = np.array(predicted_labels)
#     all_labels = np.array(all_labels)

#     losses_by_img = get_bins_metric_by_image(predicted_labels, all_labels)
#     plt.hist(losses_by_img, bins=100)
#     plt.show()

#     losses_by_bins = get_bins_metric_by_bins(predicted_labels, all_labels)
#     plt.hist(losses_by_bins, bins=100)
#     plt.show()
#     print("-" * 50)


# In[ ]:


get_ipython().system('jupyter nbconvert --to script kaggle_problems/rosneft_proppant/cnn_try.ipynb')


# In[ ]:





# In[ ]:




