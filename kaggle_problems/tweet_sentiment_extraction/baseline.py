#!/usr/bin/env python
# coding: utf-8

# In[36]:


import os
import sys
while not os.getcwd().endswith('ml'):
    os.chdir('..')
sys.path.insert(0, os.getcwd())


# In[211]:


import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import pickle
import spacy
import random
from collections import defaultdict
from spacy.util import minibatch, compounding
from spacy.lang.en import English


from libs.nlp.ner.ner import NER
from helpers.word2vec.converter import *


# In[171]:


MAIN_PART_LABEL = 'MAIN_PART_LABEL'


# In[172]:


train = pd.read_csv("kaggle_problems/tweet_sentiment_extraction/train.csv")
test = pd.read_csv("kaggle_problems/tweet_sentiment_extraction/test.csv")

train.dropna(inplace=True)
train.reset_index(drop=True, inplace=True)
test.dropna(inplace=True)
test.reset_index(drop=True, inplace=True)


# In[176]:


def preprocessing_column(column):
    column = column.apply(lambda x: ''.join([i for i in x if (i.isalpha() or i == ' ')]))
    column = column.apply(lambda x: re.sub(' +', ' ', x))
    
    column = column.apply(lambda x: x[1:] if x.startswith(' ') else x)
    column = column.apply(lambda x: x[:-1] if x.endswith(' ') else x)
    
    return column

def preprocessing(data):
    data.text = preprocessing_column(data.text)
    if 'selected_text' in data.columns:
        data.selected_text = preprocessing_column(data.selected_text)
    return data


# In[177]:


train = preprocessing(train)
test = preprocessing(test)


# In[203]:


def tokenize(s):
    return s.split(' ')

def get_start_end_words(x):
    start_char = x['text'].find(x['selected_text'])
    start_word = len(tokenize(x['text'][:start_char + 1])) - 1
    
    cnt_word = len(tokenize(x['selected_text']))
    return start_word, start_word + cnt_word

def get_start_end_char(x):
    start_char = x['text'].find(x['selected_text'])
    end_char = start_char + len(x['selected_text'])
    
    start_char = x['text'][:start_char].rfind(' ') + 1
    if start_char < 0:
        start_char = 0
        
    first_space = x['text'][end_char:].find(' ')
    if first_space < 0:
        end_char = len(x['text'])
    else:
        end_char = end_char + first_space
    return start_char, end_char


# In[218]:


def df_to_spacy_format(data):
    data.reset_index(drop=True, inplace=True)
    spacy_data  = [0] * len(data)
    for ind, line in data.iterrows():
    #     print("-" * 100)
    #     print(ind)
        start_word, end_word = get_start_end_char(line)
        spacy_data[ind] = (
            line['text'], 
            {"entities": [(start_word, end_word, MAIN_PART_LABEL)]}
        )
    #     nlp = English()
    #     tokens = nlp(line['text'])

    #     print([t.text for t in tokens])
    #     print(line['text'])
    #     print(start_word, end_word)
    #     print(line['selected_text'])
    #     print(spacy.gold.biluo_tags_from_offsets(model.make_doc(line['text']), [(start_word, end_word, MAIN_PART_LABEL)]))
    #     print("-" * 100)
    return spacy_data


# In[219]:





# ### Training

# In[228]:


for sentiment in ['positive', 'negative', 'neutral']:
    print("Training for {}".format(sentiment))
    print("-" * 100)
    model = NER()
    spacy_train_pos = df_to_spacy_format(train[train['sentiment'] == sentiment])
    model.train(spacy_train_pos, n_iter=30)
    model.save_model('kaggle_problems/tweet_sentiment_extraction/models/ner_{}'.format(sentiment))
    print("-" * 100)


# In[ ]:


get_ipython().system('jupyter nbconvert --to script kaggle_problems/tweet_sentiment_extraction/baseline.ipynb')


# In[ ]:




