#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os
import sys
while not os.getcwd().endswith('ml'):
    os.chdir('..')
sys.path.insert(0, os.getcwd())


# In[7]:


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


# In[8]:


MAIN_PART_LABEL = 'MAIN_PART_LABEL'


# In[9]:


def df_jaccard(data):
    return jaccard(data['text'], data['selected_text'])

def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    if (len(a) + len(b) == 0):
        return 0
    return float(len(c)) / (len(a) + len(b) - len(c))

def arr_jaccard(arr1, arr2):
    assert(len(arr1) == len(arr2))
    res = 0
    for str1, str2 in zip(arr1, arr2):
        res += jaccard(str1, str2)
        
    return res / len(arr1)


# In[10]:


train = pd.read_csv("kaggle_problems/tweet_sentiment_extraction/train.csv")
test = pd.read_csv("kaggle_problems/tweet_sentiment_extraction/test.csv")

train.dropna(inplace=True)
train.reset_index(drop=True, inplace=True)
test.dropna(inplace=True)
test.reset_index(drop=True, inplace=True)


# In[54]:


def preprocessing_column(column):
    #column = column.apply(lambda x: ''.join([i for i in x if (i.isalpha() or i == ' ')]))
    #column = column.apply(lambda x: re.sub(' +', ' ', x))
    
    #column = column.apply(lambda x: x[1:] if x.startswith(' ') else x)
    #column = column.apply(lambda x: x[:-1] if x.endswith(' ') else x)
    
    return column

def preprocessing(data):
    data.text = preprocessing_column(data.text)
    data['text_words'] = data['text'].apply(lambda x: x.split(" "))
    data['text_cnt_words'] = data['text'].apply(lambda x: len(x.split(" ")))
    
    if 'selected_text' in data.columns:
        data.selected_text = preprocessing_column(data.selected_text)
        data['selected_text_words'] = data['selected_text'].apply(lambda x: x.split(" "))
        data['selected_text_cnt_words'] = data['selected_text'].apply(lambda x: len(x.split(" ")))
    
    return data


# In[55]:


train = preprocessing(train)
test = preprocessing(test)


# ### Гипотеза что расстояние джакара между selected_texts and texts маленькое для text_cnt_words < X

# In[73]:


jacard_dist = defaultdict(list)
for sentiment in ['positive', 'negative']:
    for i in range(1, 10):
        msk = (train['text_cnt_words'] <= i) & (train['sentiment'] == sentiment)
        texts = train[msk]['text']
        selected_texts = train[msk]['selected_text']
        jacard_dist[sentiment].append(arr_jaccard(texts, selected_texts))
        
for sentiment in ['positive', 'negative']:
    plt.plot(jacard_dist[sentiment], label=sentiment)
plt.legend()
plt.show()


# In[75]:


jacard_dist = defaultdict(list)
for sentiment in ['positive', 'negative']:
    model = NER()
    model.load_model('kaggle_problems/tweet_sentiment_extraction/models/ner_{}'.format(sentiment))
    for i in range(1, 10):
        msk = (train['text_cnt_words'] <= i) & (train['sentiment'] == sentiment)
        
        predict_selected_texts = predict(model, train[msk])
        selected_texts = train[msk].selected_text.to_numpy()

        jacard_dist[sentiment].append(arr_jaccard(selected_texts, predict_selected_texts))
        
        
for sentiment in ['positive', 'negative']:
    plt.plot(jacard_dist[sentiment], label=sentiment)
plt.legend()
plt.show()


# In[13]:


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


# In[14]:


def df_to_spacy_format(data):
    data.reset_index(drop=True, inplace=True)
    spacy_data  = [0] * len(data)
    for ind, line in data.iterrows():
#         print("-" * 100)
#         print(ind)
        start_word, end_word = get_start_end_char(line)
        spacy_data[ind] = (
            line['text'], 
            {"entities": [(start_word, end_word, MAIN_PART_LABEL)]}
        )
#         nlp = English()
#         tokens = nlp(line['text'])

#         print([t.text for t in tokens])
#         print(line['text'])
#         print(start_word, end_word)
#         print(line['selected_text'])
#         print(spacy.gold.biluo_tags_from_offsets(nlp.make_doc(line['text']), [(start_word, end_word, MAIN_PART_LABEL)]))
#         print("-" * 100)
    return spacy_data


# In[34]:


def get_selected_text(texts, y):
    predict_selected_texts = []
    for ent, text in zip(y, texts):
        if (len(ent)):
            start = ent[0][0]
            end = ent[0][1]
        else:
            start = 0
            end = len(text)
        predict_selected_texts.append(text[start:end])
    return predict_selected_texts
        
def evaluate_score(texts, y_true, y_predict):
    selected_texts = get_selected_text(texts, y_true)
    predicted_selected_texts = get_selected_text(texts, y_predict)
        
    return arr_jaccard(selected_texts, predicted_selected_texts)


# ### Training

# In[47]:


for sentiment in ['positive', 'negative', 'neutral']:
    print("Training for {}".format(sentiment))
    print("-" * 100)
    model = NER(evaluate_score=evaluate_score)
    spacy_train_pos = df_to_spacy_format(train[train['sentiment'] == sentiment][0:10])
    train_los, validation_los = model.train(spacy_train_pos, n_iter=10)
    model.save_model('kaggle_problems/tweet_sentiment_extraction/models/ner_{}'.format(sentiment))
    pickle.dump(train_los, open("kaggle_problems/tweet_sentiment_extraction/data/baseline/train_los.pkl", 'wb'))
    pickle.dump(validation_los, open("kaggle_problems/tweet_sentiment_extraction/data/baseline/validation_los.pkl", 'wb'))
    
    print("-" * 100)


# In[48]:


train_los = pickle.load(open("kaggle_problems/tweet_sentiment_extraction/data/baseline/train_los.pkl", 'rb'))
validation_los =pickle.load(open("kaggle_problems/tweet_sentiment_extraction/data/baseline/validation_los.pkl", 'rb'))
    
plt.plot(train_los, label='train_los')
plt.plot(validation_los, label='validation_los')
plt.legend()
plt.show()


# In[37]:


print(train_los, validation_los)


# ### Prediction

# In[15]:


def predict(model, data):
    texts = data.text.to_numpy()
    
    result = model.predict(texts)
    predict_selected_texts = get_selected_text(texts, result)
    return predict_selected_texts


# ### Predict on train

# In[16]:


prediction = {}
for sentiment in ['positive', 'negative']:
    model = NER()
    model.load_model('kaggle_problems/tweet_sentiment_extraction/models/ner_{}'.format(sentiment))
    
    predict_selected_texts = predict(model, train[train['sentiment'] == sentiment])
    selected_texts = train[train['sentiment'] == sentiment].selected_text.to_numpy()
        
    print(arr_jaccard(selected_texts, predict_selected_texts))


# ### Predict on test

# In[12]:


result_df = pd.DataFrame(columns=['textID', 'selected_text'])
for sentiment in ['positive', 'negative']:
    model = NER()
    model.load_model('kaggle_problems/tweet_sentiment_extraction/models/ner_{}'.format(sentiment))
    
    predict_selected_texts = predict(model, test[test['sentiment'] == sentiment])
    
    result_df = result_df.append(
        pd.DataFrame(
            data={'textID': test[test['sentiment'] == sentiment].textID.to_numpy(),
                  'selected_text': predict_selected_texts}, 
            columns=['textID', 'selected_text'],
        )
    )
    
result_df = result_df.append(
    pd.DataFrame(
        data={'textID': test[test['sentiment'] == 'neutral'].textID.to_numpy(),
              'selected_text': test[test['sentiment'] == 'neutral'].text.to_numpy()}, 
        columns=['textID', 'selected_text'],
    )
)
result_df = result_df.set_index('textID')


# In[13]:


result_df.to_csv('kaggle_problems/tweet_sentiment_extraction/submissions/{}'.format('baseline_ner'))


# In[297]:


get_ipython().system('jupyter nbconvert --to script kaggle_problems/tweet_sentiment_extraction/baseline.ipynb')


# In[ ]:




