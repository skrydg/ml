#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
while not os.getcwd().endswith('ml'):
    os.chdir('..')
sys.path.insert(0, os.getcwd())


# In[2]:


import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict

from helpers.word2vec.converter import *


# In[3]:


MAX_SENTENCE_LEN = 50
WORD_REPRESENTATION_LEN = 300


# #### Считывание данных

# In[4]:


train = pd.read_csv("kaggle_problems/tweet_sentiment_extraction/train.csv")
test = pd.read_csv("kaggle_problems/tweet_sentiment_extraction/test.csv")


# #### Описание данных

# In[5]:


train.sample()


# In[6]:


test.sample()


# In[7]:


print(len(test), len(train))


# ### Word2Vec convertation + save on disk

# In[8]:


train = train[~train['text'].isnull()]
test = test[~test['text'].isnull()]


# In[9]:


sentence_converter = Converter(tokenizer_type=TokenizerType.tweet_tokenizer)


# In[25]:


def preprocessing(data):
    data = data[~data.isnull()]
    sentence_converter.clear_statistic()
    vectors, cleared_sentences = sentence_converter.convert_sentences(data)
    
    unknown_words = np.sum([i for i in sentence_converter.unknown_words.values() if i is not None])
    known_words = np.sum([i for i in sentence_converter.known_words.values()if i is not None])

    print("unknown_words: {}, known_words: {}, persent unknown words: {}".format( 
          unknown_words, known_words, unknown_words / (unknown_words + known_words)))
    
    return np.array([[
        [i for i in sentence[word_nmb]] 
        if word_nmb < len(sentence) and sentence[word_nmb] is not None
        else np.zeros(WORD_REPRESENTATION_LEN)
        for word_nmb in range(0, MAX_SENTENCE_LEN) 
    ] for sentence in vectors]), cleared_sentences, sentence_converter.unknown_words, sentence_converter.known_words


# In[11]:


vectors, test_cleared_sentences, test_unknown_words, test_known_words = preprocessing(test['text'])
pickle.dump(vectors, open('kaggle_problems/tweet_sentiment_extraction/pickle_dump/test.pkl', 'wb'))
pickle.dump(test_cleared_sentences, open('kaggle_problems/tweet_sentiment_extraction/pickle_dump/test_cleared_sentences.pkl', 'wb'))

pickle.dump(test_unknown_words, open('kaggle_problems/tweet_sentiment_extraction/pickle_dump/test_unknown_words.pkl', 'wb'))
pickle.dump(test_known_words, open('kaggle_problems/tweet_sentiment_extraction/pickle_dump/test_known_words.pkl', 'wb'))


# In[12]:


vectors, train_cleared_sentences, train_unknown_words, train_known_words = preprocessing(train['text'])
pickle.dump(vectors, open('kaggle_problems/tweet_sentiment_extraction/pickle_dump/train.pkl', 'wb'))
pickle.dump(train_cleared_sentences, open('kaggle_problems/tweet_sentiment_extraction/pickle_dump/train_cleared_sentences.pkl', 'wb'))

pickle.dump(train_unknown_words, open('kaggle_problems/tweet_sentiment_extraction/pickle_dump/train_unknown_words.pkl', 'wb'))
pickle.dump(train_known_words, open('kaggle_problems/tweet_sentiment_extraction/pickle_dump/train_known_words.pkl', 'wb'))


# In[13]:


vectors, train_cleared_sentences, train_unknown_words, train_known_words = preprocessing(train['selected_text'])
pickle.dump(vectors, open('kaggle_problems/tweet_sentiment_extraction/pickle_dump/selected_train.pkl', 'wb'))
pickle.dump(train_cleared_sentences, open('kaggle_problems/tweet_sentiment_extraction/pickle_dump/selected_train_cleared_sentences.pkl', 'wb'))

pickle.dump(train_unknown_words, open('kaggle_problems/tweet_sentiment_extraction/pickle_dump/selected_train_unknown_words.pkl.pkl', 'wb'))
pickle.dump(train_known_words, open('kaggle_problems/tweet_sentiment_extraction/pickle_dump/selected_train_known_words.pkl', 'wb'))


# In[14]:


train_known_words = pickle.load(open('kaggle_problems/tweet_sentiment_extraction/pickle_dump/train_known_words', 'rb'))


# In[ ]:





# In[15]:


sorted(train_unknown_words.items(), key=lambda x : x[1], reverse=True)


# In[16]:


sorted(sentence_converter.unknown_words.items(), key=lambda x : x[1], reverse=True)


# In[17]:


unknown_words = np.sum([i for i in sentence_converter.unknown_words.values()])
known_words = np.sum([i for i in sentence_converter.known_words.values()])
print(unknown_words / (unknown_words + known_words))


# #### Проверка гипотезы

# In[18]:


#
# Гипотеза: слова из selected_text образуют подотрезок из text
#
cnt_true = 0
cnt_false = 0

for index, row in train.iterrows():
    if row['selected_text'].lower() in row['text'].lower():
        cnt_true += 1
    else:
        cnt_false += 1
print(cnt_true, cnt_false)


# In[26]:


get_ipython().system('jupyter nbconvert --to script kaggle_problems/tweet_sentiment_extraction/benchmark.ipynb')


# In[20]:


# MAX_WORDS = 35

# def selected_text_start(x):
#     start_char = x['text'].find(x['selected_text'])
#     start_word = len(x['text'][:start_char].split())
#     borders = np.zeros(MAX_WORDS, dtype=int)
#     borders[start_word] = 1
#     return borders

# def selected_text_end(x):
#     end_word = np.where(x['start_word'] == 1)[0][0] + len(x['selected_text'].split()) - 1
#     borders = np.zeros(MAX_WORDS, dtype=int)
#     borders[end_word] = 1
#     return borders

# train['start_word'] = train.apply(lambda x: selected_text_start(x), axis=1)
# train['end_word'] = train.apply(lambda x: selected_text_end(x), axis=1)

