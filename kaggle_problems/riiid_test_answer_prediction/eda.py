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
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

from kaggle_problems.riiid_test_answer_prediction.workspace.common import apply_to_train


# In[3]:


input_dir = Path('kaggle_problems/riiid_test_answer_prediction/workspace/data')
batched_dir = input_dir / "batched_by_user_train"


# In[4]:


lectures = pd.read_csv("{}/lectures.csv".format(input_dir))
questions = pd.read_csv("{}/questions.csv".format(input_dir))

print("len of lectures={}, len of questions={}".format(len(lectures), len(questions)))


# In[ ]:





# In[5]:


lectures.sample(10)


# In[6]:


lectures[lectures.tag == 102]


# In[7]:


questions.sample(10)


# In[8]:


questions[questions.bundle_id == 12468]


# ### Save train in pickle format for fast reading

# In[9]:


# %%time
# import datatable as dt
# train = dt.fread("{}/train.csv".format(input_dir)).to_pandas()
# print(train.shape)
# pickle.dump(train, open("{}/train.pkl".format(input_dir), 'wb'))


# In[10]:


# train = pickle.load(open("{}/train.pkl".format(input_dir), 'rb'))


# ### Batched train by user id

# In[11]:


# batched_dir = Path("{}/batched_by_user_train".format(input_dir))
# batched_dir.mkdir(parents=True, exist_ok=True)

# train = train.sort_values('user_id')

# BATCH_SIZE = 1000000
# l = 0
# ind = 0

# while l < len(train):
#     r = min(l + BATCH_SIZE, len(train))
    
#     while (r < len(train) and train.iloc[r].user_id == train.iloc[r - 1].user_id):
#         r += 1
        
#     pickle.dump(train.iloc[l:r], open(batched_dir / "train_{}.pkl".format(ind), 'wb'))
#     print("for ind={}, len={} ".format(ind, r - l))
#     ind += 1
#     l = r


# ### Check that total len and train len is equal

# In[12]:


# total_size = 0
# for train_name in batched_dir.glob("*.pkl"):
#     train = pickle.load(open(train_name, 'rb'))
#     total_size += len(train)

# train = pickle.load(open("{}/train.pkl".format(input_dir), 'rb'))
# if (len(train) == total_size):
#     print("Len is equal")
# else:
#     print("ERROR")


# ### CV split

# In[13]:


START_INTERVAL = 5 * 365 * 24 * 60 * 60 * 1000
def enrich_event_time(train):
    users = train['user_id'].unique()
    start_time = (np.random.rand(len(users)) * START_INTERVAL).astype(np.int)
    user_start_time = pd.DataFrame(data={"user_id": users, "join_time": start_time}).set_index(['user_id'])

    train = pd.concat([train.reset_index(drop=True), 
           user_start_time.reindex(train['user_id'].values).reset_index(drop=True)], axis=1)
    
    train['event_time'] = train['join_time'] + train['timestamp']
    return train


# In[14]:


train_name = [i for i in batched_dir.glob("*.pkl")][0]
train = pickle.load(open(train_name, 'rb'))


# In[15]:


train = enrich_event_time(train)


# In[ ]:





# ### Feature generation

# In[16]:


lectures = pd.read_csv("{}/lectures.csv".format(input_dir)).set_index('lecture_id')
questions = pd.read_csv("{}/questions.csv".format(input_dir)).set_index('question_id')

def enrich_content(train):
    lecture_train = train[train.content_type_id]
    question_train = train[~train.content_type_id]
    
    lecture_train = pd.concat([lecture_train.reset_index(drop=True), 
       lectures.reindex(lecture_train['content_id'].values).reset_index(drop=True)], axis=1)
    
    question_train = pd.concat([question_train.reset_index(drop=True), 
       questions.reindex(question_train['content_id'].values).reset_index(drop=True)], axis=1)
    
    question_train = question_train.loc[:,~question_train.columns.duplicated()]
    lecture_train = lecture_train.loc[:,~lecture_train.columns.duplicated()]
    
    return lecture_train.append(question_train, ignore_index = True, sort=True)


# In[17]:


train = enrich_content(train)


# In[18]:


def enrich_persent_right_answers(train):
    question_train = train[~train.content_type_id] # stay only questions
    lectures_train = train[train.content_type_id]
    question_train = question_train.sort_values('event_time')
    question_train['count_answered_correctly'] = question_train.groupby('user_id')['answered_correctly'].cumsum().astype(int)
    question_train['count_answered_questions'] = question_train.groupby(train['user_id'])['answered_correctly'].cumcount().astype(int) + 1
    question_train['persent_answered_correctly'] = question_train['count_answered_correctly'].astype(np.double) / question_train['count_answered_questions']
    
    return pd.concat([question_train, lectures_train], ignore_index=True, sort=False)


# In[19]:


train = enrich_persent_right_answers(train)


# In[20]:


def enrich_content_mean(train):
    content_mean = train.groupby('content_id', as_index=False)['answered_correctly'].mean()
    content_mean = content_mean.rename(columns={'answered_correctly': 'content_mean'}).set_index('content_id')
    
    content_sum = train.groupby('content_id', as_index=False)['answered_correctly'].sum()
    content_sum = content_sum.rename(columns={'answered_correctly': 'content_sum'}).set_index('content_id')
    
    train = pd.concat([train.reset_index(drop=True), 
       content_mean.reindex(train['content_id'].values).reset_index(drop=True),
       content_sum.reindex(train['content_id'].values).reset_index(drop=True)], axis=1)
        
    return train


# In[21]:


train = enrich_content_mean(train)


# ### BaseLine (LGB)

# In[22]:


train = train[~train.content_type_id]


# In[23]:


import lightgbm as lgb


# In[ ]:


features = ['count_answered_correctly', 
            'count_answered_questions', 
            'persent_answered_correctly', 
            'content_id', 
            'content_mean',
            'content_sum', 
            'prior_question_elapsed_time']


# In[ ]:


train = train.sample(5)


# In[ ]:


train


# In[ ]:


y_train = train['answered_correctly']
X_train = train[features]

X_train['prior_question_elapsed_time'].fillna(0, inplace=True)


# In[ ]:


X_train


# In[ ]:




lgb_train = lgb.Dataset(X_train, y_train, categorical_feature = None)
#lgb_eval = lgb.Dataset(validation, y_val, categorical_feature = None)
#del train, y_train, validation, y_val
#gc.collect()


# In[ ]:





# In[ ]:


params = {'objective': 'binary',
          'metric': 'auc',
          #'seed': 2020,
          #'learning_rate': 0.1, #default
          #"boosting_type": "gbdt", #default
          #"num_leaves": 10
         }


# In[ ]:


# model = lgb.train(
#     params, lgb_train,
#     #valid_sets=[lgb_train],
#     verbose_eval=1,
#     num_boost_round=1,
#     early_stopping_rounds=8,
# )


# In[ ]:


lgb.train(params, lgb.Dataset(pd.DataFrame({'x': [1]}), [1], categorical_feature = None))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


pd.set_option('display.max_columns', 100)
print(train[train.user_id == 4421282].sort_values('timestamp').iloc[0:100])


# In[ ]:


train.user_answer.unique()


# In[ ]:


train.content_type_id.value_counts()


# In[ ]:


train = train[train.content_type_id == 1]


# In[ ]:


example_test = pd.read_csv("kaggle_problems/riddly/data/example_test.csv")


# In[ ]:


example_test.describe()


# In[ ]:


example_sample_submission = pd.read_csv("kaggle_problems/riddly/data/example_sample_submission.csv")
example_sample_submission.sample(10)


# In[ ]:


get_ipython().system('jupyter nbconvert --to script kaggle_problems/riiid_test_answer_prediction/eda.ipynb')

