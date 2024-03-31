#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]


# In[18]:


X,y = shuffle(data,target,random_state = 13)
X = X.astype(np.float32)
offset = int(X.shape[0]*0.9)
x_train,y_train = X[:offset],y[:offset]
x_test,y_test = X[offset:],y[offset:]


# In[20]:


params = {'n_estimators':500,'max_depth':4,'min_samples_split':2,'learning_rate':0.01,'loss':'squared_error'}
clf = ensemble.GradientBoostingRegressor(**params)
clf.fit(x_train,y_train)
mse= mean_squared_error(y_test,clf.predict(x_test))
print("MSE:%.4f"%mse)


# In[48]:


test_score = np.zeros((params['n_estimators'],))
for i,y_pred in enumerate(clf.staged_predict(x_test)):
    test_score[i] = mean_squared_error(y_test,y_pred)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))
plt.title('Deviance')
plt.subplot(1,2,1)
axes[0,1].plot(np.arange(params['n_estimators'])+1,clf.train_score_,'b-',label = 'Training Set Deviance')
axes[0,1].plot(np.arange(params['n_estimators'])+1,test_score,'r-',label = 'Test Set Deviance')
plt.legend(loc = 'upper right')
plt.xlabel('Boosting Iteration')
plt.ylabel('Deviance')

feature_importance = clf.feature_importances_
feature_importance = 100*(feature_importance/max(feature_importance))
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0])+0.5
plt.subplot(1,2,2)
plt.barh(pos,feature_importance[sorted_idx],align = 'center')
plt.yticks(pos)
plt.show()


# In[ ]:




