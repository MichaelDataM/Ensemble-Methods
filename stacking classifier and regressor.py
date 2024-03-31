#!/usr/bin/env python
# coding: utf-8

# In[7]:


from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# In[17]:


X,y = load_iris(return_X_y = True)
estimators = [
    ('rf',RandomForestClassifier(n_estimators = 10,random_state = 42)),
    ('svr',Pipeline([('scaler',StandardScaler()),
                   ('svc',LinearSVC(dual= False,random_state = 42))]))
]
# Difference between pipeline and mke_pipeline: make_pipeline is better because it is not necessary to name the model for each one
clf = StackingClassifier(
estimators = estimators,
final_estimator = LogisticRegression())

X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,random_state = 42)
clf.fit(X_train,y_train)


# In[ ]:




