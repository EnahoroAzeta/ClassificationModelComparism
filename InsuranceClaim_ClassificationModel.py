#!/usr/bin/env python
# coding: utf-8

# In[17]:


#Numpy deals with large arrays and linear algebra
import numpy as np

import pandas as pd 
 
# Metrics for Evaluation of model Accuracy and F1-score
from sklearn.metrics  import f1_score,accuracy_score
 
#Importing the Decision Tree from scikit-learn library
from sklearn.tree import DecisionTreeClassifier

#Importing the Gradient Descent Classifier from scikit-learn library
from sklearn.linear_model import SGDClassifier

#Importing the KNN from scikit-learn library
from sklearn.neighbors import KNeighborsClassifier
 
# For splitting of data into train and test set
from sklearn.model_selection import train_test_split


filePath = '/Users/apple/Documents/Kaggle/ClassificationModel/insurance2.csv'
filePath2 = '/Users/apple/Documents/Kaggle/ClassificationModel/insurance3r2.csv'

insurance1_df = pd.read_csv(filePath)
X = insurance1_df
y = X.pop('insuranceclaim')


# Split dataset
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)


# In[11]:


X_train


# # DecisionTreeClassifier

# In[15]:



model = DecisionTreeClassifier()
model.fit(X_train,y_train)

prediction = model.predict(X_valid)
print(accuracy_score(y_valid,prediction))
print(f1_score(y_valid,prediction))


# # KNN (k-nearest neighbor)

# In[18]:


KNN_model = KNeighborsClassifier()
KNN_model.fit(X_train,y_train)

prediction = KNN_model.predict(X_valid)
print(accuracy_score(y_valid,prediction))
print(f1_score(y_valid,prediction))


# # Gradient Descent Classifier

# In[19]:


SGDC_model = SGDClassifier()
SGDC_model.fit(X_train,y_train)

prediction = SGDC_model.predict(X_valid)
print(accuracy_score(y_valid,prediction))
print(f1_score(y_valid,prediction))


# In[ ]:




