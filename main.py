#!/usr/bin/env python
# coding: utf-8

# ## Link: https://www.youtube.com/watch?v=p_tpQSY1aTs&t=4146s
# 
# Mostly for beginners

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('car data.csv')
df.head()


# In[3]:


df.shape


# In[4]:


# Check unique values of categorical columns

print(df['Fuel_Type'].unique())
print(df['Seller_Type'].unique())
print(df['Transmission'].unique())
print(df['Owner'].unique())


# In[5]:


# Check missing values

df.isna().sum()


# In[6]:


df.describe()


# In[7]:


final_dataset = df[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]


# In[8]:


final_dataset['Current year'] = 2020


# In[9]:


final_dataset.head()


# In[10]:


final_dataset['no_year'] = final_dataset['Current year'] - final_dataset['Year']


# In[11]:


final_dataset


# In[12]:


final_dataset.drop(labels=['Year', 'Current year'], axis=1, inplace=True)
final_dataset.head()


# In[13]:


final_dataset = pd.get_dummies(final_dataset, drop_first=True)
final_dataset.head()


# In[14]:


final_dataset.corr()


# In[15]:


import seaborn as sns


# In[16]:


sns.pairplot(final_dataset)


# In[17]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

corrmat = final_dataset.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20, 20))
sns.heatmap(corrmat, annot=True, cmap='RdYlGn')


# In[18]:


# Independent and dependent features

X = final_dataset.iloc[:, 1:]
y = final_dataset.iloc[:, 0]


# In[19]:


X.head()


# In[20]:


y.head()


# In[21]:


# Feature importance

from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
rfr.fit(X, y)


# In[22]:


print(rfr.feature_importances_)


# In[23]:


plt.figure(figsize=(20, 8))
sns.barplot(X.columns, rfr.feature_importances_)


# In[24]:


feat_importances = pd.Series(rfr.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()


# In[25]:


# Feature importance

from sklearn.ensemble import ExtraTreesRegressor
efr = ExtraTreesRegressor()
efr.fit(X, y)


# In[26]:


plt.figure(figsize=(20, 8))
sns.barplot(X.columns, efr.feature_importances_)


# In[27]:


feat_importances = pd.Series(efr.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()


# In[28]:


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=30)


# In[29]:


xtrain.shape


# In[30]:


rfr = RandomForestRegressor()


# In[31]:


# RandomizedSearchCV

import numpy as np

# No of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=100, stop=1200, num=12)]

# No of features to consider at every split
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num=6)]
# max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]

# Above we are providing int as otherwise we will get them as float which might give error in RandomizedSearchCV


# In[32]:


from sklearn.model_selection import RandomizedSearchCV


# In[33]:


random_grid = {
    'n_estimators': n_estimators,
    'max_features': max_features,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf
}
print(random_grid)


# In[34]:


# Use the random grid to search for best parameters
# First create the base model to tune
rf = RandomForestRegressor()


# In[35]:


rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, scoring='neg_mean_squared_error', n_iter=10, cv=5, verbose=2, random_state=42, n_jobs=1)


# In[36]:


rf_random.fit(xtrain, ytrain)


# In[37]:


predictions = rf_random.predict(xtest)


# In[38]:


predictions


# In[39]:


sns.distplot(ytest-predictions)

# As here the difference between ytest and predictions is showing normal distribution so our prediction is fine
# Because of the difference is very very minimal between ytest and predictions then we will get a very close graph 
# of Gaussian distribution


# In[41]:


plt.scatter(ytest, predictions)

# We also plotted both of them and we can see that the plotting is also linearly available that means our predcition is pretty
# much good
# In this graph also we can see that graph of ytest and predictions are in line and so that is why this model seems to be good


# In[42]:


from sklearn.metrics import r2_score, mean_squared_error


# In[43]:


print(mean_squared_error(ytest, predictions))
print(r2_score(ytest, predictions))


# In[44]:


import pickle

# open file where we want to store the data
file = open('random_forest_regression_model.pkl', 'wb')

# dump information to that file
pickle.dump(rf_random, file)

# This pickle file is a serialized file which will be used for our deployment


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




