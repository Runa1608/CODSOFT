#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score



# In[4]:


iris = pd.read_csv('IRIS.csv')
iris.head()


# In[5]:


X = iris[['sepal_length', 'petal_length']]
y = iris['species']


# In[6]:


iris.duplicated().sum()


# In[7]:


iris['species'].value_counts()


# In[8]:


species_numeric = {'Iris-setosa':1 , 'Iris-versicolor':2 , 'Iris-virginica':3}
iris.species = [species_numeric[i] for i in iris.species]


# In[9]:


iris.sample(10)


# In[10]:


sns.countplot(x='species', data=iris)


# In[11]:


sns.heatmap(iris.corr(),annot =True)


# In[12]:


x = iris.drop(columns=['species'])

y = iris['species']


# In[13]:


x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = 0.33 ,random_state =42)


# In[14]:


model = LinearRegression()
model.fit(x, y)


# In[15]:


model.score(x, y)


# In[16]:


model.intercept_


# In[17]:


y_pred = model.predict(x_test)
print(y_pred)


# In[18]:


sns.scatterplot(x=y_test, y=y_pred, color='blue', label='Actual Data points')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', label='Ideal Line')
plt.legend()
plt.show()


# In[ ]:




