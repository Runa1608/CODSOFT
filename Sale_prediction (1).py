#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

sales_data = pd.read_csv('advertising.csv')
sales_data.head()


# In[2]:


sns.pairplot(sales_data)
plt.show()


# In[3]:


plt.figure(figsize=(10, 6))
sns.heatmap(sales_data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# In[4]:


sales_data.isnull().sum()


# In[5]:


sales_data.describe()


# In[6]:


print(sales_data.columns)


# In[7]:


X = sales_data[['TV', 'Radio', 'Newspaper']]
y = sales_data['Sales']



# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# In[9]:


print("Shape of X_test:", X_test.shape)
print("Shape of y_test:", y_test.shape)


# In[10]:


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared Score:", r2)

new_advertising_expenditures = [[200, 60, 20]] 
predicted_sales = model.predict(new_advertising_expenditures)
print("Predicted Sales:", predicted_sales)


# In[11]:


sns.scatterplot(x=y_test, y=y_pred, color='blue', label='Actual sales')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', label='Ideal Line')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:




