#!/usr/bin/env python
# coding: utf-8

# In[38]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve


# In[39]:


df = pd.read_csv('Titanic-Dataset.csv')


# In[40]:


df.head()


# In[41]:


df.describe()


# In[42]:


df.isnull().sum()


# In[43]:


df["Survived"].value_counts()


# In[44]:


df = df.drop(columns='Cabin', axis=1)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)


# In[45]:


df.head()


# In[46]:


df['Age'].fillna(df['Age'].mean(), inplace=True)
df.head()


# In[47]:


df.replace({'Sex': {'male': 1, 'female': 0}, 'Embarked': {'S': 0, 'C': 1, 'Q': 2}}, inplace=True)
df.head()


# In[48]:


df.describe()


# In[49]:


df['FamilySize'] = df['SibSp'] + df['Parch']
df['IsAlone'] = np.where(df['FamilySize'] == 0, 1, 0)


# In[50]:


df.describe()


# In[51]:


X = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Survived', 'SibSp', 'Parch', 'FamilySize'])
Y = df['Survived']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
model = LogisticRegression(max_iter=100)
# Hyperparameter tuning
param_grid = {'C': [0.1, 1, 10, 100], 'solver': ['liblinear']}
grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, Y_train)
best_model = grid.best_estimator_

best_model.fit(X_train, Y_train)


# In[52]:


Y_train_pred = best_model.predict(X_train)
Y_test_pred = best_model.predict(X_test)
training_accuracy = accuracy_score(Y_train, Y_train_pred)
test_accuracy = accuracy_score(Y_test, Y_test_pred)
print("Training Accuracy Score:", training_accuracy)
print("Test Accuracy Score:", test_accuracy)


# In[53]:


conf_matrix = confusion_matrix(Y_test, Y_test_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues")
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()


# In[54]:


print("Classification Report:\n", classification_report(Y_test, Y_test_pred))


roc_auc = roc_auc_score(Y_test, best_model.predict_proba(X_test)[:, 1])
fpr, tpr, _ = roc_curve(Y_test, best_model.predict_proba(X_test)[:, 1])

plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()



# In[55]:


train_sizes, train_scores, test_scores = learning_curve(best_model, X, Y, cv=5, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.2, 1.0, 10))

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean, 'o-', color='orange', label='Training score')
plt.plot(train_sizes, test_mean, 'o-', color='pink', label='Cross-validation score')

plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='orange', alpha=0.2)
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color='pink', alpha=0.2)

plt.xlabel('Training Size')
plt.ylabel('Accuracy Score')
plt.title('Learning Curves')
plt.legend(loc="best")
plt.show()


# In[ ]:





# In[ ]:




