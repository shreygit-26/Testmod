#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


#importing packages
import pandas as pd
import numpy as np


# In[ ]:


# Importing the dataset
train = pd.read_csv("/content/drive/MyDrive/Samplemod/NB Normal.csv")
train['Fault'] = 0
test = pd.read_csv("/content/drive/MyDrive/Samplemod/IR - 7 Fault.csv")
test['Fault'] = 1


# In[ ]:


dataset = train.append(test)
dataset


# In[ ]:


X = dataset.iloc[:, 0:2].values
y = dataset.iloc[:, 2]


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state
= 0)


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[ ]:


y_train


# In[ ]:


# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


# In[ ]:


# Predicting the Test set results
y_pred = classifier.predict(X_test)


# In[ ]:


classifier.score(X_test, y_test)


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
index = ['Normal','Inner Race (0.007")'] 
columns = ['Normal','Inner Race (0.007")'] 
cm_df = pd.DataFrame(cm,columns,index) 
plt.figure(figsize=(10,4))
sn.set(font_scale=1.4) # for label size
sn.heatmap(cm_df, annot=True, fmt='g') # font size
plt.title('Confusion matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[ ]:


from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred, target_names=['Normal','Inner Race (0.007")'
])
print(cr)

