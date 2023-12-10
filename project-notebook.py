#!/usr/bin/env python
# coding: utf-8

# <h1>Loading in the Data Sets</h1>

# In[1]:


import pandas as pd

results = pd.read_csv('results.csv')
qualifying = pd.read_csv('qualifying.csv')


# <h1>Cleaning the Data</h1>

# In[2]:


results.head()


# In[3]:


results_features = pd.DataFrame(results.columns)
results_features = results_features.rename(columns={0: 'Original Features'})
results_features


# In[4]:


results=results.drop(columns=['resultId', 'number', 'positionText', 'positionOrder', 'points', 'time', 'milliseconds', 'fastestLapTime', 'fastestLapSpeed', 'fastestLap', 'statusId'])
results.head()


# In[5]:


results_features = pd.DataFrame(results.columns)
results_features = results_features.rename(columns={0: 'Selected Features'})
results_features


# In[6]:


qualifying.head()


# In[7]:


quali_features = pd.DataFrame(qualifying.columns)
quali_features = quali_features.rename(columns={0: 'Original Features'})
quali_features


# In[8]:


qualifying=qualifying.drop(columns=['qualifyId', 'number', 'q1', 'q2', 'q3'])
qualifying.head()


# In[9]:


quali_features = pd.DataFrame(qualifying.columns)
quali_features = quali_features.rename(columns={0: 'Selected Features'})
quali_features


# In[10]:


#renaming the 'position' column to make it clear that this is the qualifying position, not the final race position
qualifying = qualifying.rename(columns={'position': 'qualiPosition'})
qualifying.head()


# In[11]:


quali_features = pd.DataFrame(qualifying.columns)
quali_features = quali_features.rename(columns={0: 'Selected Features'})
quali_features


# <h1>Merging the Datasets</h1>

# In[12]:


mergedDf=results.merge(qualifying, on=['raceId', 'driverId', 'constructorId'], suffixes=('_race', '_quali'))


# In[13]:


mergedDf.head()


# In[14]:


cols=list(mergedDf.columns.values)
mergedDf=mergedDf[cols[0:4] + cols[5:8] + [cols[4]]]
mergedDf.head()


# In[15]:


#renaming 'position' to 'position_race' so it is more clear that this is the final race position
mergedDf = mergedDf.rename(columns={'position': 'position_race'})


# In[16]:


mergedDf


# <h1>Cleaning the Data Some More</h1>

# In[17]:


#mergedDf.loc[mergedDf['milliseconds'] == '\\N', 'milliseconds'] = '-1'
mergedDf.loc[mergedDf['position_race'] == '\\N', 'position_race'] = '-1'
#mergedDf.loc[mergedDf['fastestLap'] == '\\N', 'fastestLap'] = '-1'
mergedDf.loc[mergedDf['rank'] == '\\N', 'rank'] = '-1'
#mergedDf.loc[mergedDf['fastestLapSpeed'] == '\\N', 'fastestLapSpeed'] = '-1'
mergedDf = mergedDf[mergedDf['position_race'] != '21']
mergedDf = mergedDf[mergedDf['position_race'] != '22']
mergedDf = mergedDf[mergedDf['position_race'] != '23']
mergedDf = mergedDf[mergedDf['position_race'] != '24']
mergedDf = mergedDf.astype(int)
mergedDf


# <h1>Loading the Final Dataset into a CSV File</h1>

# In[18]:


mergedDf.to_csv('mergedF1Data.csv', index=False)


# <h1>Implementing Decision Tree</h1>

# In[19]:


X = mergedDf.iloc[:, 0:7]
y = mergedDf.iloc[:, 7]


# In[20]:


from sklearn.model_selection import train_test_split


# In[21]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


# In[22]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[23]:


from sklearn.tree import DecisionTreeClassifier


# In[24]:


decisionTree = DecisionTreeClassifier()


# In[25]:


decisionTree.fit(X_train, y_train)


# In[26]:


y_pred = decisionTree.predict(X_test)


# In[27]:


from sklearn.metrics import confusion_matrix


# In[28]:


print(confusion_matrix(y_test, y_pred))


# In[29]:


import seaborn as sn
import matplotlib.pyplot as plt
cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(cm)
plt.figure(figsize=(10, 7))
sn.set(font_scale=0.8)
cmap = sn.cm.rocket_r
sn.heatmap(df_cm, annot=True, cmap=cmap)
plt.show()


# In[30]:


from sklearn.metrics import classification_report


# In[31]:


print(classification_report(y_test, y_pred, zero_division=0))


# In[32]:


decisionTree.feature_importances_


# In[33]:


featImportances = pd.DataFrame(decisionTree.feature_importances_, index = X.columns)


# In[34]:


featImportances


# <h1>Implementing Random Forest</h1>

# In[35]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(criterion='gini', max_depth=8, min_samples_split=10, random_state=5)


# In[36]:


rfc.fit(X_train, y_train)


# In[37]:


y_pred_rfc = rfc.predict(X_test)


# In[38]:


print(classification_report(y_test, y_pred_rfc, zero_division=0))


# <h1>Implementing Linear Regression</h1>

# In[39]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# In[40]:


print(mergedDf)


# In[41]:


x = mergedDf.drop(['position_race'],axis=1).values
y = mergedDf['position_race'].values


# In[42]:


print(x)


# In[43]:


print(y)


# In[44]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)


# In[45]:


ml=LinearRegression()
ml.fit(x_train,y_train)


# In[46]:


y_pred = ml.predict(x_test)
y_pred=np.rint(y_pred)
print(y_pred)


# In[47]:


np.rint(ml.predict([[18, 1, 1, 1, 58, 2, 1]]))


# In[48]:


from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# In[49]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# In[50]:


plt.figure(figsize=(15,10))
plt.scatter(y_test,y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted')


# In[51]:


pred_y_df = pd.DataFrame({'Actual Value': y_test, 'Predicted Value': y_pred, 'Difference': y_test-y_pred})
pred_y_df[0:20]


# <h1>Binning the Data</h1>
# <h3>bin 1 = places 1-5, bin 2 = places 6-10, bin 3 = places 11-15, bin 4 = places 16-21, bin 5 = DNF (-1)</h3>

# In[52]:


mergedDf.head()


# In[53]:


position_race = mergedDf['position_race']
position_race = position_race.replace([1, 2, 3, 4, 5], 1)
mergedDf['position_race'] = position_race
mergedDf.head()


# In[54]:


position_race = mergedDf['position_race']
position_race = position_race.replace([6, 7, 8, 9, 10], 2)
position_race = position_race.replace([11, 12, 13, 14, 15], 3)
position_race = position_race.replace([16, 17, 18, 19, 20, 21], 4)
position_race = position_race.replace([-1], 5)
mergedDf['position_race'] = position_race


# In[55]:


X = mergedDf.iloc[:, 0:7]
y = mergedDf.iloc[:, 7]


# In[56]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


# In[57]:


decisionTree = DecisionTreeClassifier()


# In[58]:


decisionTree.fit(X_train, y_train)


# In[59]:


y_pred = decisionTree.predict(X_test)


# In[60]:


print(confusion_matrix(y_test, y_pred))


# In[61]:


cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(cm)
plt.figure(figsize=(10, 7))
sn.set(font_scale=0.8)
cmap = sn.cm.rocket_r
sn.heatmap(df_cm, annot=True, cmap=cmap)
plt.show()


# In[62]:


print(classification_report(y_test, y_pred, zero_division=0))


# <h1>Predicting if a Driver Finished on the Podium (top 3 places) or Not</h1>

# <h3>position_race of 1 = did finish in top 3, position_race of 0 = did not finish in top 3</h3>

# In[63]:


originalDf = pd.read_csv('mergedF1Data.csv')
originalDf.head()


# In[64]:


position_race = originalDf['position_race']
position_race = position_race.replace([1, 2, 3], 1)
position_race = position_race.replace([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, -1], 0)
originalDf['position_race'] = position_race
originalDf.head()


# In[65]:


X = originalDf.iloc[:, 0:7]
y = originalDf.iloc[:, 7]


# In[66]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


# In[67]:


decisionTree = DecisionTreeClassifier()


# In[68]:


decisionTree.fit(X_train, y_train)


# In[69]:


y_pred = decisionTree.predict(X_test)


# In[70]:


print(confusion_matrix(y_test, y_pred))


# In[71]:


cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(cm)
plt.figure(figsize=(10, 7))
sn.set(font_scale=0.8)
cmap = sn.cm.rocket_r
sn.heatmap(df_cm, annot=True, cmap=cmap)
plt.show()


# In[72]:


print(classification_report(y_test, y_pred, zero_division=0))


# In[73]:


featImportances2 = pd.DataFrame(decisionTree.feature_importances_, index = X.columns)
featImportances2 = featImportances2.rename(columns={0: 'Feature Importance'})
featImportances2

