#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import matplotlib.pyplot as plt


df= pd.read_csv("/Users/arjuntakoria/Downloads/AB_NYC_2019.csv")
df.head()


# In[10]:


#checking for missing values:

miss_val= df.isnull().sum()
print(miss_val)


# In[12]:


#oUTLIERS

import seaborn as sns

#boxplot:

sns.boxplot(x='price',data=df)


# In[13]:


q1= df['price'].quantile(0.25)
q3= df['price'].quantile(0.75)
IQR =q3-q1

print(IQR)


# In[18]:


df= pd.read_csv("/Users/arjuntakoria/Downloads/titanic (1)/train.csv")
df.head()


# In[22]:


#making a histogram for the topic survived
# than choosing lables on both the axis

plt.hist(df["Survived"],bins=2,color="blue")
plt.xlable("Survived")
plt.ylable("Frequency")
plt.title("Survival statistics")
plt.show()


# In[23]:


#colleration matrix

import seaborn as sns
corr_mat=df.corr()
print(corr_mat)


# In[31]:


#Making a heat map:

plt.figure(figsize=(10,8))
sns.heatmap(corr_mat,annot= True,cmap='coolwarm',fmt='.2f')
plt.show()


# In[39]:


df= pd.read_csv("/Users/arjuntakoria/Downloads/Housing.csv")
print(df)


# In[40]:


df=pd.get_dummies(df,columns=["mainroad"])
df=pd.get_dummies(df,columns=["guestroom"])
df=pd.get_dummies(df,columns=["basement"])
df=pd.get_dummies(df,columns=["hotwaterheating"])
df=pd.get_dummies(df,columns=["airconditioning"])
df=pd.get_dummies(df,columns=["prefarea"])
df=pd.get_dummies(df,columns=["furnishingstatus"])

print(df)


# In[41]:


import numpy as np
from sklearn.linear_model import LinearRegression

#extract square footage and prices from the data
#we have extracted them into two different list square_foot and prices.

square_foot= df['area'].values
prices=df['price'].values

#Reshape the data so that it's 1D
#it is the x variable of the:  y=mx+c
X= square_foot.reshape(-1,1)


#create and train linear regression model
model=LinearRegression()
model.fit(X,prices)  #prices is the y variable which we are precicting.

#predict house prices
new_square_foot= np.array([[1900]])
predicted_price= model.predict(new_square_foot)
print(predicted_price)


# In[42]:


#now predicting when more than one factor is involved:

from sklearn.model_selection import train_test_split

#extrtact feature except prices:

features=df.drop('price',axis=1).values
prices=df['price'].values


#splitiing data into testing and training sets
X_train,X_test,y_train,y_test= train_test_split(features,prices,test_size=0.2,random_state=42)

#create and train linear regression model

model=LinearRegression()
model.fit(X_train,y_train)

#make prediction on the test set
predicted_test= model.predict(X_test)


# In[50]:


#skatterplot

plt.figure(figsize=(10,6))
plt.scatter(y_test,predicted_test,color='blue')
plt.title("Actual vs predicted price")
plt.grid(True)
plt.show()

#residualplt

residuals= y_test-predicted_price
plt.figure(figsize=(10,6))
plt.title("Residual Model")
plt.scatter(predicted_test,residuals,color='green')
plt.grid(True)
plt.show()


# In[ ]:


#logistics regresion:

#sigmoid function for prediction and classification

def sigmoid(z):
    return 1 /(1+np.exp(-z))


# In[54]:


df= pd.read_csv("/Users/arjuntakoria/Downloads/framingham.csv")

n= df.isnull().any()
print(n)


# In[59]:


#as they have missing value, we can fill them with their mean:

df['education'].fillna(df['education'].mean(), inplace= True)
df['cigsPerDay'].fillna(df['cigsPerDay'].mean(), inplace= True)
df['BPMeds'].fillna(df['BPMeds'].mean(), inplace= True)
df['totChol'].fillna(df['totChol'].mean(), inplace= True)
df['BMI'].fillna(df['BMI'].mean(), inplace= True)
df['heartRate'].fillna(df['heartRate'].mean(), inplace= True)
df['glucose'].fillna(df['glucose'].mean(), inplace= True)


print(df)


# In[60]:


#exctracting features from the data:

features=df.drop('TenYearCHD',axis=1).values
labels=df['TenYearCHD'].values

#split training and testing sets:

X_train,X_test,y_train,y_test= train_test_split(features,labels,test_size=0.2,random_state=42)


# In[61]:


from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_train)


# In[63]:


#create regression model:
model=LinearRegression()
model.fit(X_train_scaled,y_train)


# In[64]:


predicted=model.predict(X_test_scaled)

