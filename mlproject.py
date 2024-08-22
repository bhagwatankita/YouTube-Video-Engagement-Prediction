import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = "C:/Users/ANKITA BHAGWAT/Downloads/train_lyst1720633807653.csv"
data_train = pd.read_csv(path)
data_train.head()
data_train.shape
category={'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'H':8}
data_train["category"]=data_train["category"].map(category)
data_train.head()

data_train=data_train[data_train.views!='F']
data_train=data_train[data_train.likes!='F']
data_train=data_train[data_train.dislikes!='F']
data_train=data_train[data_train.comment!='F']

data_train.head()

data_train["views"] = pd.to_numeric(data_train["views"])
data_train["comment"] = pd.to_numeric(data_train["comment"])
data_train["likes"] = pd.to_numeric(data_train["likes"])
data_train["dislikes"] = pd.to_numeric(data_train["dislikes"])
data_train["adview"] = pd.to_numeric(data_train["adview"])

column_vidid=data_train['vidid']

from sklearn.preprocessing import LabelEncoder
data_train['duration']=LabelEncoder().fit_transform(data_train['duration'])
data_train['vidid']=LabelEncoder().fit_transform(data_train['vidid'])
data_train['published']=LabelEncoder().fit_transform(data_train['published'])

data_train.head()

import datetime
import time

def checki(x):
    y = x[2:]
    h = ''
    m = ''
    s = ''
    mm = ''
    P = ['H','M','S']
    for i in y:
        if i not in P:
            mm+=i
        else:
            if(i=="H"):
                h = mm
                mm =''
            elif(i == "M"):
                m = mm
                mm = ''
            else:
                s = mm
                mm = ''
    if(h== ''):
        h = '00'
    if(m == ''):
        m = '00'
    if(s== ''):
        s = '00'
    bp = h+':'+m+':'+s
    return bp

path = "C:/Users/ANKITA BHAGWAT/Downloads/train_lyst1720633807653.csv"
train = pd.read_csv(path)

mp = pd.read_csv("C:/Users/ANKITA BHAGWAT/Downloads/train_lyst1720633807653.csv")["duration"]
time = mp.apply(checki)

def func_sec(time_string):
    h, m, s = time_string.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)
    
time1=time.apply(func_sec)

data_train["duration"]=time1
data_train.head()

plt.hist(data_train["category"])
plt.show()
plt.plot(data_train["adview"])
plt.show()

data_train = data_train[data_train["adview"] <2000000]

#heatmap

import seaborn as sns

f, ax = plt.subplots(figsize=(10, 8))
corr = data_train.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220,10, as_cmap=True), square=True, ax=ax,annot=True)
plt.show()

Y_train = pd.DataFrame(data = data_train.iloc[:, 1].values,columns = ['target'])
data_train = data_train.drop(["adview"],axis=1)
data_train = data_train.drop(["vidid"],axis=1)
data_train.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_train, Y_train, test_size=0.2, random_state=42)

X_train.shape

from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from sklearn import metrics
import numpy as np
scaler = MinMaxScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)

X_train.mean()

#matrix
from sklearn import metrics
import numpy as np

def print_error(X_test, y_test, model_name):
    prediction = model_name.predict(X_test)
    print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, prediction))
    print("Mean Squared Error:", metrics.mean_squared_error(y_test, prediction))
    print("Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(y_test, prediction)))

# Linear Regression
from sklearn.linear_model import LinearRegression
linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)
print_error(X_test, y_test, linear_regression)

# Decision Tree
from sklearn.tree import DecisionTreeRegressor
decision_tree = DecisionTreeRegressor()
decision_tree.fit(X_train, y_train)
print_error(X_test, y_test, decision_tree)

from sklearn.ensemble import RandomForestRegressor
n_estimators = 200
max_depth = 25
min_samples_split = 15
min_samples_leaf = 2
random_forest = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth, min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf)
random_forest.fit(X_train, y_train.values.ravel())
print_error(X_test, y_test, random_forest)

from sklearn.svm import SVR
supportvector_regressor = SVR()
supportvector_regressor.fit(X_train,y_train)
print_error(X_test, y_test, linear_regression)

#artificial neural network
import keras
from keras.models import Sequential
from keras.layers import Dense

ann = keras.models.Sequential([
                                Dense(6, activation="relu",
                                input_shape=X_train.shape[1:]),
                                Dense(6, activation="relu"),
                                Dense(1)
                                ])
                                

optimizer=keras.optimizers.Adam()
loss=keras.losses.mean_squared_error
ann.compile(optimizer=optimizer,loss=loss,metrics=["mean_squared_error"])

history=ann.fit(X_train, y_train, epochs=100)

ann.summary()

print_error(X_test, y_test, ann)

import joblib
joblib.dump(decision_tree, "decisiontree_youtubeadview.pkl")

ann.save("ann_youtubeadview.keras")

