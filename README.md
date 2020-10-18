# 搭建屬於你的第一個Neural Network
###### tags: `Object Detection` `Python`
## 1 Input Data
|CSV|XLS|
|--|--|
|plain text format|binary file format|
5,6,3,12|Excel Sheets|

## 2 Build Data
```python 
np.random.normal(mean,標準差,quantity)
```
## 3 save numpy array as csv file
```python=
import numpy as np
import pandas as pd
from pandas import*
x=np.random.normal(100,70,1000)
y=np.random.normal(80,50,1000)
data=pd.DataFrame({"x":x,"y":y})
data.to_csv("First_Try.csv",sep=',',index=False)
```
## 4 Get and Split data
a) array
```python=
Data=pd.read_csv("First_Try.csv")
x=Data['x']
y=Data['y']
Xtrain=x[:800]
Xtest=x[800:]
Ytrain=y[:800]
Ytest=y[800:]
```
b) Numpy array
```python=
n=len(data['x'])
Xtrain=np.array([[data['x'][i],data['y'][i]] for i in range(n)])
Ytrain=np.array(data['class'])
```

## 5 Build Neural Network Model
```python=
model=Sequential()
model.add(Dense(25,input_shape=(2,),activation='sigmoid'))
model.add(Dense(21,activation='relu'))
model.add(Dense(12,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy',optimizer=SGD(0.9),metrics=['accuracy'])
model.fit(Xtrain,Ytrain,batch_size=200,epochs=10,verbose=1)
score = model.evaluate(Xtrain, Ytrain, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```
## Note
1. 一樣的NN每次估計的acc不一樣(隨機性)
Ex:
a. 初始化: weight
b. 正則化: dropout
c. 層: 詞嵌入
d. 最優化: stochastic optimizing
2. 有時候acc 高到92%不要太高興，好的NN經得起多次驗證
3. sigmoid + relu(適合中間層) + sigmoid 是可行的
4. 設data=1000 batch_size=200,epochs=10，則會在一個epoch隨機丟入並處理5次資料
5. Solution for overfitting and underfitting (acc↓):
a. 增加資料量
b. 在正確的epoch停止 => dropout layer
c. Reduce layer
d. Change optimizer: sigmoid=>自主改變梯度的Adam(RMSprop + momentum)
e. L1/L2 Regularization
6. 一維Gaussian Distribution資料不適合用NN分析處理 => 直接lineral regression
7. evaluate要跑的是training data
8. Logistics Regression 最適合用Softmax+Sigmoid
