import numpy as np
import pandas as pd
from pandas import*
x=np.random.normal(100,70,1000)
y=np.random.normal(80,50,1000)
data=pd.DataFrame({"x":x,"y":y})
data.to_csv("First_Try.csv",sep=',',index=False)
Data=pd.read_csv("First_Try.csv")
x=Data['x']
y=Data['y']
Xtrain=x[:800]
Xtest=x[800:]
Ytrain=y[:800]
Ytest=y[800:]
n=len(data['x'])
Xtrain=np.array([[data['x'][i],data['y'][i]] for i in range(n)])
Ytrain=np.array(data['class'])
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
