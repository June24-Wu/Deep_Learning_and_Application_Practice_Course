# -*- coding: utf-8 -*-


from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np
import pandas as pd




# read data
data = pd.read_csv('.\dataset\winequality-red.csv' ,delimiter=';' ,header=0)


# data preprocessing
x=data.iloc[:,:12].values.astype(float)

y=data.iloc[:,11].values.astype(int)

# normalization
mean=x.mean(axis=0)
x -=mean
std=x.std(axis=0)
x /=std
d=[]
for i in range(len(x)):
    for j in range(x.shape[1]):
        if x[i,j]>3. or x[i,j]<-3. :
            d.append(i)
            i+=1
x=np.delete(x,d,axis=0)
x=np.delete(x,11,axis=1)
y=np.delete(y,d,axis=0)
data=np.column_stack((x,y))
train_test=data[:,:11]
labels=data[:,11]
# train test split
train_data=train_test[:1000,:]
test_data=train_test[1000:,:]
train_labels=labels[:1000]
test_labels=labels[1000:]

val_x_data=train_data[:200,:]
val_y_data=train_labels[:200]
x_train=train_data[200:,:]
y_train=train_labels[200:]


# def model
model=Sequential()
model.add(Dense(64,activation='relu',input_shape=(11,))) #隐层1
model.add(Dense(64,activation='relu')) #隐层2
model.add(Dropout(0.25)) #隐层2随机失活25%
model.add(Dense(1))
model.compile(loss='mse', optimizer='rmsprop', metrics=["accuracy"])

history=model.fit(x_train,y_train,epochs=50,batch_size=1,validation_data=(val_x_data,val_y_data))


# evaluate model ( acc & loss)
results=model.evaluate(test_data,test_labels)
print(results)
#[0.4725847185574918, 0.6031042128603105]

# predict
predict1=model.predict(test_data)
print(predict1)