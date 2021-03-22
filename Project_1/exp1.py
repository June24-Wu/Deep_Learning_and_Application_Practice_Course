# -*- coding: utf-8 -*-



from keras.models import Sequential
from keras.layers import Dense, Dropout # import keras
from keras.utils import np_utils
# Model
model=Sequential()
model.add(Dense(16,activation='relu',input_shape=(4,))) # Hiden Layer 1
model.add(Dense(16,activation='relu')) # Hiden Layer 2
model.add(Dropout(0.25)) # drop out
model.add(Dense(3,activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('D:\downloadgg\iris.data', header = None)
data.columns = ['sepal length','sepal width','petal length','petal width','class']

# get X
X = data.iloc[:,0:4].values.astype(float)

# transformation
data.loc[ data['class'] == 'Iris-setosa', 'class' ] = 0
data.loc[ data['class'] == 'Iris-versicolor', 'class' ] = 1
data.loc[ data['class'] == 'Iris-virginica', 'class' ] = 2


# get Y
y = data.iloc[:,4].values.astype(int)


# train test split
train_x, test_x, train_y, test_y = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
mean=train_x.mean(axis=0) 
std=train_x.std(axis=0)
train_x=(train_x-mean)/std
test_x=(test_x-mean)/std


train_y_ohe = np_utils.to_categorical(train_y, 3)
test_y_ohe = np_utils.to_categorical(test_y, 3)
print('head test 5：', test_y[0:5])
print('head test 5 one hot coding：\n',test_y_ohe[0:5])
model.fit(train_x, train_y_ohe, epochs=20, batch_size=1, verbose=2, validation_data=(test_x,test_y_ohe))
#评估模型
loss, accuracy = model.evaluate(test_x, test_y_ohe, verbose=2)
print('loss = {},accuracy = {} '.format(loss,accuracy) )
#查看预测结果，属于各类的概率
classes = model.predict(test_x, batch_size=1, verbose=2)
print('sample num ：',len(classes))
print("acc:\n",classes)