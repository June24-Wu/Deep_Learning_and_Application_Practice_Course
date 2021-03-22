# -*- coding: utf-8 -*-



from keras.datasets import mnist
from keras import models
from keras import layers
from keras.layers import  Dropout
from keras.utils import to_categorical




# import data
(train_images,train_labels), (test_images,test_labels)=mnist.load_data()


# def model
model=models.Sequential()
model.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))


# using lesee dense layers
#model.add(layers.Dense(256,activation='relu',input_shape=(28*28,)))



model.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))


# add drop out
model.add(Dropout(0.1))
model.add(layers.Dense(10,activation='softmax'))



# using categorical_crossentropy
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
# using mse
#model.compile(optimizer='rmsprop',loss='mse',metrics=['accuracy'])


# using adam optimizer
#model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])



train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model.fit(train_images, train_labels, epochs=10, batch_size=128)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc:', test_acc)


#实验心得：
# 1.every time neural network have different result
# 2.when doing classification task , use categorical_crossentropy as loss monitor
#   when doing regression task , use mse as loss monitor
# 3.classifcation task use rmsprop or sgd as optimizer will be better , but regression task need use
#   adam as optimizer to get the better results
