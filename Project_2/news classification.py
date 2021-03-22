# -*- coding: utf-8 -*-


from keras.datasets import reuters
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
import numpy as np
import  pandas as pd


# read data
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)



# one hot coding
def vectorize_sequences(sequences, dimension=10000): 
    results = np.zeros((len(sequences), dimension)) 
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


# one hot
x_train = vectorize_sequences(train_data) 
x_test = vectorize_sequences(test_data)


#from keras import to_categorical can realize this function
from keras.utils.np_utils import to_categorical
one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)


# to one hot
def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results

one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)


# def model
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.compile(optimizer=optimizers.RMSprop(lr=0.001), 
loss=losses. categorical_crossentropy, 
metrics=[metrics. categorical_accuracy])



# train test split
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]


history = model.fit(partial_x_train, 
            partial_y_train,
            epochs=20,
            batch_size=512,
            validation_data=(x_val, y_val))



# use different epochs and batch_size
#model.fit(x_train, one_hot_train_labels, epochs=9, batch_size=512)
model.fit(x_train, one_hot_train_labels, epochs=9, batch_size=256)


# evaluation
loss,accuracy = model.evaluate(x_test, one_hot_test_labels)
print('loss =', loss, ' accuracy =', accuracy)


# predict
classes = model.predict(x_test)
print('predict sample：',len(classes))
print("acc:\n",classes)