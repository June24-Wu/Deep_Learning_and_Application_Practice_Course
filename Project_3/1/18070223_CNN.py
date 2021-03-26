

# load data
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import layers
from keras import models

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


# data preprocessing
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255



# one hot Y
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)



# Model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))


model.summary()

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# fit model
model.fit(train_images, train_labels, epochs=10, batch_size=64)
#在测试集上评估模型性能
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('test_loss={},test_acc={}'.format(test_loss,test_acc))

'''
Dense Result:  test_loss=0.0938,test_acc=0.978
CNN Result: test_loss=0.0361,test_acc=0.992

本次实验采用卷积神经网络对mnist数据集的手写数字进行了识别，最后与全连接层分类器构成CNN
网络模型，最后的测试精度达到了99.2%，而全连接层方法的精度为97.8%，
说明CNN的效果更好'''
