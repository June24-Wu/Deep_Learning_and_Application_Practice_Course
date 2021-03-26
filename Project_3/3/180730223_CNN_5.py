
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras import layers
from keras import models
from keras.layers import Dense,Dropout

# load data
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
# data preprocessing
train_images = train_images.reshape((50000, 32, 32, 3))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 32, 32, 3))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


# Model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(Dropout(0.25))#随机失活25%
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(Dropout(0.25))#随机失活25%
#model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(layers.Dense(10, activation='softmax'))


model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc'])


# Model Summary
model.summary()


# validation
x_val = train_images[:5000]
partial_x_train = train_images[5000:]
y_val = train_labels[:5000]
partial_y_train = train_labels[5000:]

history=model.fit(partial_x_train,partial_y_train, 
                  epochs=18, batch_size=64,validation_data=(x_val, y_val))

# plot
import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# plot
plt.clf()   # clear figure(清空图像)
acc= history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# Evaluation
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('test_loss={},test_acc={}'.format(test_loss,test_acc))

'''选择5000个样本进行Validation，利用不同卷积核数的卷积层和池化层
以及DropOut搭建卷积神经网络，经过一系列的调整和试验（比如我将第一块
的卷积核数增加至64，第二块增加至128）结构如上。
最开始我尝试了使用更长的周期来进行训练，而后来发现过拟合严重，
所以经过调试之后，选择了epochs=18，最终达到了78.3%的验证精度。'''