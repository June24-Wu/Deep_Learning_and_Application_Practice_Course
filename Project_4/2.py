# -*- coding: utf-8 -*-
# 导入cifar10数据集
from keras.datasets import cifar10
from keras.utils import to_categorical

# 从keras中读取数据集
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
# 数据预处理：格式化、标准化
train_images = train_images.reshape((50000, 32, 32, 3))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 32, 32, 3))
test_images = test_images.astype('float32') / 255
# 将标签转化为独热矩阵
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)



from keras.preprocessing.image import ImageDataGenerator

datagen_train = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=40,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

datagen_train.fit(train_images)

# 导入库
from keras import layers
from keras import models
from keras.layers import Dense, Dropout
from keras.applications import VGG16

base = VGG16(input_shape=(32,32,3),include_top=False, weights='imagenet')
# x = layers.Flatten()(base)
# x = layers.Dense(128, activation='relu')(x)
# x = Dropout(0.25)(x)
# x = layers.Dense(10, activation='softmax')(x)
model = models.Sequential()
model.add(base)

# 提取特征
base.trainable = True

# 增加全连接
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(layers.Dense(10, activation='softmax'))

# optimization
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc'])


model.summary()
history = model.fit(train_images,train_labels,
                              batch_size=20,  # batch_size
                              epochs=15,
                              verbose=1,)  # batch_size

# 绘制训练损失和验证损失随迭代次数变化图
import matplotlib.pyplot as plt

loss = history.history['loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 绘制训练精度和验证精度随迭代次数变化图
plt.clf()  # clear figure(清空图像)
acc = history.history['acc']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.title('Training accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 在测试集上评估模型性能
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_loss={},test_acc={}'.format(test_loss,test_acc))




