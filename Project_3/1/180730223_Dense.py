
from keras import models,layers
from keras.layers import Dense,Dropout
from keras.models import Sequential
from keras.datasets import mnist
from keras.utils import to_categorical

# load data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# check data：
print('Train Shape：',train_images.shape)
print('Test Shape：',test_images.shape)
print('Len test：',len(train_labels))
print('Label：',train_labels)


# data preprocessing
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Model
model=Sequential()
model = models.Sequential()
model.add(layers.Dense(512, activation="relu", input_shape=(28 * 28,)))  # 隐藏层1
model.add(layers.Dense(512, activation="relu"))  # 隐藏层2
model.add(Dropout(0.25)) #隐藏层2随机失活25%
model.add(layers.Dense(10, activation="softmax")) #输出层

model.compile( optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])  
# Optimazer use adam，Loss Function ues categorical_crossentropy





# Trainning
model.fit(train_images, train_labels, epochs=10, batch_size=128)

# Evaluation
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_loss={},test_acc={}'.format(test_loss,test_acc))

# test_loss=0.09380438923835754,test_acc=0.9779000282287598



