#对数据进行预处理
from keras.datasets import imdb
from keras.preprocessing import sequence

max_features = 10000  # （作为特征的单词个数）
maxlen = 500
# 在这么多单词之后截断文本（这些单词都属于前 max_features 个最常见的单词）
batch_size = 32

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('input_train shape:', x_train.shape)
print('input_test shape:', x_test.shape)

from keras.datasets import imdb
from keras.preprocessing import sequence
max_features = 10000 #最大特征数，即该训练集最常见的前10000个单词
maxlen = 50 #每条评论最大长度20，超过将被截断
#加载数据集，获得各评论文本对应的整数列表
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(x_train[:1,])
#整数列表填充处理：即超长截断、不足则前补0，默认从尾部截取，得到等长二维整数张量(samples,maxlen)
x_train =sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print(x_train[:1,])

#建立和训练神经网络
from keras.layers import LSTM
from keras.layers import Embedding,Dense,Dropout
from keras.models import Sequential
lstm = 8
model = Sequential()
model.add(Embedding(max_features, lstm))
model.add(LSTM(32))
model.add(Dropout(0.25))#随机失活25%
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(x_train, y_train,
epochs=10,
batch_size=32,
validation_split=0.2)


# 在测试集上评估模型性能
test_loss, test_acc = model.evaluate(x_test, y_test)
print("LSTM :   ", lstm )
print('test_loss={},test_acc={}'.format(test_loss,test_acc))


"""随着embedding层的增大，会产生过拟合的现象，虽然总体在训练集上的表现好了，但是在
测试集上并没有改善太多，而embedding层少，虽然在训练集上的表现不如embedding大的，但是
在测试集上的表现会更好

# LSTM :  64 
# test_loss=0.5282579660415649,test_acc=0.7883999943733215

# LSTM :    32
# test_loss=0.48259589076042175,test_acc=0.795520007610321

# LSTM :    16
# test_loss=0.44402405619621277,test_acc=0.8091199994087219

# LSTM :    8
# test_loss=0.41271016001701355,test_acc=0.8162800073623657
"""
