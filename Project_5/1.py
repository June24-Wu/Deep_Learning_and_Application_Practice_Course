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

# 需要调整的
maxlen = 50 #每条评论最大长度20，超过将被截断
embedding =32



#加载数据集，获得各评论文本对应的整数列表
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(x_train[:1,])
#整数列表填充处理：即超长截断、不足则前补0，默认从尾部截取，得到等长二维整数张量(samples,maxlen)
x_train =sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print(x_train[:1,])


#建立和训练神经网络
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding
model = Sequential()
#添加embedding层，将10000个单词嵌入到维度8的向量中，输入序列长度20
model.add(Embedding(10000, embedding, input_length=maxlen))
#将三维的嵌入张量展平成形状为(samples, maxlen * 8) 的二维张量
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train,
epochs=10,
batch_size=32,
validation_split=0.2)
print(maxlen)

# 在测试集上评估模型性能
test_loss, test_acc = model.evaluate(x_test, y_test)
print("embedding : ",embedding)
print("maxlen  : ",maxlen)
print('test_loss={},test_acc={}'.format(test_loss,test_acc))


"""
发现随着maxlen增大，在测试集上的表现更好，随着embedding的增大，在训练集上的表现更好，但是在测试集
上面的表现就会变差，而减小embedding会发现在测试集上的表现会更好，说明embedding太大会导致网络过拟合

以下是实验结果：


# embedding :  8
# maxlen  :  20
# test_loss=0.5152689218521118,test_acc=0.7581599950790405


# embedding :  8
# maxlen  :  15
# test_loss=0.5299501419067383,test_acc=0.7457200288772583


# embedding :  8
# maxlen  :  50
# test_loss=0.4321190118789673,test_acc=0.8154399991035461


# embedding :  16
# maxlen  :  50
# test_loss=0.474353551864624,test_acc=0.8009200096130371

# embedding :  32
# maxlen  :  50
# test_loss=0.5516825318336487,test_acc=0.7904000282287598

# embedding :  3
# maxlen  :  50
# test_loss=0.40679094195365906,test_acc=0.8216000199317932
"""
