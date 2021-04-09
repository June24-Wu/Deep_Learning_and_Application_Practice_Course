import os
from matplotlib import pyplot as plt
import numpy as np
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

# read_date
data= './jena_climate_2009_2016.csv'

f = open(data) #打开文件
data = f.read() #读取文件
f.close() #关闭

lines = data.split('\n') #按行切分
header = lines[0].split(',') #每行按，切分
lines = lines[1:] #去除第0行，第0行为标记

print(header)
print(len(lines))

#将所有的数据转为float型数组
float_data = np.zeros((len(lines), len(header)-1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values

'''
temp = float_data[:, 1]
plt.figure()
plt.plot(range(len(temp)), temp)
plt.legend()

plt.figure()
plt.plot(range(1440), temp[:1440])
plt.show()
'''
#数据标准化，减去平均值，除以标准值
mean = float_data[:200000].mean(axis = 0) 
float_data -= mean
std = float_data[:200000].std(axis = 0)
float_data /= std
#数据生成器
def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index = lookback
    while 1:
        if shuffle: #打乱顺序
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback #超过记录序号时，从头开始
            rows = np.arange(i, min(i+batch_size, max_index))
            i += len(rows)
        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows), ))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets

lookback = 1440 #给定10天的观测数据
step = 6 #每6个采样一次，即每小时一个数据点
delay = 144 #目标是未来24小时之后的数据
batch_size = 128

train_gen = generator(float_data, lookback=lookback, delay=delay,
                      min_index=0,max_index=200000, shuffle=True,
                      step=step, batch_size = batch_size)
val_gen = generator(float_data, lookback=lookback, delay=delay,
                    min_index=200001, max_index=300000, step=step,
                    batch_size=batch_size)
test_gen = generator(float_data, lookback=lookback, delay=delay,
                     min_index=300001, max_index=None, step=step,
                     batch_size=batch_size)
val_steps = (300000 - 200001- lookback) // batch_size
test_steps = (len(float_data) - 300001 - lookback) // batch_size
#计算mae
def evaluate_naive_method():
    batch_maes = []
    for step in range(val_steps):
        samples, targets = next(val_gen)
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    print('mae=',np.mean(batch_maes))

#evaluate_naive_method()


#一维卷积基与GRU融合
model  = Sequential()
model.add(layers.Conv1D(32, 5, activation='relu', input_shape=(None, float_data.shape[-1])))
model.add(layers.MaxPooling1D(3))
model.add(layers.Conv1D(32, 5, activation='relu'))
# 使用LSTM 替代GRU
model.add(layers.LSTM(32, dropout = 0.1, recurrent_dropout = 0.5))
# model.add(layers.GRU(32, dropout = 0.1, recurrent_dropout = 0.5))

# 尝试添加更多的全连接层
model.add(layers.Dense(32))
model.add(layers.Dense(16))
model.add(layers.Dense(1))

# 尝试多个学习率
# model.compile(optimizer=RMSprop(lr=0.1), loss='mae')
# model.compile(optimizer=RMSprop(lr=0.01), loss='mae')
model.compile(optimizer=RMSprop(lr=0.001), loss='mae')
history = model.fit_generator(train_gen, steps_per_epoch=500, epochs=20,
                              validation_data=val_gen, validation_steps=val_steps)

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(loss) + 1)

plt.figure()
plt.plot(epochs, loss, 'bo', label = 'training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('Training and Validation loss')
plt.legend()

plt.show()


# final loss : 0.21
