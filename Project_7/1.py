
# 多输出 TEXT

import pandas as pd
import numpy as np


data = pd.read_csv('data.csv')
data = data[['comment_text','toxic','severe_toxic','obscene','threat','insult','identity_hate']]


# 数据预处理
import re

def preprocessing(row_data):
    # 过滤不了\\ \ 中文（）还有————
    r1 = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'#用户也可以在此进行自定义过滤字符
    # 者中规则也过滤不完全
    r2 = "[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+"
    # \\\可以过滤掉反向单杠和双杠，/可以过滤掉正向单杠和双杠，第一个中括号里放的是英文符号，第二个中括号里放的是中文符号，第二个中括号前不能少|，否则过滤不完全
    r3 =  "[.!//_,$&%^*()<>+\"'?@#-|:~{}]+|[——！\\\\，。=？、：“”‘’《》【】￥……（）]+"
    # 去掉括号和括号内的所有内容
    r4 =  "\\【.*?】+|\\《.*?》+|\\#.*?#+|[.!/_,$&%^*()<>+""'?@|:~{}#]+|[——！\\\，。=？、：“”‘’￥……（）《》【】]"

    sentence = row_data
    cleanr = re.compile('<.*?>')
    sentence = re.sub(cleanr, ' ', sentence)        #去除html标签
    sentence = re.sub(r4,'',sentence)
    return sentence

import jieba

def cut_word(row_data):
    sentence = row_data
    sentence_seg = jieba.cut(sentence)
    result = ' '.join(sentence_seg)
    return result
from sklearn.feature_extraction.text import TfidfVectorizer
vector = TfidfVectorizer()


input_text = []
for i in range(len(data)):
    row_data = data.iloc[i:i+1,0].values[0]
    row_data = preprocessing(row_data)
    row_data = cut_word(row_data)
    input_text.append(row_data)
    print(i)
tf_data = vector.fit_transform(input_text)


# model
from keras import layers , Input
from keras.models import Model
vocabulary_size = len(data)
num_income_groups = 10
posts_input = Input(shape = (None,),name='posts')
embedded_posts = layers.Embedding(256,vocabulary_size)(posts_input)
x = layers.Conv1D(128,5,activation = 'relu')(embedded_posts)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256,5,activation = 'relu')(x)
x = layers.Conv1D(256,5,activation = 'relu')(x)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256,5,activation = 'relu')(x)
x = layers.Conv1D(256,5,activation = 'relu')(x)
x = layers.Dense(128,activation = 'relu')(x)

toxic_prediction = layers.Dense(1,activation = 'sigmoid',name = 'toxic')(x)
severe_toxic_prediction = layers.Dense(1,activation = 'sigmoid',name = 'severe_toxic')(x)
obscene_prediction = layers.Dense(1,activation = 'sigmoid',name = 'obscene')(x)
threat_prediction = layers.Dense(1,activation = 'sigmoid',name = 'threat')(x)
insult_prediction = layers.Dense(1,activation = 'sigmoid',name = 'insult')(x)
identity_hate = layers.Dense(1,activation = 'sigmoid',name = 'identity_hate')(x)

model = Model(posts_input,[toxic_prediction,
                           severe_toxic_prediction,
                           obscene_prediction,
                           threat_prediction,
                           insult_prediction,
                           identity_hate
                          ])

model.compile(optimizer='rmsprop',loss=['binary_crossentropy','binary_crossentropy',
                                       'binary_crossentropy','binary_crossentropy',
                                       'binary_crossentropy','binary_crossentropy']),
model.summary()

new = tf_data.A
model.fit(new,[data['toxic'],data['severe_toxic'],
                  data['obscene'],data['threat'],
                  data['insult'],data['identity_hate']],
         batch_size=200,epochs=10)

