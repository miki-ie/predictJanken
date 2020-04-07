#!/usr/bin/env python
# coding: utf-8

# In[1]:


import urllib.request
import json


# In[2]:


#じゃんけんAPIのURL
url = 'https://janken.miki-ie.com/jankenApi.php'
#API認証のBearerを、個人の認証トークンに変更　@@API_KEY@@
headers = {
    'Authorization': 'Bearer @@API_KEY@@',
    'Content-Type': 'application/json; charset=utf-8'
}


# In[3]:


method = 'POST'


# In[4]:


#めざましじゃんけんの結果を一括取得
data = {
    "mod": "getAllHistory"
}


# In[5]:


json_data = json.dumps(data).encode("utf-8")
print(json_data)


# In[6]:


req = urllib.request.Request(url=url, data=json_data, headers=headers, method=method)
res = urllib.request.urlopen(req, timeout=30)


# In[7]:


print("Http status: {0} {1}".format(res.status, res.reason))
response = res.read().decode("utf-8")
print(response)


# In[8]:


import pandas as pd

dataframe = pd.read_json(response)
print(dataframe)


# In[9]:


dataframe.tv


# In[10]:


#データーセットにグー、チョキ、パーの状態を登録する列を追加
dataframe['Goo'] = 0
dataframe['Choki'] = 0
dataframe['Pa'] = 0


# In[11]:


#tv列のじゃんけん結果に合わせて、グー、チョキ、パー列にじゃんけん結果を登録
dataframe.loc[dataframe.tv == 1, 'Goo'] = 1
dataframe.loc[dataframe.tv == 2, 'Choki'] = 1
dataframe.loc[dataframe.tv == 3, 'Pa'] = 1


# In[12]:


print(dataframe)


# In[13]:


#利用するグー、チョキ、パーの３つの列を切り出す
df = dataframe[["Goo","Choki","Pa"]]


# In[14]:


import numpy as np
from keras.layers import LSTM, Activation, Dense
from keras.models import Sequential


# In[15]:


look_back = 10  # 遡る時間
res_file = 'lstm'


# In[16]:


def shuffle_lists(list1, list2):
    '''リストをまとめてシャッフル'''
    seed = np.random.randint(0, 1000)
    np.random.seed(seed)
    np.random.shuffle(list1)
    np.random.seed(seed)
    np.random.shuffle(list2)


# In[17]:


def get_data(df):
    '''データ作成'''
    dataset = df.values.astype(np.float32)

    X_data, y_data = [], []
    for i in range(len(dataset) - look_back - 1):
        x = dataset[i:(i + look_back)]
        X_data.append(x)
        y_data.append(dataset[i + look_back])

    # X_data = np.array(X_data)
    # y_data = np.array(y_data)
    X_data = np.array(X_data[-500:])
    y_data = np.array(y_data[-500:])
    last_data = np.array([dataset[-look_back:]])

    # シャッフル
    shuffle_lists(X_data, y_data)

    return X_data, y_data, last_data


# In[18]:


def get_model():
    model = Sequential()
    model.add(LSTM(16, input_shape=(look_back, 3)))
    model.add(Dense(3))
    model.add(Activation('softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# In[19]:


def pred(model, X, Y, label):
    '''正解率 出力'''
    predictX = model.predict(X)
    correct = 0
    for real, predict in zip(Y, predictX):
        if real.argmax() == predict.argmax():
            correct += 1
    correct = correct / len(Y)
    print(label + '正解率 : %02.2f ' % correct)


# In[20]:


# データ取得
X_data, y_data, last_data = get_data(df)


# In[21]:


# データ分割
mid = int(len(X_data) * 0.7)
train_X, train_y = X_data[:mid], y_data[:mid]
test_X, test_y = X_data[mid:], y_data[mid:]


# In[22]:


# 学習
model = get_model()
hist = model.fit(train_X, train_y, epochs=100, batch_size=16,
                 validation_data=(test_X, test_y))


# In[23]:


# 正解率出力
pred(model, train_X, train_y, 'train')
pred(model, test_X, test_y, 'test')


# In[24]:


# 次回のテレビのじゃんけんの手を予想
next_hand = model.predict(last_data)
print(next_hand[0])
hands = ['グー', 'チョキ', 'パー']
print('次回の手 : ' + hands[next_hand[0].argmax()])


# In[25]:


#予想した、テレビの出し手から、じゃんけんに勝てる出し手へ変換
if next_hand[0].argmax() == 0:
    next_predict = 3
elif next_hand[0].argmax() == 1:
    next_predict = 1
else:
    next_predict = 2


# In[26]:


#めざましじゃんけん広場に、じゃんけん予想を登録
#登録対象のじゃんけんの、日付と何回戦かを、dateとtimesに指定
data = {
    "mod": "updatePredict",
    "date": "2020-04-08",
    "times": 1,
    "predict": next_predict
}

json_data = json.dumps(data).encode("utf-8")
print(json_data)

req = urllib.request.Request(url=url, data=json_data, headers=headers, method=method)
res = urllib.request.urlopen(req, timeout=30)

print("Http status: {0} {1}".format(res.status, res.reason))
response = res.read().decode("utf-8")
print(response)


# In[ ]:



    

