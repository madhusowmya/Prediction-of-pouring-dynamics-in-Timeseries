# -*- coding: utf-8 -*-
"""Main.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1PqZjgSR8TlY-gWOMuDdYQqbKjBYDayBV
"""

import numpy as np

# from google.colab import drive
# drive.mount('/content/drive')

data = np.load(r'C:\Users\madhu\Desktop\Deep Learning\Project2\project2submission\Robot_Trials_DL.npy')#data load from local drive
data.shape
print(data[0])

len(data[1])

from sklearn import model_selection
# data_train, data_test = model_selection.train_test_split(data,test_size = 0.35)

data_train, data_test = model_selection.train_test_split(data, test_size=0.2)
data_train, data_val =  model_selection.train_test_split(data_train, test_size=0.2)
print(len(data_train), 'train examples')
print(len(data_val), 'validation examples')
print(len(data_test), 'test examples')

print(data_train.shape, data_test.shape, data_val.shape)

target_train=[]
features_train=[]

for i in data_train:
  for j in i:
    target_train.append(j[1])
    k = []
    k=j.tolist()
    k.pop(1)
    # k.insert(0, k.pop()) 
    # k = k[:-4]
    features_train.append(k)
print(len(features_train))
target_train = np.asarray(target_train)#weight f(t)
features_train = np.asarray(features_train)# 6 features
features_train = features_train.reshape(len(data_train),700,6)
target_train = target_train.reshape(len(data_train),700,1)
print(features_train.shape,target_train.shape)

target_val=[]
features_val=[]

for i in data_val:
  for j in i:
    target_val.append(j[1])
    k = []
    k=j.tolist()
    k.pop(1)
    # k.insert(0, k.pop()) 
    # k = k[:-4]
    features_val.append(k)
print(len(features_val))
target_val = np.asarray(target_val)#weight f(t)
features_val = np.asarray(features_val)# 6 features
features_val = features_val.reshape(len(data_val),700,6)
target_val = target_val.reshape(len(data_val),700,1)
print(features_val.shape,target_val.shape)

target_test=[]
features_test=[]

for i in data_test:
  for j in i:
    target_test.append(j[1])
    k = []
    k=j.tolist()
    k.pop(1)
    # k.insert(0, k.pop()) 
    # k = k[:-4]
    features_test.append(k)
print(len(features_test))
target_test = np.asarray(target_test)
features_test = np.asarray(features_test)
features_test = features_test.reshape(len(data_test),700,6)
target_test = target_test.reshape(len(data_test),700,1)
print(features_test.shape,target_test.shape)

target_train = np.asarray(target_train)#weight f(t)
features_train = np.asarray(features_train)# 6 features





features_train_np=features_train.reshape(-1, features_train.shape[-1])
print('shape',features_train.shape)
features_train_np = features_train_np[~np.all(features_train_np == 0., axis=1)]
print('Xshape',features_train_np.shape)
std=np.std(features_train_np,axis=0)
print(std)
print('\n')
mean=np.mean(features_train_np,axis=0)
print(mean)

#features_train = features_train.astype
features_train_copy = features_train
for i in range(len(features_train)):
  for j in range(len(features_train[i])):
    if (~np.all(features_train_copy[i,j] == 0, axis=0)): 
      features_train_copy[i,j] = ((features_train[i,j]-mean)/std)

print(features_train_copy)

#features_train = features_train.astype
features_val_copy = features_val
for i in range(len(features_val)):
  for j in range(len(features_val[i])):
    if (~np.all(features_val_copy[i,j] == 0, axis=0)): 
      features_val_copy[i,j] = ((features_val[i,j]-mean)/std)

print(features_val_copy)

# #features_train = features_train.astype
# features_train_copy = features_train
# for i in range(len(features_train)):
#   for j in range(len(features_train[i])):
#     if (~np.all(features_train_copy[i,j] == 0, axis=0)): 
#       features_train_copy[i,j] = ((features_train[i,j]-min(features_train[i,j]))/(max(features_train[i,j])-min(features_train[i,j])))



# # features_train = np.insert(features_train_scal, res, 0, axis = 0)
# len(res)
# for i in range(0,len(features_train)):
#   for j in range(0,len(res)):
#     if features_train[i] == res[i]:
#       print(i)

features_test_np=features_test.reshape(-1, features_test.shape[-1])
print('shape',features_test.shape)
features_test_np = features_test_np[~np.all(features_test_np == 0., axis=1)]
print('Xshape',features_test_np.shape)

# std_test=np.std(features_test_np,axis=0)
# mean_test=np.mean(features_test_np,axis=0)

# for i in features_train:
#   for j in range(len(i)):
#     if i[j]==0:
#       if i[j]==0:
#         i[j]=mn2[j]


# # testing

# std_test

# mean

# #features_train = features_train.astype
# features_test_copy = features_test
# for i in range(len(features_test)):
#   for j in range(len(features_test[i])):
#     if ((~np.all(features_test_copy[i,j] == 0, axis=0) and (~np.all(features_train_copy[i,j] == 0, axis=0)))): 
#       features_test_copy[i,j] = ((features_test[i,j]-min(features_train[i,j]))/(max(features_train[i,j])-min(features_train[i,j])))

#features_train = features_train.astype
features_test_copy = features_test
for i in range(len(features_test)):
  for j in range(len(features_test[i])):
    if ((~np.all(features_test_copy[i,j] == 0, axis=0))): 
      features_test_copy[i,j] = ((features_test[i,j]-mean)/std)

features_test_copy



import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.compat.v1.keras.layers import Dense,CuDNNLSTM, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import math 

model=Sequential()
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Masking(mask_value=0.,input_shape=(700,6)))
model.add(LSTM(16, return_sequences=True,dropout=0.15,recurrent_dropout=0.0,input_shape=(700,6)))
model.add(LSTM(16, return_sequences=True,dropout=0.15,recurrent_dropout=0.0))
model.add(LSTM(16, return_sequences=True,dropout=0.15,recurrent_dropout=0.0))
model.add(LSTM(16, return_sequences=True,dropout=0.15,recurrent_dropout=0.0))
# model.add(LSTM(64, return_sequences=True,dropout=0.20,recurrent_dropout=0.0))
model.add(Dense(1)) 
model.compile(loss='mean_squared_error',optimizer='adam')



model.summary()
# model.fit(features_train, madhu, epochs=10, batch_size=1, verbose=1)



history = model.fit(features_train_copy, target_train,validation_data=(features_val_copy,target_val),epochs=50, batch_size=32, verbose=1)

model.save('C:\Users\madhu\Desktop\Deep Learning\Project2\project2submission\model.hdf5')# saving model at a location in C drive


from keras.models import load_model
model = load_model('C:\Users\madhu\Desktop\Deep Learning\Project2\project2submission\model.hdf5')#laoding the model from that location


features_test_copy

output_test = model.predict(features_test_copy)
for i in range(len(features_test_copy)):
  for j in range(len(features_test_copy[i])):
    if ((np.all(features_test_copy[i,j] == 0, axis=0))): 
      output_test[i,j] = 0



print(output_test)

output_test = output_test.reshape(len(output_test)*700,1)
target_test = target_test.reshape(len(target_test)*700,1)

Score = math.sqrt(mean_squared_error(output_test, target_test))
print('Test RMSE Score: %.2f RMSE' % (Score))

import matplotlib.pyplot as plt
plt.xlabel("data")
plt.ylabel("output")
plt.title("prediction vs true data")
plt.plot(target_test)
plt.plot(output_test)
plt.show()



loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(0,100)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

tr_predictions = model.predict(features_train_copy)
tr_predictions.shape

tr_predictions = tr_predictions.reshape((len(data_train)*700),1)
target_train = target_train.reshape((len(data_train)*700),1)

from sklearn.metrics import mean_squared_error
Score1 = math.sqrt(mean_squared_error(tr_predictions, target_train))
print('Train Score: %.2f RMSE' % (Score1))

std

print(mean)

data_test

# np.save('/content/drive/My Drive/Project2/finaltest.npy',data[100:200])

# np.load('/content/drive/My Drive/Project2/finaltest.npy')

