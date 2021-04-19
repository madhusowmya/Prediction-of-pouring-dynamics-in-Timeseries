
import numpy as np

# from google.colab import drive
# drive.mount('/content/drive')

"""1. MODEL LOADING"""

from keras.models import load_model
model = load_model('C:\Users\madhu\Desktop\Deep Learning\Project2\project2submission\model1b.hdf5')#laoding the model from that location

model.summary()

"""RANDOM TEST DATA LOADING

"""

new_data = np.load(r'C:\Users\madhu\Desktop\Deep Learning\Project2\project2submission\Robot_Trials_DL_Test_30pct.npy')#loading thetest  data
new_data.shape

"""MEAN AND STANDARD DEVIATION OF THE TRAINED MODEL"""

std = np.asarray([36.10369924,  0.24980134,  0.11392954, 83.98855534, 16.48147341,
        0.52042226])
mean = np.asarray([-4.35345497e+01, 7.01052766e-01,  2.41029831e-01,  2.23413096e+02,
  8.69148584e+01,  8.27336380e-02])

"""PREPROCESSING TEST DATA, SEPERATING TARGET AND FEATURES"""

def preprocessing1(data_test):
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
  features_test_copy = features_test
  for i in range(len(features_test)):
    for j in range(len(features_test[i])):
      if ((~np.all(features_test_copy[i,j] == 0, axis=0))): # feature scaling without padded zeroes
        features_test_copy[i,j] = ((features_test[i,j]-mean)/std)
  return features_test_copy,target_test

features_test_copy, target_test = preprocessing1(new_data)
target_test.shape

"""PREDICTING TARGET FOR THE GIVEN FEATURES WITH THE MODEL BUILT"""

output_test = model.predict(features_test_copy)
for i in range(len(features_test_copy)):
  for j in range(len(features_test_copy[i])):
    if ((np.all(features_test_copy[i,j] == 0, axis=0))): 
      output_test[i,j] = 0
# print(output_test)
output_test.shape

"""SAVING THE PREDICTED OUTPUT """

np.save('C:\Users\madhu\Desktop\Deep Learning\Project2\project2submission\output.npy')#saving predictions output 

tr_predictions = output_test.reshape((len(output_test)*700),1)
target_test = target_test.reshape((len(target_test)*700),1)

"""CALCULATING RMSE VALUE OF THE PREDICTED OUTPUT"""

from sklearn.metrics import mean_squared_error
import math
Score1 = math.sqrt(mean_squared_error(tr_predictions, target_test))
print('RMSE Score: %.2f RMSE' % (Score1))

