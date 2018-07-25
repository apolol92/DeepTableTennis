import numpy as np
import cv2
import os
import random
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv2D, GlobalAveragePooling1D, MaxPooling2D,Flatten


paths = os.listdir("output")
imgs = []
outputs = []
for i in range(0,len(paths)):
    tmp = "output/"+str(paths[i])
    imgs.append(cv2.imread(tmp,cv2.IMREAD_GRAYSCALE ))
    parts = paths[i].split("_")
    parts[2] = parts[2].split(".")[0]
    outputs.append([int(parts[0]),int(parts[1]),int(parts[2])])
    #print([int(parts[0]),int(parts[1]),int(parts[2])])
x_train = np.array(imgs)[0:int(len(paths)*0.8)]
y_train = np.array(outputs)[0:int(len(paths)*0.8)]
x_test = np.array(imgs)[int(len(paths)*0.8):]
y_test = np.array(outputs)[int(len(paths)*0.8):]

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

m_batch_size = 32
m_epochs = 1
model = Sequential()
# 1.Layer
model.add(Conv2D(32, (3, 3), activation='linear', input_shape=(x_train.shape[1], x_train.shape[2], 1)))
model.add(Conv2D(32, (3, 3), activation='linear'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
'''
# 2.Layer
model.add(Conv2D(64, (3, 3), activation='linear'))
model.add(Conv2D(64, (3, 3), activation='linear'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 3.Layer
model.add(Conv2D(64, (3, 3), activation='linear'))
model.add(Conv2D(64, (3, 3), activation='linear'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
'''
model.add(Flatten())
model.add(Dense(y_train.shape[1], activation='linear'))
model.compile(optimizer='rmsprop',
              loss='mse')
model.fit(x_train, y_train,
          epochs=m_epochs,
          batch_size=m_batch_size)
score = model.evaluate(x_test, y_test, batch_size=m_batch_size)
print("Test-Loss: " + str(score))

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
 
# later...
 
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
