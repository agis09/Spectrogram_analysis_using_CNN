import numpy as np
import keras
import sys
from sklearn.model_selection import train_test_split
from keras.models import Sequential,Model
from keras.layers.convolutional import Conv2D,MaxPooling2D,AveragePooling2D
from keras.layers.pooling import MaxPool2D,GlobalAveragePooling2D
from keras.layers.recurrent import GRU
from keras.optimizers import Adam,SGD
from keras.layers import Dense, Activation, Dropout, Flatten,BatchNormalization,Input,merge
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score
from keras.initializers import TruncatedNormal, Constant
from Resnet import ResnetBuilder


name = ['ACC_X', 'ACC_Y', 'ACC_Z', 'GYRO_X', 'GYRO_Y', 'GYRO_Z', 'EOG_L', 'EOG_R', 'EOG_H', 'EOG_V']

file_name = ['180117_G1_JINSMEME', '180117_H1_JINSMEME', '180112_E1_JINSMEME', '171227_C1_JINS_MEME']

clm_num = 256
"""
X_0 = np.array([])
for path in file_name:
    for i, clm in enumerate(name):
        X = np.array([])
        x = np.load('specgram/' + path + '/' + clm + '_0.npy')
        # 縦1行削る
        x1, x = np.split(x, [1], axis=0)
        while len(x[0]) >= clm_num:
            x1, x = np.split(x, [clm_num], axis=1)
            X = np.append(X, x1)

        X_tmp = np.reshape(X, (len(X), 1))
        if i != 0:
            X_0 = np.concatenate((X_0, X_tmp), axis=1)
        else:
            X_0 = X_tmp

X_0 = np.reshape(X_0, (len(X_0) // clm_num // clm_num, clm_num, clm_num, 10))
print(X_0.shape)

X_1 = np.array([])
for path in file_name:
    for i, clm in enumerate(name):
        X = np.array([])
        x = np.load('specgram/' + path + '/' + clm + '_1.npy')
        # 257->256
        x1, x = np.split(x, [1], axis=0)
        while len(x[0]) >= clm_num:
            x1, x = np.split(x, [clm_num], axis=1)
            X = np.append(X, x1)

        X_tmp = np.reshape(X, (len(X), 1))
        if i != 0:
            X_1 = np.concatenate((X_1, X_tmp), axis=1)
        else:
            X_1 = X_tmp

X_1 = np.reshape(X_1, (len(X_1) // clm_num // clm_num, clm_num, clm_num, 10))
print(X_1.shape)


print("######################")
print(len(X_0),len(X_1))

if len(X_0)>len(X_1):
    (X_0,x)=train_test_split(X_0,train_size=len(X_1)/len(X_0))
else:
    (X_1,x)=train_test_split(X_1,train_size=len(X_0)/len(X_1))


print("######################")
print(len(X_0),len(X_1))

X = np.concatenate((X_0, X_1), axis=0)
Y_0 = np.zeros((len(X_0),1))
Y_1 = np.ones((len(X_1),1))
Y = np.concatenate((Y_0,Y_1), axis=0)


np.save("X_"+str(clm_num)+"_"+str(clm_num)+".npy",X)
np.save("Y_"+str(clm_num)+"_"+str(clm_num)+".npy",Y)
"""
X = np.load("X_"+str(clm_num)+"_"+str(clm_num)+".npy")
Y = np.load("Y_"+str(clm_num)+"_"+str(clm_num)+".npy")

(X_train,X_test,Y_train,Y_test) = train_test_split(X,Y,test_size=0.2)


def resnet():
    model = ResnetBuilder.build_resnet_101((10,256,256),1)
    model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model


class RocAucEvaluation(Callback):
        def __init__(self, validation_data=(), interval=1):
            super(Callback, self).__init__()

            self.interval = interval
            self.X_val, self.y_val = validation_data

        def on_epoch_end(self, epoch, logs={}):
            if epoch % self.interval == 0:
                y_pred = self.model.predict(self.X_val, verbose=0)
                score = roc_auc_score(self.y_val, y_pred)
                print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch + 1, score))


RocAuc = RocAucEvaluation(validation_data=(X_test, Y_test), interval=1)

model = resnet()

model.fit(X_train,Y_train,
          validation_data=(X_test,Y_test),
          batch_size=10,
          epochs=10,
          callbacks=[RocAuc],
          verbose=1)

prediction = model.predict(X_test)

for i,j in zip(Y_test,prediction):
    print(i,j)
