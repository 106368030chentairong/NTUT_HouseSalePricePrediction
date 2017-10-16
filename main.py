import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import *
from sklearn.preprocessing import *

def delete(data,data_):
    temp=[]
    for x in range(data.shape[0]):
        if data['bedrooms'].values[x] >=20 :
            temp.append(x)
            print(x)
    data_=np.delete(data_,temp,0)

    return data_

data_1 = pd.read_csv('train-v3.csv')
X_train = data_1.drop(['price','id'],axis=1).values
Y_train = data_1['price'].values


data_2 = pd.read_csv('valid-v3.csv')
X_valid = data_2.drop(['price','id'],axis=1).values
Y_valid = data_2['price'].values

data_3 = pd.read_csv('test-v3.csv')
X_test = data_3.drop(['id'],axis=1).values

X_train=scale(X_train)
X_valid=scale(X_valid)
X_test=scale(X_test)

model = Sequential()
model.add(Dense(32, input_dim=X_train.shape[1],  kernel_initializer='normal',activation='relu'))
model.add(Dense(128, input_dim=32,  kernel_initializer='normal',activation='relu'))
model.add(Dense(128, input_dim=128,  kernel_initializer='normal',activation='relu'))
model.add(Dense(32, input_dim=128,  kernel_initializer='normal',activation='relu'))
model.add(Dense(X_train.shape[1], input_dim=128,  kernel_initializer='normal',activation='relu'))
model.add(Dense(1,  kernel_initializer='normal'))

model.compile(loss='MAE', optimizer='adam')

nb_epoch = 500
batch_size = 32

file_name=str(nb_epoch)+'_'+str(batch_size)
TB=TensorBoard(log_dir='logs/'+file_name, histogram_freq=0)
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,verbose=1,validation_data=(X_valid, Y_valid),callbacks=[TB])
model.save('h5/'+file_name+'.h5')

Y_predict = model.predict(X_test)
np.savetxt('test.csv', Y_predict, delimiter = ',')
