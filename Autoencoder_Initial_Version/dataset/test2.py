#C:/Users/tianjiahe189/Anaconda3/envs/wdnmd/python.exe d:/pyWorkSpace/test2.py

from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from sklearn import preprocessing

encoding_dim = 12  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats


# "encoded" is the encoded representation of the input
# "decoded" is the lossy reconstruction of the input
input_layer = Input(shape=(16, ))


encoder = Dense(12, activation="tanh")(input_layer)
encoder = Dense(8, activation="relu")(encoder)

decoder = Dense(12, activation='tanh')(encoder)
decoder = Dense(16, activation='relu')(decoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)

autoencoder.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])

#from keras.datasets import mnist
import numpy as np
#(x_train, _), (x_test, _) = mnist.load_data()
#x_train = x_train.astype('float32') / 255.
#x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
#print(x_train[1])
def standardization(data):
    return [(float(i)-min(data))/float(max(data)-min(data)) for i in data]

def stdfake(data,data2):
    data[7]=data[7]+10
    mu = np.mean(data2, axis=0)
    sigma = np.std(data2, axis=0)
    return (data - mu) / sigma

np.set_printoptions(suppress=True)

c=np.loadtxt("D:\pyWorkSpace/train.txt")
#g=stdfake(c[20],c)
min_max_scaler = preprocessing.MinMaxScaler()
d=min_max_scaler.fit_transform(c)
print(len(c))
print(len(d))

autoencoder.fit(d, d,
                epochs=200,
                batch_size=32,
                shuffle=True,
                validation_split=0.1,
                verbose=1)


# encode and decode some digits
# note that we take them from the *test* set
#encoded_imgs = encoder.predict(x_test)
#decoded_imgs = decoder.predict(encoded_imgs)

e=np.array([d[20000]])
f=np.array([d[10000]])
loss1=autoencoder.evaluate(e,e)
loss2=autoencoder.evaluate(f,f)
print(loss1)
print(loss2)
