#from keras.datasets import mnist
import numpy as np
#(x_train, _), (x_test, _) = mnist.load_data()
#x_train = x_train.astype('float32') / 255.
#x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
#print(x_train[1])
from sklearn import preprocessing

# def standardization(data):
#     mu = np.mean(data, axis=0)
#     sigma = np.std(data, axis=0)
#     return (data - mu) / sigma

np.set_printoptions(suppress=True)
close=np.loadtxt("D:\pyWorkSpace/train.txt")
close=close.astype('float32')
b=close
# print(close[0]) 
# b=np.array([])

# for i in range(90000):
#     a=close[i]
#     a=np.append(close[i],close[i+1])
#     a=np.append(a,close[i+2])
#     a=np.append(a,close[i+3])
#     a=np.append(a,close[i+4])
#     b=np.append(b,a)

b=b.reshape((20000, 80))
print(len(b))
np.savetxt("D:\pyWorkSpace/train33.txt", b)
#c=np.loadtxt("b.txt")
print(b)
