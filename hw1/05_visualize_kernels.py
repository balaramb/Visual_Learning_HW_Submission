from tensorflow.python import pywrap_tensorflow
import numpy as np
import scipy.misc
import tensorflow as tf
#import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pylab
#from pylab import *


reader = tf.train.NewCheckpointReader("model.ckpt-40000")
#var_to_shape_map = reader.get_variable_to_shape_map()
#for key in var_to_shape_map:
#   print("tensor_name: ", key)
#   print(reader.get_tensor(key))


scaler = MinMaxScaler()
weights = reader.get_tensor('conv2d/kernel/Momentum')											#11x11x3x96
weights_transposed = np.transpose (weights, (3, 0, 1, 2))						#96x11x11x3
fig, axs = plt.subplots(10, 10,figsize=(15,15))

for filter_num in range(weights_transposed.shape[0]):
    extracted_filter = weights_transposed[filter_num, :, :, :]
    for i in range(3):
        extracted_filter[:,:,i] = scaler.fit_transform(extracted_filter[:,:,i])
    
    axs[filter_num//10, filter_num%10].imshow(extracted_filter, interpolation='nearest')

for i in range(10):
    for j in range(10):
        axs[i,j].tick_params(labelbottom='off',  labeltop='off', labelleft='off', labelright='off') 

pylab.savefig('alexnet_40000ckpt.png')




reader = tf.train.NewCheckpointReader("model.ckpt-39601")
#var_to_shape_map = reader.get_variable_to_shape_map()
#for key in var_to_shape_map:
#   print("tensor_name: ", key)
#   print(reader.get_tensor(key))


scaler = MinMaxScaler()
weights = reader.get_tensor('conv2d/kernel/Momentum')											#11x11x3x96
weights_transposed = np.transpose (weights, (3, 0, 1, 2))						#96x11x11x3
fig, axs = plt.subplots(10, 10,figsize=(15,15))

for filter_num in range(weights_transposed.shape[0]):
    extracted_filter = weights_transposed[filter_num, :, :, :]
    for i in range(3):
        extracted_filter[:,:,i] = scaler.fit_transform(extracted_filter[:,:,i])
    
    axs[filter_num//10, filter_num%10].imshow(extracted_filter, interpolation='nearest')

for i in range(10):
    for j in range(10):
        axs[i,j].tick_params(labelbottom='off',  labeltop='off', labelleft='off', labelright='off') 

pylab.savefig('alexnet_39601ckpt.png')



reader = tf.train.NewCheckpointReader("model.ckpt-39201")
#var_to_shape_map = reader.get_variable_to_shape_map()
#for key in var_to_shape_map:
#   print("tensor_name: ", key)
#   print(reader.get_tensor(key))


scaler = MinMaxScaler()
weights = reader.get_tensor('conv2d/kernel/Momentum')											#11x11x3x96
weights_transposed = np.transpose (weights, (3, 0, 1, 2))						#96x11x11x3
fig, axs = plt.subplots(10, 10,figsize=(15,15))

for filter_num in range(weights_transposed.shape[0]):
    extracted_filter = weights_transposed[filter_num, :, :, :]
    for i in range(3):
        extracted_filter[:,:,i] = scaler.fit_transform(extracted_filter[:,:,i])
    
    axs[filter_num//10, filter_num%10].imshow(extracted_filter, interpolation='nearest')

for i in range(10):
    for j in range(10):
        axs[i,j].tick_params(labelbottom='off',  labeltop='off', labelleft='off', labelright='off') 

pylab.savefig('alexnet_39201ckpt.png')