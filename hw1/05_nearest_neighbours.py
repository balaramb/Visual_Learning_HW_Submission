from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python import pywrap_tensorflow

reader = pywrap_tensorflow.NewCheckpointReader("../../../vgg_16.ckpt")

# Imports
import sys
import numpy as np
import tensorflow as tf
import argparse
import os.path as osp
from PIL import Image
from functools import partial
from sklearn.neighbors import NearestNeighbors
from eval import compute_map
#import models

tf.logging.set_verbosity(tf.logging.INFO)

CLASS_NAMES = [
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor',
]

def preprocess_image(image):
    temp1 = tf.image.random_flip_left_right(tf.convert_to_tensor(image))
    temp2 = tf.random_crop (temp1,[227,227,3])
    return temp2

def cnn_model_fn_alexnet_pool5(features, labels, mode, num_classes):
    input_layer = tf.reshape(features["x"], [-1, 256, 256, 3])
    image_num = features["n"]

    if mode == tf.estimator.ModeKeys.TRAIN:
       img_list = tf.unstack(input_layer, num=input_layer.get_shape()[0], axis=0)
       input_layer = tf.stack([preprocess_image(i) for i in img_list],axis=0)
    else:
       input_layer = tf.image.crop_to_bounding_box(input_layer,15,15,227,227) 

    input_layer = tf.cast(input_layer, tf.float32)

    #init_gen = tf.random_normal_initializer(stddev = 0.01)
    init_gen = tf.truncated_normal_initializer(stddev = 0.01)

    # Convolutional Layer #1 and Pooling Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=96,
        kernel_size=[11, 11],
        strides = [4, 4],
        padding="valid",
        activation=tf.nn.relu,
        kernel_initializer=init_gen)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=256,
        kernel_size=[5, 5],
        strides = [1, 1],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=init_gen)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], strides=2)

    # Convolutional Layer #3
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=384,
        kernel_size=[3, 3],
        strides = [1, 1],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=init_gen)

    # Convolutional Layer #4
    conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters=384,
        kernel_size=[3, 3],
        strides = [1, 1],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=init_gen)

    # Convolutional Layer #5 and Pooling Layer
    conv5 = tf.layers.conv2d(
        inputs=conv4,
        filters=256,
        kernel_size=[3, 3],
        strides = [1, 1],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=init_gen)
    pool3 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[3, 3], strides=2)

    # Dense Layer
    #CHANGED THIS DIMENSION
    pool3_flat = tf.reshape(pool3, [-1, 6 * 6 * 256])

    #predictions = {"activation_maps":tf.concat(pool3_flat,tf.to_float(image_num))}
    predictions = {"activation_maps":pool3_flat}
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    dense1 = tf.layers.dense(inputs=pool3_flat, units=4096,
                            activation=tf.nn.relu, kernel_initializer=init_gen)
    dropout1 = tf.layers.dropout(
        inputs=dense1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    dense2 = tf.layers.dense(inputs=dropout1, units=4096,
                            activation=tf.nn.relu, kernel_initializer=init_gen)
    dropout2 = tf.layers.dropout(
        inputs=dense2, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout2, units=20, kernel_initializer=init_gen)

    loss = tf.identity(tf.losses.sigmoid_cross_entropy(
        multi_class_labels=labels, logits=logits), name='loss')

    global_step=tf.train.get_global_step()
    learning_rate = tf.train.exponential_decay(learning_rate=0.001, global_step=global_step , decay_steps=10000, decay_rate=0.5, staircase=True)


    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=global_step)
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["probabilities"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)





def cnn_model_fn_alexnet_fc7(features, labels, mode, num_classes):
    input_layer = tf.reshape(features["x"], [-1, 256, 256, 3])
    image_num = features["n"]

    if mode == tf.estimator.ModeKeys.TRAIN:
       img_list = tf.unstack(input_layer, num=input_layer.get_shape()[0], axis=0)
       input_layer = tf.stack([preprocess_image(i) for i in img_list],axis=0)
    else:
       input_layer = tf.image.crop_to_bounding_box(input_layer,15,15,227,227) 

    input_layer = tf.cast(input_layer, tf.float32)

    #init_gen = tf.random_normal_initializer(stddev = 0.01)
    init_gen = tf.truncated_normal_initializer(stddev = 0.01)

    # Convolutional Layer #1 and Pooling Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=96,
        kernel_size=[11, 11],
        strides = [4, 4],
        padding="valid",
        activation=tf.nn.relu,
        kernel_initializer=init_gen)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=256,
        kernel_size=[5, 5],
        strides = [1, 1],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=init_gen)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], strides=2)

    # Convolutional Layer #3
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=384,
        kernel_size=[3, 3],
        strides = [1, 1],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=init_gen)

    # Convolutional Layer #4
    conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters=384,
        kernel_size=[3, 3],
        strides = [1, 1],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=init_gen)

    # Convolutional Layer #5 and Pooling Layer
    conv5 = tf.layers.conv2d(
        inputs=conv4,
        filters=256,
        kernel_size=[3, 3],
        strides = [1, 1],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=init_gen)
    pool3 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[3, 3], strides=2)

    # Dense Layer
    #CHANGED THIS DIMENSION
    pool3_flat = tf.reshape(pool3, [-1, 6 * 6 * 256])

    dense1 = tf.layers.dense(inputs=pool3_flat, units=4096,
                            activation=tf.nn.relu, kernel_initializer=init_gen)

    dropout1 = tf.layers.dropout(
        inputs=dense1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)


    dense2 = tf.layers.dense(inputs=dropout1, units=4096,
                            activation=tf.nn.relu, kernel_initializer=init_gen)

    predictions = {"activation_maps":dense2}
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    dropout2 = tf.layers.dropout(
        inputs=dense2, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout2, units=20, kernel_initializer=init_gen)

    loss = tf.identity(tf.losses.sigmoid_cross_entropy(
        multi_class_labels=labels, logits=logits), name='loss')

    global_step=tf.train.get_global_step()
    learning_rate = tf.train.exponential_decay(learning_rate=0.001, global_step=global_step , decay_steps=10000, decay_rate=0.5, staircase=True)


    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=global_step)
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["probabilities"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)




def cnn_model_fn_vgg_pretrained_pool5(features, labels, mode, num_classes=20):
    input_layer = tf.reshape(features["x"], [-1, 256, 256, 3])
    #tf.summary.image("image", input_layer)

    if mode == tf.estimator.ModeKeys.TRAIN:
       img_list = tf.unstack(input_layer, num=input_layer.get_shape()[0],axis=0)
       input_layer = tf.stack([preprocess_image(i) for i in img_list],axis=0)
    else:
       input_layer = tf.image.crop_to_bounding_box(input_layer, 16, 16, 224, 224)

    input_layer = tf.cast(input_layer, tf.float32)

    init_gen = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=64,
        kernel_size=[3, 3],
        strides = [1, 1],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.constant_initializer(reader.get_tensor('vgg_16/conv1/conv1_1/weights')),
        bias_initializer=tf.constant_initializer(reader.get_tensor('vgg_16/conv1/conv1_1/biases'))
        )

    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=64,
        kernel_size=[3, 3],
        strides = [1, 1],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.constant_initializer(reader.get_tensor('vgg_16/conv1/conv1_2/weights')),
        bias_initializer=tf.constant_initializer(reader.get_tensor('vgg_16/conv1/conv1_2/biases'))
        )

    pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    conv3 = tf.layers.conv2d(
        inputs=pool1,
        filters=128,
        kernel_size=[3, 3],
        strides = [1, 1],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.constant_initializer(reader.get_tensor('vgg_16/conv2/conv2_1/weights')),
        bias_initializer=tf.constant_initializer(reader.get_tensor('vgg_16/conv2/conv2_1/biases'))
        )

    conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters=128,
        kernel_size=[3, 3],
        strides = [1, 1],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.constant_initializer(reader.get_tensor('vgg_16/conv2/conv2_2/weights')),
        bias_initializer=tf.constant_initializer(reader.get_tensor('vgg_16/conv2/conv2_2/biases'))        
        )
    
    pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)	

    conv5 = tf.layers.conv2d(
        inputs=pool2,
        filters=256,
        kernel_size=[3, 3],
        strides = [1, 1],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.constant_initializer(reader.get_tensor('vgg_16/conv3/conv3_1/weights')),
        bias_initializer=tf.constant_initializer(reader.get_tensor('vgg_16/conv3/conv3_1/biases'))        
        )

    conv6 = tf.layers.conv2d(
        inputs=conv5,
        filters=256,
        kernel_size=[3, 3],
        strides = [1, 1],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.constant_initializer(reader.get_tensor('vgg_16/conv3/conv3_2/weights')),
        bias_initializer=tf.constant_initializer(reader.get_tensor('vgg_16/conv3/conv3_2/biases'))        
        )

    conv7 = tf.layers.conv2d(
        inputs=conv6,
        filters=256,
        kernel_size=[3, 3],
        strides = [1, 1],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.constant_initializer(reader.get_tensor('vgg_16/conv3/conv3_3/weights')),
        bias_initializer=tf.constant_initializer(reader.get_tensor('vgg_16/conv3/conv3_3/biases'))        
        )
    
    pool3 = tf.layers.max_pooling2d(inputs=conv7, pool_size=[2, 2], strides=2)

    conv8 = tf.layers.conv2d(
        inputs=pool3,
        filters=512,
        kernel_size=[3, 3],
        strides = [1, 1],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.constant_initializer(reader.get_tensor('vgg_16/conv4/conv4_1/weights')),
        bias_initializer=tf.constant_initializer(reader.get_tensor('vgg_16/conv4/conv4_1/biases'))
        )

    conv9 = tf.layers.conv2d(
        inputs=conv8,
        filters=512,
        kernel_size=[3, 3],
        strides = [1, 1],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.constant_initializer(reader.get_tensor('vgg_16/conv4/conv4_2/weights')),
        bias_initializer=tf.constant_initializer(reader.get_tensor('vgg_16/conv4/conv4_2/biases'))
        )

    conv10 = tf.layers.conv2d(
        inputs=conv9,
        filters=512,
        kernel_size=[3, 3],
        strides = [1, 1],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.constant_initializer(reader.get_tensor('vgg_16/conv4/conv4_3/weights')),
        bias_initializer=tf.constant_initializer(reader.get_tensor('vgg_16/conv4/conv4_3/biases'))
        )
    
    pool4 = tf.layers.max_pooling2d(inputs=conv10, pool_size=[2, 2], strides=2)

    conv11 = tf.layers.conv2d(
        inputs=pool4,
        filters=512,
        kernel_size=[3, 3],
        strides = [1, 1],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.constant_initializer(reader.get_tensor('vgg_16/conv5/conv5_1/weights')),
        bias_initializer=tf.constant_initializer(reader.get_tensor('vgg_16/conv5/conv5_1/biases'))
        )

    conv12 = tf.layers.conv2d(
        inputs=conv11,
        filters=512,
        kernel_size=[3, 3],
        strides = [1, 1],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.constant_initializer(reader.get_tensor('vgg_16/conv5/conv5_2/weights')),
        bias_initializer=tf.constant_initializer(reader.get_tensor('vgg_16/conv5/conv5_2/biases'))        
        )

    conv13 = tf.layers.conv2d(
        inputs=conv12,
        filters=512,
        kernel_size=[3, 3],
        strides = [1, 1],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.constant_initializer(reader.get_tensor('vgg_16/conv5/conv5_3/weights')),
        bias_initializer=tf.constant_initializer(reader.get_tensor('vgg_16/conv5/conv5_3/biases'))
        )
    
    pool5 = tf.layers.max_pooling2d(inputs=conv13, pool_size=[2, 2], strides=2)

    #print(pool5.get_shape())

    pool5_flat = tf.reshape(pool5, [-1, 7 * 7 * 512])

    predictions = {"activation_maps":pool5_flat}
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    
    dense1 = tf.layers.dense(inputs=pool5_flat, units=4096,
                            activation=tf.nn.relu, 
                            kernel_initializer=tf.constant_initializer(reader.get_tensor('vgg_16/fc6/weights')),
                            bias_initializer=tf.constant_initializer(reader.get_tensor('vgg_16/fc6/biases'))
                            )

    dropout1 = tf.layers.dropout(
        inputs=dense1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    dense2 = tf.layers.dense(inputs=dropout1, units=4096,
                            activation=tf.nn.relu, 
                            kernel_initializer=tf.constant_initializer(reader.get_tensor('vgg_16/fc7/weights')),
                            bias_initializer=tf.constant_initializer(reader.get_tensor('vgg_16/fc7/biases'))
                            )

    dropout2 = tf.layers.dropout(
        inputs=dense2, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    dense3 = tf.layers.dense(inputs=dropout2, units=1000,
                            activation=tf.nn.relu, kernel_initializer=init_gen)

    logits = tf.layers.dense(inputs=dense3, units=20, kernel_initializer=init_gen)

    predictions = {
        "probabilities": tf.nn.sigmoid(logits, name="sigmoid_tensor")
    }

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.identity(tf.losses.sigmoid_cross_entropy(
        multi_class_labels=labels, logits=logits), name='loss')

    global_step=tf.train.get_global_step()
    learning_rate = tf.train.exponential_decay(learning_rate=0.0001, global_step=global_step , decay_steps=1000, decay_rate=0.5, staircase=True)


    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
        learning_rate = optimizer._learning_rate

        grads_and_vars=optimizer.compute_gradients(loss)

        train_op = optimizer.minimize(
            loss=loss,
            global_step=global_step)
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op)

    accuracy = tf.metrics.accuracy(
            labels=labels, predictions=predictions["probabilities"])
    tf.summary.scalar('validation_accuracy', accuracy)
    tf.summary.scalar('learning_rate', learning_rate)
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad_histogram".format(v.name), g)
    #summary_op = tf.identity(tf.summary.merge_all(), name="summary_op")


    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": accuracy}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)



def cnn_model_fn_vgg_pretrained_fc7(features, labels, mode, num_classes=20):
    input_layer = tf.reshape(features["x"], [-1, 256, 256, 3])
    #tf.summary.image("image", input_layer)

    if mode == tf.estimator.ModeKeys.TRAIN:
       img_list = tf.unstack(input_layer, num=input_layer.get_shape()[0],axis=0)
       input_layer = tf.stack([preprocess_image(i) for i in img_list],axis=0)
    else:
       input_layer = tf.image.crop_to_bounding_box(input_layer, 16, 16, 224, 224)

    input_layer = tf.cast(input_layer, tf.float32)

    init_gen = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=64,
        kernel_size=[3, 3],
        strides = [1, 1],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.constant_initializer(reader.get_tensor('vgg_16/conv1/conv1_1/weights')),
        bias_initializer=tf.constant_initializer(reader.get_tensor('vgg_16/conv1/conv1_1/biases'))
        )

    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=64,
        kernel_size=[3, 3],
        strides = [1, 1],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.constant_initializer(reader.get_tensor('vgg_16/conv1/conv1_2/weights')),
        bias_initializer=tf.constant_initializer(reader.get_tensor('vgg_16/conv1/conv1_2/biases'))
        )

    pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    conv3 = tf.layers.conv2d(
        inputs=pool1,
        filters=128,
        kernel_size=[3, 3],
        strides = [1, 1],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.constant_initializer(reader.get_tensor('vgg_16/conv2/conv2_1/weights')),
        bias_initializer=tf.constant_initializer(reader.get_tensor('vgg_16/conv2/conv2_1/biases'))
        )

    conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters=128,
        kernel_size=[3, 3],
        strides = [1, 1],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.constant_initializer(reader.get_tensor('vgg_16/conv2/conv2_2/weights')),
        bias_initializer=tf.constant_initializer(reader.get_tensor('vgg_16/conv2/conv2_2/biases'))        
        )
    
    pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)	

    conv5 = tf.layers.conv2d(
        inputs=pool2,
        filters=256,
        kernel_size=[3, 3],
        strides = [1, 1],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.constant_initializer(reader.get_tensor('vgg_16/conv3/conv3_1/weights')),
        bias_initializer=tf.constant_initializer(reader.get_tensor('vgg_16/conv3/conv3_1/biases'))        
        )

    conv6 = tf.layers.conv2d(
        inputs=conv5,
        filters=256,
        kernel_size=[3, 3],
        strides = [1, 1],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.constant_initializer(reader.get_tensor('vgg_16/conv3/conv3_2/weights')),
        bias_initializer=tf.constant_initializer(reader.get_tensor('vgg_16/conv3/conv3_2/biases'))        
        )

    conv7 = tf.layers.conv2d(
        inputs=conv6,
        filters=256,
        kernel_size=[3, 3],
        strides = [1, 1],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.constant_initializer(reader.get_tensor('vgg_16/conv3/conv3_3/weights')),
        bias_initializer=tf.constant_initializer(reader.get_tensor('vgg_16/conv3/conv3_3/biases'))        
        )
    
    pool3 = tf.layers.max_pooling2d(inputs=conv7, pool_size=[2, 2], strides=2)

    conv8 = tf.layers.conv2d(
        inputs=pool3,
        filters=512,
        kernel_size=[3, 3],
        strides = [1, 1],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.constant_initializer(reader.get_tensor('vgg_16/conv4/conv4_1/weights')),
        bias_initializer=tf.constant_initializer(reader.get_tensor('vgg_16/conv4/conv4_1/biases'))
        )

    conv9 = tf.layers.conv2d(
        inputs=conv8,
        filters=512,
        kernel_size=[3, 3],
        strides = [1, 1],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.constant_initializer(reader.get_tensor('vgg_16/conv4/conv4_2/weights')),
        bias_initializer=tf.constant_initializer(reader.get_tensor('vgg_16/conv4/conv4_2/biases'))
        )

    conv10 = tf.layers.conv2d(
        inputs=conv9,
        filters=512,
        kernel_size=[3, 3],
        strides = [1, 1],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.constant_initializer(reader.get_tensor('vgg_16/conv4/conv4_3/weights')),
        bias_initializer=tf.constant_initializer(reader.get_tensor('vgg_16/conv4/conv4_3/biases'))
        )
    
    pool4 = tf.layers.max_pooling2d(inputs=conv10, pool_size=[2, 2], strides=2)

    conv11 = tf.layers.conv2d(
        inputs=pool4,
        filters=512,
        kernel_size=[3, 3],
        strides = [1, 1],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.constant_initializer(reader.get_tensor('vgg_16/conv5/conv5_1/weights')),
        bias_initializer=tf.constant_initializer(reader.get_tensor('vgg_16/conv5/conv5_1/biases'))
        )

    conv12 = tf.layers.conv2d(
        inputs=conv11,
        filters=512,
        kernel_size=[3, 3],
        strides = [1, 1],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.constant_initializer(reader.get_tensor('vgg_16/conv5/conv5_2/weights')),
        bias_initializer=tf.constant_initializer(reader.get_tensor('vgg_16/conv5/conv5_2/biases'))        
        )

    conv13 = tf.layers.conv2d(
        inputs=conv12,
        filters=512,
        kernel_size=[3, 3],
        strides = [1, 1],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.constant_initializer(reader.get_tensor('vgg_16/conv5/conv5_3/weights')),
        bias_initializer=tf.constant_initializer(reader.get_tensor('vgg_16/conv5/conv5_3/biases'))
        )
    
    pool5 = tf.layers.max_pooling2d(inputs=conv13, pool_size=[2, 2], strides=2)

    pool5_flat = tf.reshape(pool5, [-1, 7 * 7 * 512])
    
    dense1 = tf.layers.dense(inputs=pool5_flat, units=4096,
                            activation=tf.nn.relu, 
                            kernel_initializer=tf.constant_initializer(reader.get_tensor('vgg_16/fc6/weights')),
                            bias_initializer=tf.constant_initializer(reader.get_tensor('vgg_16/fc6/biases'))
                            )

    dropout1 = tf.layers.dropout(
        inputs=dense1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    dense2 = tf.layers.dense(inputs=dropout1, units=4096,
                            activation=tf.nn.relu, 
                            kernel_initializer=tf.constant_initializer(reader.get_tensor('vgg_16/fc7/weights')),
                            bias_initializer=tf.constant_initializer(reader.get_tensor('vgg_16/fc7/biases'))
                            )

    dropout2 = tf.layers.dropout(
        inputs=dense2, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    dense3 = tf.layers.dense(inputs=dropout2, units=1000,
                            activation=tf.nn.relu, kernel_initializer=init_gen)

    predictions = {"activation_maps":dense3}
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


    logits = tf.layers.dense(inputs=dense3, units=20, kernel_initializer=init_gen)

    predictions = {
        "probabilities": tf.nn.sigmoid(logits, name="sigmoid_tensor")
    }

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.identity(tf.losses.sigmoid_cross_entropy(
        multi_class_labels=labels, logits=logits), name='loss')

    global_step=tf.train.get_global_step()
    learning_rate = tf.train.exponential_decay(learning_rate=0.0001, global_step=global_step , decay_steps=1000, decay_rate=0.5, staircase=True)


    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
        learning_rate = optimizer._learning_rate

        grads_and_vars=optimizer.compute_gradients(loss)

        train_op = optimizer.minimize(
            loss=loss,
            global_step=global_step)
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op)

    accuracy = tf.metrics.accuracy(
            labels=labels, predictions=predictions["probabilities"])
    tf.summary.scalar('validation_accuracy', accuracy)
    tf.summary.scalar('learning_rate', learning_rate)
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad_histogram".format(v.name), g)
    #summary_op = tf.identity(tf.summary.merge_all(), name="summary_op")


    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": accuracy}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)




def load_pascal(data_dir, split='test'):
    CLASS_NAMES = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor',]
    interim_labels = {}
    interim_weights = {}
    labels_dict = {-1:0, 0:0, 1:1}          #using the uncertain labels as negative examples
    weights_dict = {-1:1, 0:0, 1:1}         #setting confidence of uncertain labels to 0

    for class_name in CLASS_NAMES:
        #obtain each filename and open it
        curr_fn = osp.join(data_dir,'ImageSets/Main/',class_name+'_'+split+'.txt')
        curr_file = open(curr_fn,'r')
        curr_content = curr_file.read().split('\n')
        
        #remove unnecessary entries at the end
        if curr_content[-1] == '':
            curr_content = curr_content[:-1]

        #do the pre-processing and create a dict comprehension each for labels and weights
        curr_content = [line.split() for line in curr_content]
        curr_content = [[(line[0]),int(line[1])] for line in curr_content]
        curr_label_dict = {key:[labels_dict[value]] for [key,value] in curr_content}
        curr_weight_dict = {key:[weights_dict[value]] for [key,value] in curr_content}

        #append the values for each class keys
        if class_name == CLASS_NAMES[0]:
            interim_labels = curr_label_dict
            interim_weights = curr_weight_dict
        else:
            for (key,value) in curr_label_dict.items():
                interim_labels[key] += value
            for [key,value] in curr_weight_dict.items():
                interim_weights[key] += value

        #close the file
        curr_file.close()
    
    labels = []
    weights = []
    image_nums = []
    for (key,value) in interim_labels.items():
        image_nums.append(int(key))
        labels.append(value)
        weights.append(interim_weights[key])

    order = np.argsort(image_nums)

    image_nums = [image_nums[i] for i in order]
    labels = [labels[i] for i in order]
    weights = [weights[i] for i in order]

    images = np.zeros(shape = (len(interim_labels), 256, 256, 3))
    counter = 0
    for (key,value) in interim_labels.items():
        curr_fn = osp.join(data_dir,'JPEGImages/',key+'.jpg')
        img = Image.open(curr_fn)
        img = img.resize((256,256), Image.NEAREST)
        images[counter] = img
        counter += 1

    return np.array(images,dtype=float),np.array(labels,dtype=int),np.array(weights,dtype=int),np.array(image_nums,dtype=int)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a classifier in tensorflow!')
    parser.add_argument(
        'data_dir', type=str, default='data/VOC2007',
        help='Path to PASCAL data storage')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args


def _get_el(arr, i):
    try:
        return arr[i]
    except IndexError:
        return arr


def main():
    args = parse_args()

    eval_data, eval_labels, eval_weights, eval_nums = load_pascal(
        args.data_dir, split='test')

    pascal_classifier = tf.estimator.Estimator(
        model_fn=partial(cnn_model_fn_alexnet_pool5,
                         num_classes=eval_labels.shape[1]),
        model_dir="./01_pascal_alexnet_modified_improved_model_info")

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data, "w": eval_weights, "n": eval_nums},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)

    pred_alexnet_pool5 = list(pascal_classifier.predict(input_fn=eval_input_fn))
    pred_alexnet_pool5 = np.stack([p['activation_maps'] for p in pred_alexnet_pool5])
    pred_alexnet_pool5 = np.ndarray.tolist(pred_alexnet_pool5)


    pascal_classifier = tf.estimator.Estimator(
        model_fn=partial(cnn_model_fn_alexnet_fc7,
                         num_classes=eval_labels.shape[1]),
        model_dir="./01_pascal_alexnet_modified_improved_model_info")

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data, "w": eval_weights, "n": eval_nums},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)

    pred_alexnet_fc7 = list(pascal_classifier.predict(input_fn=eval_input_fn))
    pred_alexnet_fc7 = np.stack([p['activation_maps'] for p in pred_alexnet_fc7])
    pred_alexnet_fc7 = np.ndarray.tolist(pred_alexnet_fc7)


    pascal_classifier = tf.estimator.Estimator(
        model_fn=partial(cnn_model_fn_vgg_pretrained_pool5,
                         num_classes=eval_labels.shape[1]),
        model_dir="../../../01_pascal_VGG16_modified_pretrained_model_info")

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data, "w": eval_weights, "n": eval_nums},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)

    pred_vgg_pool5 = list(pascal_classifier.predict(input_fn=eval_input_fn))
    pred_vgg_pool5 = np.stack([p['activation_maps'] for p in pred_vgg_pool5])
    pred_vgg_pool5 = np.ndarray.tolist(pred_vgg_pool5)


    pascal_classifier = tf.estimator.Estimator(
        model_fn=partial(cnn_model_fn_vgg_pretrained_fc7,
                         num_classes=eval_labels.shape[1]),
        model_dir="../../../01_pascal_VGG16_modified_pretrained_model_info")

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data, "w": eval_weights, "n": eval_nums},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)

    pred_vgg_fc7 = list(pascal_classifier.predict(input_fn=eval_input_fn))
    pred_vgg_fc7 = np.stack([p['activation_maps'] for p in pred_vgg_fc7])
    pred_vgg_fc7 = np.ndarray.tolist(pred_vgg_fc7)


    #common
    eval_nums = np.ndarray.tolist(eval_nums)
    actual_reference_images = [418,521,560,600,618,665,668,696,706,817,846,968,976,1046,1080]
    corresponding_indices_in_pred = [eval_nums.index(a) for a in actual_reference_images]
    


    pseudo_reference_pred = [pred_alexnet_pool5[i] for i in corresponding_indices_in_pred]
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(pred_alexnet_pool5)
    distances, indices = nbrs.kneighbors(pseudo_reference_pred)
    print('Alexnet_pool5_predcitions:\n')
    for pseudo_list in indices:
        print([eval_nums[i] for i in pseudo_list])


    pseudo_reference_pred = [pred_alexnet_fc7[i] for i in corresponding_indices_in_pred]
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(pred_alexnet_fc7)
    distances, indices = nbrs.kneighbors(pseudo_reference_pred)
    print('Alexnet_fc7_predcitions:\n')
    for pseudo_list in indices:
        print([eval_nums[i] for i in pseudo_list])


    pseudo_reference_pred = [pred_vgg_pool5[i] for i in corresponding_indices_in_pred]
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(pred_vgg_pool5)
    distances, indices = nbrs.kneighbors(pseudo_reference_pred)
    print('VGG_pool5_predcitions:\n')
    for pseudo_list in indices:
        print([eval_nums[i] for i in pseudo_list])


    pseudo_reference_pred = [pred_vgg_fc7[i] for i in corresponding_indices_in_pred]
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(pred_vgg_fc7)
    distances, indices = nbrs.kneighbors(pseudo_reference_pred)
    print('VGG_fc7_predcitions:\n')
    for pseudo_list in indices:
        print([eval_nums[i] for i in pseudo_list])


if __name__ == "__main__":
    main()
