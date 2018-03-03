from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn.manifold import TSNE
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
import matplotlib.pyplot as plt
from matplotlib import pylab


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

def cnn_model_fn(features, labels, mode, num_classes):
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
    save_keys = []
    count = 0
    for (key,value) in interim_labels.items():
        image_nums.append(int(key))
        labels.append(value)
        weights.append(interim_weights[key])
        save_keys.append(key)
        count += 1
        if count==999:
	    break

    order = np.argsort(image_nums)

    image_nums = [image_nums[i] for i in order]
    labels = [labels[i] for i in order]
    weights = [weights[i] for i in order]

    images = np.zeros(shape = (len(labels), 256, 256, 3))
    counter = 0
    for key in save_keys:
        print(key)
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
        model_fn=partial(cnn_model_fn,
                         num_classes=eval_labels.shape[1]),
        model_dir="./01_pascal_alexnet_modified_improved_model_info")

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data, "w": eval_weights, "n": eval_nums},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
	
    pred = list(pascal_classifier.predict(input_fn=eval_input_fn))
    pred = np.stack([p['activation_maps'] for p in pred])
    predtSNE = TSNE(n_components=2).fit_transform(pred)
    color = np.linspace(1,0,20)
    temp1 = np.sum(eval_labels*color, axis=1)
    temp2 = np.count_nonzero(eval_labels,axis=1)
    fc7color = temp1/temp2
  
    fig, ax = plt.subplots()
    scatter = ax.scatter(predtSNE[:,0], predtSNE[:,1], c=fc7color, cmap = plt.get_cmap('tab20c'))
    cbar = plt.colorbar(scatter)
    cbar.ax.set_yticklabels([], minor=False)
    pylab.savefig('tsne.png')

if __name__ == "__main__":
    main()
