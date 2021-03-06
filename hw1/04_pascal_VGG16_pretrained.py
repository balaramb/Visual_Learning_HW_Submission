from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python import pywrap_tensorflow

reader = pywrap_tensorflow.NewCheckpointReader("vgg_16.ckpt")

# Imports
import sys
import numpy as np
import tensorflow as tf
import argparse
import os.path as osp
from PIL import Image
from functools import partial

from eval import compute_map
import models

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
    temp2 = tf.random_crop (temp1,[224,224,3])
    return temp2

def cnn_model_fn(features, labels, mode, num_classes=20):
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
        # # Generate predictions (for PREDICT and EVAL mode)
        # "classes": tf.argmax(input=logits, axis=1),
        # # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # # `logging_hook`.
        #CHANGED THIS
        "probabilities": tf.nn.sigmoid(logits, name="sigmoid_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

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

def load_pascal(data_dir, split='train'):
    """
    Function to read images from PASCAL data folder.
    Args:
        data_dir (str): Path to the VOC2007 directory.
        split (str): train/val/trainval split to use.
    Returns:
        images (np.ndarray): Return a np.float32 array of
            shape (N, H, W, 3), where H, W are 224px each,
            and each image is in RGB format.
        labels (np.ndarray): An array of shape (N, 20) of
            type np.int32, with 0s and 1s; 1s for classes that
            are active in that image.
        weights: (np.ndarray): An array of shape (N, 20) of
            type np.int32, with 0s and 1s; 1s for classes that
            are confidently labeled and 0s for classes that 
            are ambiguous.
    """
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
        curr_content = [[line[0],int(line[1])] for line in curr_content]
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
    for (key,value) in interim_labels.items():
        labels.append(value)
    for (key,value) in interim_weights.items():
        weights.append(value)

    images = np.zeros(shape = (len(interim_labels), 256, 256, 3))
    counter = 0
    for (key,value) in interim_labels.items():
        curr_fn = osp.join(data_dir,'JPEGImages/',key+'.jpg')
        img = Image.open(curr_fn)
        img = img.resize((256,256), Image.NEAREST)
        images[counter] = img
        counter += 1

    return np.array(images,dtype=float),np.array(labels,dtype=int),np.array(weights,dtype=int)

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
    global learning_rate
    global grads_and_vars
    args = parse_args()
    # Load training and eval data
    train_data, train_labels, train_weights = load_pascal(
        args.data_dir, split='trainval')
    eval_data, eval_labels, eval_weights = load_pascal(
        args.data_dir, split='test')

    config = tf.estimator.RunConfig(
        save_summary_steps=400,
        save_checkpoints_steps=400,
        keep_checkpoint_max=1,
        log_step_count_steps=400
    )
    pascal_classifier = tf.estimator.Estimator(
        model_fn=partial(cnn_model_fn,
                         num_classes=train_labels.shape[1]),
        model_dir="./01_pascal_VGG16_modified_pretrained_model_info",
        config = config)
    tensors_to_log = {"loss": "loss"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=400)
    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data, "w": train_weights},
        y=train_labels,
        batch_size=10,
        num_epochs=None,
        shuffle=True)
    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data, "w": eval_weights},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)

        
    mAP_values = []
    summary_hook = tf.train.SummarySaverHook(
                 save_steps = 2000,
                 output_dir = 'VGGParamsFinal',
                 scaffold = tf.train.Scaffold(summary_op=tf.summary.merge_all())
                 )
    for _ in range(10):
        pascal_classifier.train(
            input_fn=train_input_fn,
            steps=400,
            hooks=[logging_hook,summary_hook])
        pred = list(pascal_classifier.predict(input_fn=eval_input_fn))
        pred = np.stack([p['probabilities'] for p in pred])
        print(pred)
        rand_AP = compute_map(
            eval_labels, np.random.random(eval_labels.shape),
            eval_weights, average=None)
        print('Random AP: {} mAP'.format(np.mean(rand_AP)))
        gt_AP = compute_map(
            eval_labels, eval_labels, eval_weights, average=None)
        print('GT AP: {} mAP'.format(np.mean(gt_AP)))
        AP = compute_map(eval_labels, pred, eval_weights, average=None)
        print('Obtained {} mAP'.format(np.mean(AP)))
        print('per class:')
        mAP_values.append(np.mean(AP))
        for cid, cname in enumerate(CLASS_NAMES):
            print('{}: {}'.format(cname, _get_el(AP, cid)))
    
    filen = open('mAP_01_pascal_VGG16_modified_pretrained.txt','w')
    for i in mAP_values:
        filen.write(str(i)+'\n')
    filen.close()


if __name__ == "__main__":
    main()
