#!/usr/bin/env python3

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
import sys
import numpy as np 
from PIL import Image, ImageOps


class Model():

    sess = tf.Session()

    def __init__(self, load_from="saved_model/model.ckpt"):

        self.filter_size = 5
        self.size_hidden_layer = 64
        self.height_pic = 112
        self.width_pics = 112
        self.size_classes = 2
        self.learning_rate = 0.00001



        # placeholder for the data training. One neuron for each pixel
        self.x = tf.placeholder(tf.float32, [None, self.height_pic * self.width_pics])
        # correct output
        self.y = tf.placeholder(tf.float32, [None, self.size_classes])

        # reshape for WxH, and only macro - 1 channel
        self.x_image = tf.reshape(self.x, [-1, self.height_pic, self.width_pics, 1])

        # create first convolutional followed by pooling
        self.conv1 = self.conv_layer(self.x_image, shape=[self.filter_size, self.filter_size, 1, 32]) # in this case, a filter of filter_sizexfilter_size, used 32 times over the image
        # the result of conv1, which is 112x112x32, we feed to pooling
        self.conv1_pool = self.max_pool_2x2(self.conv1) # the result of this first polling will be 56X56X32

        # create second convolutional followed by pooling. 32 came from the first convol
        self.conv2 = self.conv_layer(self.conv1_pool, shape=[self.filter_size, self.filter_size, 32, 64]) # the result here will be 56X56X64
        self.conv2_pool = self.max_pool_2x2(self.conv2) # the result will be 28X28X64

        # create a third layer
        self.conv3 = self.conv_layer(self.conv2_pool, shape=[self.filter_size, self.filter_size, 64, 128]) # the result here will be 28X28X128
        self.conv3_pool = self.max_pool_2x2(self.conv3) # the result will be 14X14X128

        # create a forth layer
        self.conv4 = self.conv_layer(self.conv3_pool, shape=[self.filter_size, self.filter_size, 128, 256]) # the result here will be 14X14X256
        self.conv4_pool = self.max_pool_2x2(self.conv4) # the result will be 7X7X256

        # flat the final results, for then put in a fully connected layer
        # since the result data is 28X23X64 and we want to flat, Just a big array
        self.conv5_flat = tf.reshape(self.conv4_pool, [-1, 7*7*256])

        # create fully connected layer and train - Foward
        self.full_1 = tf.nn.relu(self.full_layer(self.conv5_flat, self.size_hidden_layer))

        # for dropout
        self.keep_prob = tf.placeholder(tf.float32)
        self.full1_drop = tf.nn.dropout(self.full_1, keep_prob=self.keep_prob) # for test, we will use full drop(no drops)

        # for output - For training
        # In this case, weights will have size of 10 - Because we have 10 classes as output
        self.y_conv = self.full_layer(self.full1_drop, self.size_classes) # one last layer, for the outputs

        # error function. Using cross entropy to calculate the distance between probabilities
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y_conv, labels=self.y))

        # define which optimezer to use. How to change bias and weights to get to the result
        self.optimizer_train = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cross_entropy)

        # correct prediction
        self.correct_pred = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y, 1))

        # check for accuracy
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32)) 


    # create a weight variable - Filter
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1) # to init with a random normal distribution with stndard deviation of 1
        return tf.Variable(initial) # create a variable

    # create a bias variable
    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # execute and return a convolutional over a data x, with a filter/weights W
    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    # create, execute and return the result of a convolutional layer, already with an activation 
    def conv_layer(self, input_, shape):
        W = self.weight_variable(shape)
        b = self.bias_variable([shape[3]])
        return tf.nn.relu(self.conv2d(input_, W) + b)

    # with X as a result of a convolutional layer, we will max pool
    # with a filter of 2x2
    # this basically gets the most important features, and reduces the size of the inputs for the final densed layer
    def max_pool_2x2(self, x, ksize_=[1, 2, 2, 1], strides_=[1, 2, 2, 1]):
        return tf.nn.max_pool(x, ksize=ksize_, strides=strides_, padding='SAME')

    # After all convolutional layers has been applied, we get all the final results, and make a full connected layer
    def full_layer(self, input, size):
        in_size = int(input.get_shape()[1])
        W = self.weight_variable([in_size, size])
        b = self.bias_variable([size])
        # tf.matmul is a matrix multiplication from tensorn. This is the basic idea of ML
        # multiply 2 matrix and add a bias. This is the foward when we implement ANN
        return tf.matmul(input, W) + b

    # the idea is to select a random part of the data, with size = MINIBATCH_SIZE
    def next_batch(self, data, label_data, size_data):
        l_bound = random.randint(0,size_data - self.MINIBATCH_SIZE)
        u_bound = l_bound + self.MINIBATCH_SIZE

        return data[l_bound:u_bound], label_data[l_bound:u_bound]
    
    def restore_model(self, load_from):
        # create a variable to load trained graph
        saver = tf.train.Saver()
        # restore Graph trained
        saver.restore(self.sess, load_from)
        print('[LOG] Model Loaded')

    def infer(self, img, load_from="saved_model/model.ckpt"):
        #Restore saved model
        self.restore_model(load_from)
        pred = self.sess.run(self.y_conv, feed_dict={self.x: [img.reshape(self.width_pics*self.height_pic)], self.keep_prob: 1.0})
        pred = tf.argmax(pred, 1)
        
        # 0 for non-glasses and 1 for glasses
        return self.sess.run(pred)[0]    

def load_image(img_path):
    #Load image and convert to grayscale
    img = Image.open(img_path)
    img = ImageOps.grayscale(img)
    img = img.resize((112,112))
    img = np.array(img, dtype=np.uint8)
    return img


if __name__ == '__main__':

    #Create and load saved graph
    model = Model()

    #Check command line args
    if len(sys.argv) == 2:
        img_path = sys.argv[1]
    else:
        print('[ERROR] Provide path to input image')
        exit()

    #Load image and preprocess
    img = load_image(img_path)

    #Run inference
    pred = model.infer(img)
    
    print("\nYES, Person is using Glasses\n" if pred ==1 else "NO, Person is not using Glasses\n")