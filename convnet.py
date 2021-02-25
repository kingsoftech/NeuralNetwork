# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 23:30:46 2020

@author: SOFTECH
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('mnist/', one_hot = 'True')

n_classes = 10
batch_size =128
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def conv2d(x,W):
    return tf.nn.conv2d(x,W, strides =[1,1,1,1], padding = 'SAME')
def maxpool2d(x):
    return tf.nn.max_pool2d(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
def convnet(x):
    weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,1,32])),
               'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
               'W_fc':tf.Variable(tf.random_normal([7*7*64,1024])),
               'output':tf.Variable(tf.random_normal([1024, n_classes]))
               }
    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
              'b_conv2':tf.Variable(tf.random_normal([64])),
              'b_fc':tf.Variable(tf.random_normal([1024])),
              'b_out':tf.Variable(tf.random_normal([10])),
              }
    x = tf.reshape(x, shape=[-1,28,28,1])
    x = conv2d(x, weights['W_conv1'])
    x = maxpool2d(x)
    
    x = conv2d(x, weights['W_conv2'])
    x = maxpool2d(x)
    
    x = tf.reshape(x, shape=[-1,7*7*64])
    x = tf.matmul(x, weights['W_fc']) + biases['b_fc']
    x = tf.nn.relu(x)
    
    x = tf.matmul(x,weights['output'])+biases['b_out']
    
    return x
def train_neural_network(x):
    prediction = convnet(x)
    
    output = tf.nn.softmax(prediction)
    #calculating the loses
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y,logits = prediction))
    #optimization
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)
    hm_epochs = 10
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                sess.run(optimizer, feed_dict = {x:batch_x, y:batch_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', hm_epochs)
            
        correct = tf.equal(tf.arg_max(prediction, 1), tf.arg_max(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('accuracy', sess.run(accuracy, feed_dict ={x:mnist.test.images, y:mnist.test.labels}))
        labels =mnist.test.labels[:10]
        result = sess.run(output, feed_dict= {x:mnist.test.images[:10]})
        print('predicted labels',sess.run(tf.argmax(result)))
        print('actual label:',sess.run(tf.argmax(labels)))
           
       # sess.run(prediction, feed_dict = {x})
train_neural_network(x)
