# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 17:09:37 2020

@author: SOFTECH
"""
# importing the necessary libraries
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
#Reading the mnist dataset
mnist = input_data.read_data_sets("mnist/", one_hot = True)
# numbers of nuerons per layer
no_of_nuerons_l1 = 500
no_of_nuerons_l2 = 500
no_of_nuerons_l3 = 500
# number of classes to be predicted
n_classes = 10
#the batch size of data you want to pass
batch_size = 100
#creating the placeholders for x, y
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')
#creating the network
def neural_network_model(data):
    # creating the weights and biases
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784, no_of_nuerons_l1])),
                      'biases': tf.Variable(tf.random_normal([no_of_nuerons_l1]))}
    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([no_of_nuerons_l1, no_of_nuerons_l2])),
                      'biases': tf.Variable(tf.random_normal([no_of_nuerons_l2]))}
    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([no_of_nuerons_l2, no_of_nuerons_l3])),
                      'biases': tf.Variable(tf.random_normal([no_of_nuerons_l3]))}
    output_layer = {'weights': tf.Variable(tf.random_normal([no_of_nuerons_l3, n_classes])),
                      'biases': tf.Variable(tf.random_normal([n_classes]))}
    #forward pass
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']),hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)
    
    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']),hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)
    
    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']),hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)
    
    output = tf.matmul(l3, output_layer['weights'])+output_layer['biases']
    
    return output
#training process
def train_neural_network(x):
    prediction = neural_network_model(x)
    
    output = tf.nn.softmax(prediction)
    #calculating the loses
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y,logits = prediction))
    #optimization
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    hm_epochs = 10
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict = {x:batch_x, y:batch_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss', epoch_loss)
            
        correct = tf.equal(tf.arg_max(prediction, 1), tf.arg_max(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('accuracy', sess.run(accuracy, feed_dict ={x:mnist.test.images, y:mnist.test.labels}))
        labels =mnist.test.labels[:10]
        result = sess.run(output, feed_dict= {x:mnist.test.images[:10]})
        print('predicted labels',sess.run(tf.argmax(result)))
        print('actual label:',sess.run(tf.argmax(labels)))
           
       # sess.run(prediction, feed_dict = {x})
train_neural_network(x)
