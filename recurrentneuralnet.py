# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 12:00:01 2020

@author: SOFTECH
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np 

class TimeSeriesData():
    def __init__(self, num_points, xmin, xmax):
        self.num_point = num_points
        self.xmin = xmin
        self.xmax =xmax
        self.resolution = (xmax-xmin)/num_points
        self.x_data = np.linspace(xmin,xmax, num_points)
        self.y_true = np.sin(self.x_data)
    def ret_true(self,x_series):
        return np.sin(x_series)
    def next_batch(self, batch_size, steps,return_batch_ts =False):
        rand_ts = np.random.rand(batch_size, 1)
        
        ts_start = rand_ts*(self.xmax-self.xmin-(steps*self.resolution))
        
        batch_ts = ts_start+ np.arange(0.0, steps+1)*self.resolution
        
        y_batch = np.sin(batch_ts)
        
        if return_batch_ts:
            return y_batch[:,:-1].reshape(-1,steps, 1), y_batch[:,1:].reshape(-1,steps, 1),batch_ts
        else:
            return y_batch[:,:-1].reshape(-1,steps, 1), y_batch[:,1:].reshape(-1,steps, 1)
ts_data = TimeSeriesData(250,0,10)
ply1,y2,ts = ts_data.next_batch(1, 30,True)
ts =ts.flatten()[1:]
#plt.plot(ts_data.x_data, ts_data.y_true, label = 'sin(t)')
#plt.plot(ts, y2.flatten(),'*', label ='single training insta')
#plt.legend()
training_inst = np.linspace(5,5+ts_data.resolution*(30+1), 30+1)
plt.plot(training_inst[:-1], ts_data.ret_true(training_inst[:-1]), 'bo', markersize=15, alpha =0.5, label = 'instance')
plt.plot(training_inst[1:], ts_data.ret_true(training_inst[1:]), 'ko', markersize=7, label = 'Target')
num_inputs = 1
num_neuron = 100
num_outputs = 1
learning_rate = 0.0001
num_train_iteration =2000
batch_size = 1

X =tf.placeholder(tf.float32, [None, 30, num_inputs])
y = tf.placeholder(tf.float32, [None, 30, num_outputs])
cell = tf.contrib.rnn.BasicRNNCell(num_units=num_neuron, activation=tf.nn.relu)
cell = tf.contrib.rnn.OutputProjectionWrapper(cell, output_size= num_outputs)
output,state =  tf.nn.dynamic_rnn(cell,X, dtype =tf.float32)
loss = tf.reduce_mean(tf.square(output-y))  
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train = optimizer.minimize(loss)
init = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    for iteration in range(num_train_iteration):
        x_batch, y_batch =ts_data.next_batch(batch_size, 30)
        sess.run(train, feed_dict = {X:x_batch, y:y_batch}) 
        if iteration%100 == 0:
            mse =loss.eval(feed_dict = {X:x_batch, y:y_batch})
            print(iteration, 'mse', mse)
    saver.save(sess, 'rnn_code')
    
#with tf.Session() as sess:
#    saver.restore(sess, 'rnn_code')
 #   x_new = np.sin(np.array(training_inst[:-1].reshape(-1, 30,num_outputs)))
    y_pred= sess.run(output, feed_dict={X:x_new})
