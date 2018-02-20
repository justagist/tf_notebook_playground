import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# %matplotlib inline

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

'''
Extracting MNIST_data/train-images-idx3-ubyte.gz
Extracting MNIST_data/train-labels-idx1-ubyte.gz
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
'''

device_string = "/cpu:0"

with tf.device(device_string):
    x = tf.placeholder(tf.float32, [None, 784])
    ## Encoder

    with tf.variable_scope('encoder'):
        W_fc1 = tf.Variable(tf.random_uniform([784,50], dtype=tf.float32))
        b_fc1 = tf.Variable(tf.random_uniform([50], dtype=tf.float32)) 
        
        ## Bottleneck
        W_fc2 = tf.Variable(tf.random_uniform([50,2], dtype=tf.float32))
        b_fc2 = tf.Variable(tf.random_uniform([2], dtype=tf.float32)) 
    
        
        h1_enc = tf.nn.tanh(tf.matmul(x, W_fc1) + b_fc1)
        encoder_op = tf.nn.tanh(tf.matmul(h1_enc, W_fc2) + b_fc2)


    with tf.variable_scope('decoder'):

        code_in = tf.placeholder(tf.float32,[None,2])
        
        W_fc1 = tf.Variable(tf.random_uniform([2,50], dtype=tf.float32))
        b_fc1 = tf.Variable(tf.random_uniform([50], dtype=tf.float32)) 
        
        W_fc2 = tf.Variable(tf.random_uniform([50,784], dtype=tf.float32))
        b_fc2 = tf.Variable(tf.random_uniform([784], dtype=tf.float32)) 
        
        h1_dec = tf.nn.tanh(tf.matmul(encoder_op, W_fc1) + b_fc1)
        
        decode = tf.nn.tanh(tf.matmul(h1_dec, W_fc2) + b_fc2)
        
        h1_dec = tf.nn.tanh(tf.matmul(code_in, W_fc1) + b_fc1)

        decoder = tf.nn.tanh(tf.matmul(h1_dec, W_fc2) + b_fc2)

with tf.device(device_string):
    y_ = tf.placeholder(tf.float32, [None, 784]) # Correct answer
    pv = tf.placeholder(tf.float32, [1, 2]) # Sparsity prob
    beta = tf.placeholder(tf.float32, [1, 1]) # Sparsity penalty (lagrange multiplier)

# Aditional loss for penalising high activations (http://ufldl.stanford.edu/tutorial/unsupervised/Autoencoders/)
# with tf.device(device_string):
#     p = tf.nn.softmax(encoder_op)
#     kl_divergence = tf.reduce_mean(tf.mul(pv,tf.log(tf.div(pv,p))))
#     sparsity_loss = tf.mul(beta,kl_divergence)

with tf.device(device_string):
    weight_decay_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
    squared_loss = tf.reduce_sum(tf.square(decode - y_))

with tf.device(device_string):
    loss_op = tf.reduce_mean(squared_loss) + 0.1*weight_decay_loss #+ sparsity_loss

with tf.device(device_string):
    train_op = tf.train.AdadeltaOptimizer(learning_rate=0.1, rho=0.1, epsilon=0.0001).minimize(loss_op)
    init_op = tf.initialize_all_variables()
