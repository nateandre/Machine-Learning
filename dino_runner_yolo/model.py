"""
Holds the functions necessary to use the trained tensorflow model
"""

import tensorflow as tf

# Defining constant layer for 2d convolution, batch norm, and activation
def conv(the_input,layer,f,ks):
    """
    the_input: the layer which will be used as input in conv layer
    layer: specifies the layer number for naming sections of graph
    f (filters): the number of filters to be used for conv layer
    ks (kernel_size): kernel size for conv2d layer
    Note - conv2d layers all use padding
    """
    layer = str(layer)
    Z = tf.layers.conv2d(the_input,filters=f,kernel_size=[ks,ks],strides=(1,1),padding="same",name="Z"+layer,kernel_initializer=tf.contrib.layers.xavier_initializer(seed=0))
    Bn = tf.layers.batch_normalization(Z,name="Bn"+layer)
    A = tf.nn.leaky_relu(Bn,alpha=0.1,name="A"+layer)
    return A

# Building the forward pass based on Darket-19
# Note - forward pass will use leaky_relu
def forward_pass(X):
    input_layer = tf.reshape(X,[-1,448,448,3]) # Input shape of images
    S1 = conv(input_layer,1,32,3)
    P1 = tf.layers.max_pooling2d(S1,pool_size=[2,2],strides=2,padding="valid",name="P1") # 224x224
    S2 = conv(P1,2,64,3)
    P2 = tf.layers.max_pooling2d(S2,pool_size=[2,2],strides=2,padding="valid",name="P2") # 112x112
    S3 = conv(P2,3,128,3)
    S4 = conv(S3,4,64,1)
    S5 = conv(S4,5,128,3)
    P5 = tf.layers.max_pooling2d(S5,pool_size=[2,2],strides=2,padding="valid",name="P5") # 56x56
    S6 = conv(P5,6,256,3)
    S7 = conv(S6,7,128,1)
    S8 = conv(S7,8,256,3)
    P8 = tf.layers.max_pooling2d(S8,pool_size=[2,2],strides=2,padding="valid",name="P8") # 28x28
    S9 = conv(P8,9,512,3)
    S10 = conv(S9,10,256,1)
    S11 = conv(S10,11,512,3)
    S12 = conv(S11,12,256,1)
    S13 = conv(S12,13,512,3)
    P13 = tf.layers.max_pooling2d(S13,pool_size=[2,2],strides=2,padding="valid",name="P13") #14x14
    S14 = conv(P13,14,1024,3)
    S15 = conv(S14,15,512,1)
    S16 = conv(S15,16,1024,3)
    S17 = conv(S16,17,512,1)
    S18 = conv(S17,18,2014,3)
    # Final layer - no batch norm, linear activation
    S19 = tf.layers.conv2d(S18,filters=14,kernel_size=[1,1],strides=(1,1),padding="valid",name="S19",activation=None)
    return S19