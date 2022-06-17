'''
references 

https://arxiv.org/pdf/1805.08318.pdf
https://github.com/brain-research/self-attention-gan
https://github.com/taki0112/Self-Attention-GAN-Tensorflow

excellent https://lilianweng.github.io/posts/2018-06-24-attention 

Attention(Q,K,V) = softmax( ( Q * K.T ) / sqrt(n) ) * V

Key : f(x) = W_f * x
Query: g(x) = W_g * x
Value: h(x) = W_h * x


def attention(self, x, channels):
    f = conv(x, channels // 8, kernel=1, stride=1, sn=self.sn, scope='f_conv') # [bs, h, w, c']
    g = conv(x, channels // 8, kernel=1, stride=1, sn=self.sn, scope='g_conv') # [bs, h, w, c']
    h = conv(x, channels, kernel=1, stride=1, sn=self.sn, scope='h_conv') # [bs, h, w, c]

    # N = h * w
    s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True) # # [bs, N, N]

    beta = tf.nn.softmax(s)  # attention map

    o = tf.matmul(beta, hw_flatten(h)) # [bs, N, C]
    gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

    o = tf.reshape(o, shape=x.shape) # [bs, h, w, C]
    o = conv(o, channels, kernel=1, stride=1, sn=self.sn, scope='attn_conv')

    x = gamma * o + x


    batch size : bs
    image,channel size : h,w,c
    attention channel' size: c/k, where k = 8 per paper

'''

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import numpy as np

def hw_flatten(x) :
    shape = x.get_shape().as_list()
    dim = np.prod(shape[1:-1])
    return tf.reshape(x, [-1, dim, shape[-1]])

class Attention2D(keras.layers.Layer):
    def __init__(self,channel,trainable=True,**kwargs):
        super(Attention2D, self).__init__(**kwargs)
        self.channel = channel # channel from prior layer
        self.trainable = trainable

    def get_config(self):
        cfg = super().get_config()
        return cfg

    def build(self, input_shape):
        
        sn = tfa.layers.SpectralNormalization
        mykwargs = dict(kernel_size=1, strides=1, use_bias=False, padding="same",trainable=self.trainable)
        
        self.conv_f = sn(layers.Conv2D(self.channel // 8, **mykwargs))# [bs, h, w, c']
        self.conv_g = sn(layers.Conv2D(self.channel // 8, **mykwargs)) # [bs, h, w, c']
        self.conv_h = sn(layers.Conv2D(self.channel, **mykwargs)) # [bs, h, w, c]
        self.conv_v = sn(layers.Conv2D(self.channel, **mykwargs))# [bs, h, w, c]

        self.gamma = self.add_weight(name='gamma',shape=(1,),initializer='zeros',trainable=self.trainable)

    def call(self, inputs):
        f = self.conv_f(inputs) # key
        g = self.conv_g(inputs) # query
        h = self.conv_h(inputs) # value

        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True) # [bs, N, N], N = h * w
        beta = tf.nn.softmax(s) # beta is your attention map
        
        o = tf.matmul(beta, hw_flatten(h)) # [bs, N, C]
        input_shape = (-1,)+tuple(inputs.get_shape().as_list()[1:])
        o = tf.reshape(o, shape=input_shape) # [bs, h, w, C]
        o = self.conv_v(o)

        y = self.gamma * o + inputs

        return y


if __name__ == "__main__":
    input_size = (128,128,32)
    c = input_size[-1]
    x = layers.Input(input_size)
    y = Attention2D(c)(x)
    model = keras.Model(x,y)
    model.summary()
