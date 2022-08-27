import os,sys

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K

def prepare_models(
    input_dim=(8,64,64,1),latent_dim=(8,16,16,10),
    mykernel=5,mystrides=(1,2,2),
    num_list=[64,64],
    dis_num_list=[16,32,64],
    ):

    lx,ly,lz,lw = latent_dim

    # encorder  -------------------
    ENC = 'ENC'
    with tf.name_scope(ENC) as scope:
        
        class Sampling(layers.Layer):
            """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

            def call(self, inputs):
                z_mean, z_log_var = inputs
                latent_shape = tf.shape(z_mean)
                epsilon = tf.keras.backend.random_normal(shape=latent_shape)
                return z_mean + tf.exp(0.5 * z_log_var) * epsilon

        # encoder -------------------
        encoder_inputs = keras.Input(shape=input_dim)

        def res_down(filters,x):
            r = layers.Conv3D(filters, kernel_size=1, strides=mystrides, padding='same')(x)
            d = layers.Conv3D(filters, kernel_size=mykernel, strides=1, padding='same')(x)
            d = layers.BatchNormalization()(d)
            d = layers.LeakyReLU(alpha=0.2)(d)
            #d = layers.concatenate([x,d],axis=-1)
            d = layers.Conv3D(filters, kernel_size=mykernel, strides=mystrides, padding='same')(d)
            d = layers.BatchNormalization()(d)
            d = layers.add([d,r])
            d = layers.LeakyReLU(alpha=0.2)(d)
            return d
            
        for l,num in enumerate(num_list):
            if l == 0:
                x = encoder_inputs 
            x = res_down(num,x)
            # TODO: add in attention

        z_mean = layers.Conv3D(lw, 1, activation="linear")(x)
        z_log_var = layers.Conv3D(lw, 1, activation="linear")(x)
        z = Sampling()([z_mean, z_log_var])

        encoder = keras.Model([encoder_inputs], [z_mean, z_log_var, z], name="encoder")
        encoder.summary()

    # decoder  -------------------
    DEC = 'DEC'
    with tf.name_scope(DEC) as scope:

        def res_up(filters,x):
            r = layers.Conv3DTranspose(filters, kernel_size=1, strides=mystrides, padding='same')(x)
            d = layers.Conv3D(filters, kernel_size=mykernel, strides=1, padding='same')(x)
            d = layers.BatchNormalization()(d)
            d = layers.LeakyReLU(alpha=0.2)(d)
            #d = layers.concatenate([x,d],axis=-1)
            d = layers.Conv3DTranspose(filters, kernel_size=mykernel, strides=mystrides, padding='same')(d)
            d = layers.BatchNormalization()(d)
            d = layers.add([d,r])
            d = layers.LeakyReLU(alpha=0.2)(d)
            return d

        decoder_inputs = keras.Input(shape=latent_dim)

        for l,num in enumerate(num_list[::-1]):
            if l == 0:
                x = decoder_inputs

            x = res_up(num,x)

        decoder_outputs = layers.Conv3D(input_dim[-1], mykernel, activation="tanh", padding="same")(x)
        decoder = keras.Model(decoder_inputs, decoder_outputs, name="decoder")
        decoder.summary()

    # discriminator  -------------------
    DISCR = 'DISCR'
    with tf.name_scope(DISCR) as scope:
        discr_strides = (1,2,2)
        discr_mykernel = (1,3,3)
        def discr_res_down(filters,x):
            r = layers.Conv3D(filters, kernel_size=1, strides=discr_strides, padding='same')(x)
            d = layers.Conv3D(filters, kernel_size=discr_mykernel, strides=1, padding='same')(x)
            d = layers.BatchNormalization()(d)
            d = layers.LeakyReLU(alpha=0.2)(d)
            d = layers.concatenate([x,d],axis=-1)
            d = layers.Conv3D(filters, kernel_size=discr_mykernel, strides=discr_strides, padding='same')(d)
            d = layers.BatchNormalization()(d)
            d = layers.LeakyReLU(alpha=0.2)(d)
            return d

        discr_inputs = keras.Input(shape=input_dim)
        for l,num in enumerate(dis_num_list):
            if l == 0:
                x = discr_inputs
            
            x = discr_res_down(num,x)

        discr_output = layers.Conv3D(1, mykernel, activation="sigmoid", padding="same")(x)
        discr = keras.Model(discr_inputs,discr_output, name="discr")
        discr.summary()

    return encoder, decoder, discr, input_dim, latent_dim, None

# `tf.name_scope` doesn't work with layers.* yet, thus have `name=` all over. TODO: remove `name=` once tf.name_scope works.
#
# source https://github.com/pangyuteng/hello-mnist/blob/4ac2511a0829a0e525085c12ee490c94250481bd/selfsuper/hello-vae/8-fc-conv-vae-gan/vaegan.py
# adding cvae  https://agustinus.kristia.de/techblog/2016/12/17/conditional-vae
#              https://gist.github.com/naotokui/2201cf1cab6608aee18d34c0ea748f84
#
class VAEGAN(keras.Model):
    def __init__(self, 
        input_dim=(8,64,64,1),latent_dim=(8,16,16,10),
        num_list=[64,64],dis_num_list=[16,32,64],
        mykernel=5,mystrides=(1,2,2),
        beta_init=0, **kwargs):
        super(VAEGAN, self).__init__(**kwargs)

        self.encoder, self.decoder, self.discr, \
            self.input_dim, self.latent_dim, _ = prepare_models(
                input_dim=input_dim,latent_dim=latent_dim,
                num_list=num_list,dis_num_list=dis_num_list,
                mystrides=mystrides,mykernel=mykernel,                
            )

        self.beta = K.variable(beta_init,name='kl_beta')
        self.beta._trainable = False
        
    def call(self, inputs, *args, training=False, **kwargs):
        x = inputs
        z_mean, z_log_var, z  = self.encoder(x)
        x_hat = self.decoder(z)
        d_x_hat = self.discr(x_hat)
        return z_mean, z_log_var, z, x_hat, d_x_hat

    def train_step(self, data):

        with tf.GradientTape(persistent=True) as tape:
            
            cutout_x, x = data
            
            ################################################################
            # x -> enc > dec > discr 
            
            z_mean, z_log_var, z  = self.encoder(cutout_x)
            
            x_hat = self.decoder(z)
            d_x_hat = self.discr(x_hat)
            
            ################################################################
            # latent -> dec > discr
            
            latent_shape = tf.shape(z)
            z_p = tf.keras.backend.random_normal(shape=latent_shape)
            
            x_p = self.decoder(z_p)
            d_p = self.discr(x_p)
            
            ################################################################
            # x -> discr
            d_x = self.discr(x)
            
            ################################################################
            ################################################################
            # recon loss             
            reconstruction_loss = tf.reduce_mean(
                tf.keras.losses.mean_squared_error(x, x_hat)
            )
            
            reconstruction_loss *= self.input_dim[0]*self.input_dim[1]*self.input_dim[2]
            
            ################################################################
            # latent loss 
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5 * self.latent_dim[0]*self.latent_dim[1]*self.latent_dim[2]

            ################################################################
            # gen loss
            c = 0.05
            d_shape = tf.shape(d_x_hat)
            offset = c*tf.keras.backend.random_normal(shape=d_shape)
            gen_fake_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(
                    tf.clip_by_value(tf.ones_like(d_x_hat)-offset,0,1),
                    d_x_hat
            ))
            offset = c*tf.keras.backend.random_normal(shape=d_shape)
            gen_rand_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(
                    tf.clip_by_value(tf.ones_like(d_p)-offset,0,1),
                    d_p
            ))
            offset = c*tf.keras.backend.random_normal(shape=d_shape)
            gen_real_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(
                    tf.clip_by_value(tf.zeros_like(d_x)+offset,0,1),
                    d_x
            ))

            ################################################################
            # discr loss 
            offset = c*tf.keras.backend.random_normal(shape=d_shape)
            discr_real_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(
                    tf.clip_by_value(tf.ones_like(d_x)-offset,0,1),
                    d_x
            ))
            offset = c*tf.keras.backend.random_normal(shape=d_shape)
            discr_fake_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(
                    tf.clip_by_value(tf.zeros_like(d_x_hat)+offset,0,1),
                    d_x_hat
            ))
            offset = c*tf.keras.backend.random_normal(shape=d_shape)
            discr_rand_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(
                    tf.clip_by_value(tf.zeros_like(d_p)+offset,0,1),
                    d_p
            ))
            
            enc_loss = reconstruction_loss + self.beta*kl_loss
            
            dec_loss = reconstruction_loss + gen_fake_loss + gen_rand_loss + gen_real_loss
            
            discr_loss = discr_real_loss + discr_fake_loss + discr_rand_loss

            total_loss = reconstruction_loss + self.beta*kl_loss
            
        
        self.enc_vars = self.encoder.trainable_variables
        self.dec_vars = self.decoder.trainable_variables
        self.discr_vars = self.discr.trainable_variables
        
        grads = tape.gradient(enc_loss, self.enc_vars)
        self.optimizer.apply_gradients(zip(grads, self.enc_vars))
        
        grads = tape.gradient(dec_loss, self.dec_vars)
        self.optimizer.apply_gradients(zip(grads, self.dec_vars))
        
        grads = tape.gradient(discr_loss, self.discr_vars)
        self.optimizer.apply_gradients(zip(grads, self.discr_vars))

        self.enc_loss = enc_loss
        self.dec_loss = dec_loss
        self.discr_loss = discr_loss
        self.total_loss = total_loss
        self.reconstruction_loss = reconstruction_loss
        self.kl_loss = kl_loss
        
        return {
            'enc_loss': enc_loss,
            'dec_loss': dec_loss,
            'discr_loss': discr_loss,
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

if __name__ == "__main__":
    model = VAEGAN(
        input_dim=(8,64,64,1),latent_dim=(8,64,64,10),
        mykernel=5,mystrides=(1,1,1),
        num_list=[8,8],
        dis_num_list=[16,32,64],
    )
    for n,m in [
        ('encoder',model.encoder),
        ('encoder',model.decoder),
        ('encoder',model.discr),
        ]:
        print(n)
        m.summary()
        print('-----')