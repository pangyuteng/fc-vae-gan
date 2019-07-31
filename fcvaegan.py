'''
    https://stackoverflow.com/questions/49634488/keras-variational-autoencoder-nan-loss
'''
import traceback
import sys, os
import shutil
import numpy as np
import tensorflow as tf

epsilon = 1e-10
infinite = 1e10
SEED = 42
tf.random.set_random_seed(SEED)
tf.logging.set_verbosity(tf.logging.INFO)

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
session_conf.gpu_options.allow_growth=True

from tensorflow.python import debug as tf_debug
from tensorflow.python.estimator.model_fn import ModeKeys as Modes
from tensorflow.contrib import slim
from tensorflow.python.ops import array_ops

INPUT_TENSOR_NAME = 'inputs'
SIGNATURE_NAME = 'predictions'
MODEL_DIR = 'model_fcvaegan'

from tensorflow.contrib import slim
conv2d = tf.contrib.layers.convolution2d


from abstract_network import (
conv2d_bn_lrelu,
conv2d_t_bn_lrelu,
fc_bn_lrelu,
)


class Model(object):

    def __init__(self, 
        learning_rate=None,
        data_dims=None,
        latent_dims=None,
        is_training=None,
        batch_size=None,
        **params,
        ):     
        self.is_training = is_training
        self.learning_rate = learning_rate
        self.data_dims = data_dims
        self.latent_dims  = latent_dims
        self.batch_size  = batch_size
        self.params = params

        self.cs = [32,64,128,] # num_outputs
        self.ks = [5,5] # kernel_size
        self.stride = params['stride']

    def _build(self,input_layer):
        
        # input layer
        self.x = input_layer

        # stride of 2
        # 64 <- 128
        # 32 <- 64
        # 16 <- 32
        #     NA           flatten
        kwargs=dict(is_training=self.is_training)
        # inference
        conv0 = self._encoder(self.x,0,name='ENC',**kwargs)
        conv1 = self._encoder(conv0,1,name='ENC',**kwargs)
        conv2 = self._encoder(conv1,2,name='ENC',**kwargs)
        

        # variational inference
        self.mu0,self.sd0 = self._latent(conv2,2,name='ENC',**kwargs) 
        
        # latent variables
        self.sd0+=epsilon
        self.noise0 = tf.random_normal(shape=tf.shape(self.mu0),mean=0.0,stddev=1.0)
        self.z = self.mu0 + tf.multiply(self.sd0,self.noise0)
        

        self._printout(self.z,'latent shape (self.z)')

        self.zp = self.z+tf.random_normal(shape=tf.shape(self.z), mean=0, stddev=0.5, dtype=tf.float32,seed=SEED)

        # generator
        convt0 = self._decoder(self.z,2,name='DEC',**kwargs)
        convt1 = self._decoder(convt0,1,name='DEC',**kwargs) 
        self.x_hat = self._decoder(convt1,0,name='DEC',**kwargs)
        self.x_hat = tf.clip_by_value(self.x_hat,0.0,1.0)

        convt0 = self._decoder(self.zp,2,name='DEC',reuse=True,**kwargs)
        convt1 = self._decoder(convt0,1,name='DEC',reuse=True,**kwargs) 
        self.x_p = self._decoder(convt1,0,name='DEC',reuse=True,**kwargs)
        self.x_p = tf.clip_by_value(self.x_p,0.0,1.0)

        # discriminator
        discr_conv0 = self._encoder(self.x,0,reuse=False,name='DISCR',**kwargs)
        discr_conv1 = self._encoder(discr_conv0,1,reuse=False,name='DISCR',**kwargs)
        discr_conv2 = self._encoder(discr_conv1,2,reuse=False,name='DISCR',**kwargs)
        self.style, self.d = self._flatter(discr_conv2,1,reuse=False,name='DISCR',**kwargs)
        self.d = tf.clip_by_value(self.d,0.0,1.0)
        self.style = tf.clip_by_value(self.style,infinite*-1,infinite)

        discr_conv0 = self._encoder(self.x_hat,0,reuse=True,name='DISCR',**kwargs)
        discr_conv1 = self._encoder(discr_conv0,1,reuse=True,name='DISCR',**kwargs)
        discr_conv2 = self._encoder(discr_conv1,2,reuse=True,name='DISCR',**kwargs)
        self.style_hat, self.d_hat = self._flatter(discr_conv2,1,reuse=True,name='DISCR',**kwargs)
        self.d_hat = tf.clip_by_value(self.d_hat,0.0,1.0)
        self.style_hat = tf.clip_by_value(self.style_hat,infinite*-1,infinite)

        discr_conv0 = self._encoder(self.x_p,0,reuse=True,name='DISCR',**kwargs)
        discr_conv1 = self._encoder(discr_conv0,1,reuse=True,name='DISCR',**kwargs)
        discr_conv2 = self._encoder(discr_conv1,2,reuse=True,name='DISCR',**kwargs)
        self.style_p, self.d_p = self._flatter(discr_conv2,1,reuse=True,name='DISCR',**kwargs)
        self.d_p = tf.clip_by_value(self.d_p,0.0,1.0)
        self.style_p = tf.clip_by_value(self.style_p,infinite*-1,infinite)

    def _printout(self,tensor=None,text=None):

        if tensor is not None:
            shape = tensor.get_shape().as_list()
        else:
            shape = None

        print(shape,text,'**********************')

    def _flatter(self,in_conv,out_num,activation_fn=tf.sigmoid,name=None,reuse=False,is_training=True):
        scope_text = '{}_flatter'.format(name)
        with tf.variable_scope(scope_text) as scope:
            if reuse:
                scope.reuse_variables()
            
            self._printout(in_conv,'input '+scope_text)

            conv = self._encoder(in_conv,2,name=scope_text,reuse=reuse,is_training=is_training)
            _shape = conv.get_shape().as_list()
            flat = slim.flatten(conv)
            flat = slim.fully_connected(flat,512,activation_fn=tf.nn.relu)
            flat = slim.fully_connected(flat,32,activation_fn=tf.nn.relu)
            out = slim.fully_connected(flat, out_num,activation_fn=activation_fn)
            
            self._printout(out,'out '+scope_text)
            return in_conv, out

    def _encoder(self,x,level,name=None,reuse=False,is_training=True):
        scope_text = '{}_encoder{}'.format(name,level,)
        with tf.variable_scope(scope_text) as scope:
            if reuse:
                scope.reuse_variables()
                
            conv1 = conv2d_bn_lrelu(
                 x, self.cs[level], self.ks, self.stride[level], is_training)
            conv2 = conv2d_bn_lrelu(
                 conv1, self.cs[level], self.ks, 1, is_training)

            self._printout(conv2,scope_text)
            return conv2

    def _latent(self,x,level,name=None,reuse=False,is_training=True):
        scope_text = '{}_latent{}'.format(name,level)
        with tf.variable_scope(scope_text) as scope:
            if reuse:
                scope.reuse_variables()
            
            conv1 = conv2d_bn_lrelu(
                 x, self.cs[level], self.ks, 1, is_training)
            
            conv2 = conv2d_bn_lrelu(
                conv1, self.cs[level], self.ks, 1, is_training)
                            
            mu = slim.convolution2d(
                conv2, self.latent_dims[level], [1,1],
                stride=1, padding='SAME',activation_fn=tf.identity)

            sd = conv2d(
                conv2, self.latent_dims[level], [1,1], 1,
                activation_fn=tf.sigmoid,
                weights_initializer=tf.zeros_initializer(),
                weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5))

            self._printout(conv2,scope_text)
            
            return mu, sd            
            
    def _decoder(self, lad, level,c=None,name=None,reuse=False,is_training=True):
        scope_text = '{}_decoder{}'.format(name,level)
        with tf.variable_scope(scope_text) as scope:
            if reuse:
                scope.reuse_variables()

            x = tf.concat(values=[lad], axis=-1)
                
            conv1 = conv2d_t_bn_lrelu(
                 x, self.cs[level], self.ks, self.stride[level], is_training)
            
            if level == 0:
                if c is None:
                    c =self.data_dims[2] 
                else:
                    pass
                # final output layer.
                conv2 = tf.contrib.layers.convolution2d_transpose(
                    conv1, c, [3, 3], 1,
                    activation_fn=tf.sigmoid)
            else:
                conv2 = conv2d_t_bn_lrelu(
                    conv1, self.cs[level], self.ks, 1, is_training)
            
            self._printout(conv2,scope_text)
            return conv2
        
    def NLLNormal(self, pred, target):
        #ref. https://github.com/andersbll/autoencoding_beyond_pixels/blob/master/model/ae.py
        #ref. https://github.com/pangyuteng/vae-gan-tensorflow/blob/master/vaegan.py
        c = -0.5 * tf.log(2 * np.pi)
        multiplier = 1.0 / (2.0 * 1)
        tmp = tf.verify_tensor_all_finite(tf.square(pred - target),'wtf')
        tmp *= -multiplier
        tmp += c
        return tmp

    def _setup_loss(self,global_step):

        def _like(liketype,tensor,scale_factor,random=True):
            if liketype == 'ones':
                if random:
                    return tf.random_normal(shape=tensor.get_shape().as_list(), mean=1.-scale_factor, stddev=0.05, dtype=tf.float32,seed=SEED)
                else:
                    return tf.ones_like(tensor) - scale_factor
            elif liketype == 'zeros':
                if random:
                    return tf.random_normal(shape=tensor.get_shape().as_list(), mean=0.+scale_factor, stddev=0.05, dtype=tf.float32,seed=SEED)
                else:
                    return tf.zeros_like(tensor) + scale_factor
            else:
                raise NotImplementedError()

        self._printout(tensor=None,text='setup loss started...')
        # Loss

        # Reconstruction loss
        #self.recon_loss = tf.reduce_sum(tf.reduce_mean(tf.abs(self.x-self.x_hat)))
        # Loss
        # Reconstruction loss
        # Minimize the cross-entropy loss
        # H(x, x_hat) = -\Sigma x*log(x_hat) + (1-x)*log(1-x_hat)
        recon_loss = -tf.reduce_sum(
            self.x * tf.log(epsilon+self.x_hat) + (1-self.x) * tf.log(epsilon+1-self.x_hat), 
            axis=1
        )
        self.recon_loss = tf.reduce_mean(recon_loss)
        tf.verify_tensor_all_finite(self.recon_loss, "recon_loss not finite!")

        self.z_loss = tf.reduce_mean(-0.5 * tf.reduce_sum(1 + self.z - tf.square(self.mu0) - tf.exp(self.sd0), axis=1))
        
        tf.verify_tensor_all_finite(self.z_loss, "z_loss not finite!")

        current_step = tf.to_float(global_step)

        warmup_until = self.params['warmup_until']
        
        self.warmup =  1 - tf.exp(-1.*current_step/warmup_until)
        self.warmup *= 0.5

        self.recon_const = 1/np.prod(self.data_dims)
        self.recon_const *= self.params['recon_const']

        self.z_const = 1/np.prod(self.z.get_shape().as_list()[1:])
        self.z_const *= self.params['latent_factor']
        
        # perceptual loss
        self.perceptual_loss = tf.reduce_mean(tf.reduce_sum(self.NLLNormal(self.style_hat, self.style), [1,2,3]))
        self.perceptual_const = 1/np.prod(self.style.get_shape().as_list()[1:])
        self.perceptual_const *= self.params['perceptual_factor']
        
        # ENCODER LOSS = recon + latent + loc
        self.e_loss = 0
        self.e_loss += self.recon_loss*self.recon_const
        self.e_loss += self.z_loss*self.warmup*self.z_const
        self.e_loss += -self.perceptual_loss*self.perceptual_const

        tf.verify_tensor_all_finite(self.e_loss, "e_loss not finite!")

        self.g_scale_factor = self.params['g_scale_factor']
        self.d_scale_factor = self.params['d_scale_factor']
        
        # G loss # https://stackoverflow.com/questions/40158633/how-to-solve-nan-loss
        self.g_gen_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=_like('ones',self.d_p,self.g_scale_factor), logits=tf.clip_by_value(self.d_p,epsilon,1.0)))
        
        self.g_fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=_like('ones',self.d_hat,self.g_scale_factor), logits=tf.clip_by_value(self.d_hat,epsilon,1.0)))
                
        self.g_real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=_like('zeros',self.d,self.g_scale_factor), logits=tf.clip_by_value(self.d,epsilon,1.0)))

        self.g_loss = 0 
        self.g_loss += self.recon_loss*self.recon_const 
        self.g_loss += self.g_gen_loss 
        self.g_loss += self.g_fake_loss 
        self.g_loss += self.g_real_loss 
        self.g_loss += -self.perceptual_loss*self.perceptual_const
        tf.verify_tensor_all_finite(self.g_loss, "g_loss not finite!")

        # D loss
        self.d_fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=_like('zeros',self.d_p,self.d_scale_factor), logits=tf.clip_by_value(self.d_p,epsilon,1.0)))
        
        self.d_hat_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=_like('zeros',self.d_hat,self.d_scale_factor), logits=tf.clip_by_value(self.d_hat,epsilon,1.0)))

        self.d_real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=_like('ones',self.d,self.d_scale_factor), logits=tf.clip_by_value(self.d,epsilon,1.0)))
        

        self.d_loss = self.d_fake_loss + self.d_hat_loss + self.d_real_loss
        tf.verify_tensor_all_finite(self.d_loss, "d_loss not finite!")
        
        self.t_vars = tf.trainable_variables()

        self.e_vars = [var for var in self.t_vars if 'ENC' in var.name]
        self.g_vars = [var for var in self.t_vars if 'DEC' in var.name]
        self.d_vars = [var for var in self.t_vars if 'DISCR' in var.name]

        self.e_lr = tf.train.exponential_decay(self.learning_rate,
            global_step=global_step, decay_steps=10000, decay_rate=0.98)
        self.g_lr = tf.train.exponential_decay(self.learning_rate,
            global_step=global_step, decay_steps=10000, decay_rate=0.98)
        self.d_lr = tf.train.exponential_decay(self.learning_rate,
            global_step=global_step, decay_steps=10000, decay_rate=0.98)
        
        # NAN loss. clipping, norm, nan checks...
        # https://towardsdatascience.com/debugging-a-machine-learning-model-written-in-tensorflow-and-keras-f514008ce736
        # https://stackoverflow.com/questions/36498127/how-to-apply-gradient-clipping-in-tensorflow
        def get_op(loss,lr,var_list,global_step):
            opt = tf.train.RMSPropOptimizer(learning_rate=lr,epsilon=epsilon)
            grads, vars = zip(*opt.compute_gradients(loss,var_list=var_list))
            grads, norm = tf.clip_by_global_norm(grads, 1.0)
            train_op = opt.apply_gradients(zip(grads,vars),global_step=global_step)
            #_grads,norm = tf.clip_by_global_norm(grads, 5.0)
            #norm = 5.0
            #capped_gvs = [(tf.clip_by_value(g, -1.*norm, norm), v) for g,v in zip(grads,vars)]
            #train_op = opt.apply_gradients(capped_gvs,global_step=global_step)
            return train_op

        # encoder
        #self.e_op = tf.train.RMSPropOptimizer(learning_rate=self.e_lr,epsilon=epsilon)
        #self.e_op = tf.contrib.estimator.clip_gradients_by_norm(self.e_op, clip_norm=5.0)
        #self.e_op = self.e_op.minimize(self.e_loss,global_step=global_step,var_list=self.e_vars)
        self.e_op = get_op(self.e_loss,self.e_lr,self.e_vars,global_step)

        # generator
        # self.g_op = tf.train.RMSPropOptimizer(learning_rate=self.g_lr,epsilon=epsilon)
        # self.g_op = tf.contrib.estimator.clip_gradients_by_norm(self.g_op, clip_norm=5.0)
        # self.g_op = self.g_op.minimize(self.g_loss,global_step=global_step,var_list=self.g_vars)
        self.g_op = get_op(self.g_loss,self.g_lr,self.g_vars,global_step)

        # discriminator
        # self.d_op = tf.train.RMSPropOptimizer(learning_rate=self.d_lr,epsilon=epsilon)
        # self.d_op = tf.contrib.estimator.clip_gradients_by_norm(self.d_op, clip_norm=5.0)
        # self.d_op = self.d_op.minimize(self.d_loss,global_step=global_step,var_list=self.d_vars)
        self.d_op = get_op(self.d_loss,self.d_lr,self.d_vars,global_step)

        self.train_op = tf.group(self.e_op,self.g_op,self.d_op,)
        
        for var in tf.trainable_variables():
            tf.summary.histogram(var.name.replace(':','_'), var)
        merged_summary = tf.summary.merge_all()
        
        self._printout(tensor=None,text='setup loss done.')
        return True
    
    def restore(self,fpath):
        self._printout(tensor=None,text='restoring... ')

        if getattr(self,'sess',None) is None:      
            self.sess = tf.InteractiveSession()
            self.sess.run(tf.global_variables_initializer())

        if getattr(self,'saver',None) is None:
            self.saver = tf.train.Saver()
            
        self.saver.restore(self.sess, fpath)

        self._printout(tensor=None,text='restore... ')

    # x -> x_hat
    def reconstructor(self, x):
        x_hat = self.sess.run(self.x_hat, feed_dict={self.x: x})
        return x_hat

    # x -> z
    def transformer(self, x):
        z = self.sess.run(self.z, feed_dict={self.x: x})
        return z
    
    # z -> x
    def generator(self,zp):
        x_p = self.sess.run(self.x_p, feed_dict={self.zp:zp})
        return x_p

def model_fn(features, labels, mode, params):

    if mode in (Modes.PREDICT, Modes.EVAL):
        pass
    if mode in (Modes.TRAIN):
        pass

    # this script takes learning_rate as a hyperparameter
    sz = [-1]+params.get("data_dims")
    print('@@@@@@@@@@',sz)
    # Input Layer
    input_layer = tf.reshape(features[INPUT_TENSOR_NAME], sz,name='input_layer')
    print(input_layer.get_shape().as_list())
    print('reshape is okay')
    vae = Model(**params)
    vae._build(input_layer)
    
    global_step = tf.train.get_or_create_global_step()
    vae._setup_loss(global_step)
    
    output_layer = vae.x_hat
    # Define operations
    if mode in (Modes.PREDICT, Modes.EVAL):
        probabilities = output_layer
        predictions = tf.reshape(output_layer, [-1])

    if mode in (Modes.TRAIN, Modes.EVAL):

        for var in tf.trainable_variables():
            tf.summary.histogram(var.name.replace(':','_'), var)
        merged_summary = tf.summary.merge_all()
        
        tf.summary.image("x", vae.x)
        tf.summary.image("x_hat", vae.x_hat)
        tf.summary.image("x_p", vae.x_p)

        tf.summary.scalar('d_loss', vae.d_loss)
        tf.summary.scalar('g_loss', vae.g_loss)
        tf.summary.scalar('e_loss', vae.e_loss)
        tf.summary.scalar('perceptual_loss', vae.perceptual_loss) 
        tf.summary.scalar('recon_loss', vae.recon_loss)
        tf.summary.scalar('z_loss', vae.z_loss)
        tf.summary.scalar('warmup', vae.warmup)

        
    if mode == Modes.PREDICT:
        predictions = {
            'probabilities': probabilities
        }
        export_outputs = {
            SIGNATURE_NAME: tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(
            mode, predictions=predictions, export_outputs=export_outputs)

    if mode == Modes.TRAIN:
        
        print('************ TRAINING ************')

        train_op = vae.train_op
        loss = vae.recon_loss
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


    if mode == Modes.EVAL:
        expected_output = input_layer
        loss = vae.recon_loss
        eval_metric_ops = {
            'mean_squared_error': tf.metrics.mean_squared_error(expected_output, output_layer)
        }
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=eval_metric_ops)

import data    
data_module = data.pascal

read_and_decode = data_module.read_and_decode
TRAIN_DIR = data_module.TF_DATA_DIR
W,H,C,CY=(data_module.w,data_module.h,data_module.c,data_module.cy)
NUM_EXAMPLES_TRAIN,NUM_EXAMPLES_VALIDATION,NUM_EXAMPLES_TEST = (
    data_module.NUM_EXAMPLES_TRAIN,
    data_module.NUM_EXAMPLES_VALIDATION,
    data_module.NUM_EXAMPLES_TEST,
)


PARAMS = {
    'learning_rate': 1e-6,
    'latent_dims':[10,10,10],
    'data_dims': [W,H,C],
    'is_training':True,
    'batch_size':4,
    'warmup_until':1000000,
    'g_scale_factor':0.2,
    'd_scale_factor':0.2,
    'recon_const':0.0,
    'latent_factor':0.5,
    'perceptual_factor':0.25,
    'stride':[2,2,2],
}
epochs = 80000

# using main, so tf_debug can be used.
def main(training,warm_start,batch_size,debug):
    params = dict(PARAMS)
    params.update({
        'training':training,
        'batch_size':batch_size,
    })
    
    def train_input_fn(training_dir=TRAIN_DIR, batch_size=batch_size, params=None):
        return _input_fn(training_dir, 'train.tfrecords', batch_size=batch_size)

    def eval_input_fn(training_dir=TRAIN_DIR, batch_size=batch_size, params=None):
        return _input_fn(training_dir, 'validation.tfrecords', batch_size=batch_size)

    def _input_fn(training_dir, training_filename, batch_size=batch_size):
        test_file = os.path.join(training_dir, training_filename)
        filename_queue = tf.train.string_input_producer([test_file])

        image, label = read_and_decode(filename_queue)
        
        images, labels = tf.train.batch(
            [image, label], batch_size=batch_size,
            capacity=1000 + 10 * batch_size)
        return {INPUT_TENSOR_NAME: images}, labels
    
    tf.reset_default_graph()
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    init = tf.global_variables_initializer()
    sess.run(init)

    steps_per_epoch = int(num_samples / batch_size)
    print(steps_per_epoch)

    total_steps = round(float(steps_per_epoch) * epochs)
    print(total_steps)
    
    training_success=False
    if training:

        if os.path.exists(MODEL_DIR) and warm_start is False:
            shutil.rmtree(MODEL_DIR)

        checkpointing_config = tf.estimator.RunConfig(
            save_checkpoints_secs = 2*60,  # Save checkpoints every 2 minutes.
            keep_checkpoint_max = 10,       # Retain the 10 most recent checkpoints.
        )
        
        if warm_start:
            ws = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=MODEL_DIR)
        else:
            ws = None

        estimator = tf.estimator.Estimator(
            model_fn,
            model_dir=MODEL_DIR,
            config=checkpointing_config,
            params=params,
            warm_start_from=ws)

        hooks = []
        if debug:
            hooks.append(tf_debug.LocalCLIDebugHook())

        try:  
            
            # setup train spec
            train_spec = tf.estimator.TrainSpec(
                input_fn=train_input_fn,
                max_steps=total_steps)

            # setup eval spec evaluating ever n seconds
            eval_every_n_secs = 60*60 
            steps = 100
            eval_spec = tf.estimator.EvalSpec(
                input_fn=eval_input_fn,
                steps=steps,
                throttle_secs=eval_every_n_secs)
            # run train and evaluatev
            #output = tf.estimator.train_and_evaluate(estimator,train_spec, eval_spec)
            output = estimator.train(
                train_input_fn,steps=total_steps,
                hooks=hooks,
            )
            print(output)
            training_success=True
        except:
            traceback.print_exc()

        evaluation = estimator.evaluate(input_fn=train_input_fn,steps=500)
        print("Training Set:")
        print("Loss: %s" % evaluation["loss"])
        print("mse: %f" % evaluation["mean_squared_error"])        

        evaluation = estimator.evaluate(input_fn=eval_input_fn,steps=500)
        print("Validation Set:")
        print("Loss: %s" % evaluation["loss"])
        print("mse: %f" % evaluation["mean_squared_error"])

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--training',type=str,choices=['False','True'],default='True')
    parser.add_argument('-w','--warm_start',type=str,choices=['False','True'],default='False')
    parser.add_argument('-b','--batch_size',type=int,default=4)
    parser.add_argument('-d','--debug',type=str,choices=['False','True'],default='False')
    args = parser.parse_args()
    training = eval(args.training)
    warm_start = eval(args.warm_start)
    debug = eval(args.debug)
    batch_size = args.batch_size

    main(training,warm_start,batch_size,debug,)
