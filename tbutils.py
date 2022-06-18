
# ref 
# https://keras.io/guides/customizing_what_happens_in_fit
# https://gist.github.com/joelthchao/ef6caa586b647c3c032a4f84d52e3a11
# https://stackoverflow.com/questions/43784921/how-to-display-custom-images-in-tensorboard-using-keras
# https://www.tensorflow.org/api_docs/python/tf/summary/image
# https://www.tensorflow.org/tensorboard/migrate
# https://raw.githubusercontent.com/pangyuteng/cycle-gan-apes/main/tbutils.py

import os
import sys
import random
import numpy as np
import tensorflow as tf
import json
import datetime
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow.keras.backend as K

class Warmup(keras.callbacks.Callback):
    # 1 / (e^-x +1)
    scale = 10 # use to determine max value.
    slope = 0.5 # slope, higher means steeper.
    shift = 10 # shift so that all values > 0.
    def on_epoch_begin(self, epoch, logs=None):
        K.set_value(
            self.model.beta,
            1/(tf.exp(self.shift-epoch*self.slope)+1)*self.scale
        )
        print("epoch {} beta {:1.4f}".format(epoch,K.get_value(self.model.beta)))
        
class ImageSummaryCallback(tf.keras.callbacks.Callback):
    def __init__(self, data_gen, logdir):
        super(ImageSummaryCallback, self).__init__()
        self.logdir = logdir
        self.file_writer = tf.summary.create_file_writer(self.logdir)
        self.count = 0
        self.epoch = 0
        self.data_gen = data_gen

    def on_batch_end(self, batch, logs=None):
        self.on_end("batch", batch)
    def on_epoch_end(self, epoch, logs=None):        
        self.epoch = epoch
        self.on_end("epoch", epoch)

    def on_end(self, kind, batch, logs=None):        
        os.makedirs('images', exist_ok=True)
        # self.data_gen.on_epoch_end()

        for n,(cutout_x, x) in zip(range(1),self.data_gen):

            z_mean, z_log_var, z = self.model.encoder(cutout_x)
            x_hat = self.model.decoder(z)
            merged_imgs = np.concatenate([x, cutout_x, x_hat],axis=-1)
            break

        if merged_imgs.shape[-1] == 3: # 1channels
            i,j,k = 0,1,2
        elif merged_imgs.shape[-1] == 12: # 4channels
            i,j,k = 0,4,8
        elif merged_imgs.shape[-1] == 9:  # 3channels
            i,j,k = 0,3,6
        else:
            print(merged_imgs.shape,'!!!!!!!!!!!!!!!!')
            raise NotImplementedError()
        # Rescale images 0 - 1
        merged_imgs = 0.5 * merged_imgs + 0.5
        
        s = int(merged_imgs.shape[1]/2.0)
        row0 = np.concatenate([
            merged_imgs[0,s,:,:,i],
            merged_imgs[0,s,:,:,j],
            merged_imgs[0,s,:,:,k],
        ],axis=1)

        merged_img = (255*row0).astype(np.uint8)
        merged_img = np.expand_dims(merged_img,axis=0)
        merged_img = np.expand_dims(merged_img,axis=-1)

        fig, ax = plt.subplots(1,1)
        plt.imshow(merged_img.squeeze(),cmap='gray',vmin=0,vmax=255)
        ax.axis('off')
        if kind == 'epoch':
            fname = f"images/{batch:04d}_end.png"
        else:
            fname = f"images/{self.epoch:04d}_{batch:04d}.png"
        fig.savefig(fname)
        plt.close()


        with self.file_writer.as_default():
            tf.summary.image("sample", merged_img, step=self.count)
            self.file_writer.flush()
            self.count+=1

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
class ModelSaveCallback(tf.keras.callbacks.Callback):
    def __init__(self,):
        super(ModelSaveCallback, self).__init__()
    def on_batch_end(self, epoch, logs=None):
        os.makedirs(os.path.join(THIS_DIR,'saved_models'),exist_ok=True)
        encoder_weight_file = os.path.join(THIS_DIR,'saved_models','enc.h5')
        decoder_weight_file = os.path.join(THIS_DIR,'saved_models','dec.h5')
        discr_weight_file = os.path.join(THIS_DIR,'saved_models','discr.h5')
        self.model.encoder.save_weights(encoder_weight_file)
        self.model.decoder.save_weights(decoder_weight_file)
        self.model.discr.save_weights(discr_weight_file)

class MetricSummaryCallback(tf.keras.callbacks.Callback):
    def __init__(self, logdir):
        super(MetricSummaryCallback, self).__init__()
        self.logdir = logdir
        self.file_writer = tf.summary.create_file_writer(self.logdir)
        self.count = 0
        self.tstamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    def on_batch_end(self, batch, logs=None):
        self.on_end("batch", batch)

    def on_epoch_end(self, epoch, logs=None):
        self.on_end("epoch", epoch)

    def on_end(self, kind, epoch, logs=None):

        mydict = dict(
            enc_loss = float(np.around(self.model.enc_loss.numpy(),5)),
            dec_loss = float(np.around(self.model.dec_loss.numpy(),5)),
            discr_loss = float(np.around(self.model.discr_loss.numpy(),5)),
            total_loss = float(np.around(self.model.total_loss.numpy(),5)),
            reconstruction_loss = float(np.around(self.model.reconstruction_loss.numpy(),5)),
            kl_loss = float(np.around(self.model.kl_loss.numpy(),5)),
            kl_beta = float(np.around(self.model.beta.numpy(),5)),
            gamma = float(np.around(self.model.gamma.numpy(),5)),
        )

        with self.file_writer.as_default():
            for name, value in mydict.items():
                tf.summary.scalar(name, value, step=self.count)
                self.file_writer.flush()

        mydict['count']=self.count
        if kind == "epoch":
            mydict['epoch']=epoch

        with open(os.path.join(self.logdir,f"metrics-{self.tstamp}.json"),'a+') as f:
            f.write(json.dumps(mydict)+"\n")
        self.count+=1