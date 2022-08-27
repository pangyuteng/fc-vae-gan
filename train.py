# source https://github.com/pangyuteng/hello-mnist/blob/4ac2511a0829a0e525085c12ee490c94250481bd/selfsuper/hello-vae/8-fc-conv-vae-gan/vaegan.py

import os,sys
import pandas as pd

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K

from data_gen import seed_everything, DataGenerator
seed_everything()

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from models import VAEGAN
from tbutils import Warmup,ImageSummaryCallback, MetricSummaryCallback, ModelSaveCallback

if __name__ == '__main__':

    csv_file = sys.argv[1]

    if csv_file == 'ped-ct-seg.csv':
        input_dim=(8,64,64,1)
        latent_dim=(8,16,16,10)
        num_list=[64,64]
        dis_num_list=[16,32,64]
        mystrides=(1,2,2)
        mykernel=5
        init_lr=1e-4
        
    elif csv_file == 'brats19.csv':
        input_dim=(1,240,240,3)
        latent_dim=(1,60,60,32)
        num_list=[16,32]
        dis_num_list=[16,32,64]
        mystrides=(1,2,2)
        mykernel=(1,15,15)
        init_lr=1e-5

    epochs = 100000
    batch_size = 64
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=init_lr,
        decay_steps=1000,
        decay_rate=0.96
    )

    df = pd.read_csv(csv_file)
    mygen = DataGenerator(df,output_shape=input_dim,shuffle=True,augment=True,batch_size=batch_size)
    valgen = DataGenerator(df,output_shape=input_dim,shuffle=True,augment=True,batch_size=1)
    
    run_eagerly = True
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))
    with strategy.scope():
        mymodel = VAEGAN(
            input_dim=input_dim,latent_dim=latent_dim,
            num_list=num_list,dis_num_list=dis_num_list,
            mystrides=mystrides,mykernel=mykernel,
        )
        mymodel.compile(optimizer=keras.optimizers.Adam(lr_schedule),run_eagerly=run_eagerly)

    # logging
    log_dir = './log'
    tensorboard_cb = keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False,
        update_freq='batch', profile_batch=2, embeddings_freq=0)
    model_checkpoint_cb = ModelSaveCallback()
    metric_cb = MetricSummaryCallback(log_dir)    
    image_summary_callback = ImageSummaryCallback(valgen,log_dir)
    callback_list = [Warmup(),tensorboard_cb,model_checkpoint_cb,metric_cb,image_summary_callback]
    
    history = mymodel.fit(
        mygen, 
        epochs=epochs, 
        workers=10,
        callbacks=callback_list,
    )



"""

tensorboard --bind_all --logdir=log

python train.py brats19.csv

"""
