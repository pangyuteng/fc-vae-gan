import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import os, sys
import numpy as np
import tensorflow as tf
from tsne import Parametric_tSNE
import h5py 

# configure Tensorflow
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
session_conf.gpu_options.allow_growth=True

# intialize TF
tf.reset_default_graph()
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
init = tf.global_variables_initializer()
sess.run(init)

# import stuff.
from fcvaegan import (
    Model,
    MODEL_DIR,
    INPUT_TENSOR_NAME,
    SIGNATURE_NAME,
    read_and_decode,
    PARAMS,
    W,H,C,CY,
    TRAIN_DIR,
    NUM_EXAMPLES_TRAIN,
    NUM_EXAMPLES_VALIDATION,
    NUM_EXAMPLES_TEST,
)
params = dict(PARAMS)
params.update({
    'is_training':True,
    'batch_size':4,
})


print('training tsne...')
tsne_high_dims = params['latent_dims'][-1]
tsne_num_outputs = 2
tsne_perplexity = 5
tsne_dropout = 0.3
epochs = 5
do_pretrain = False
batch_size = 1024
path = '/media/external/scisoft/fc-vae-gan/data/latent.h5'


for tsne_perplexity in [5,10,15,30]:

    tsne_weight_file = os.path.join(MODEL_DIR,'tsne_{}.hdf5'.format(tsne_perplexity))

    tsne = Parametric_tSNE(
        tsne_high_dims,
        tsne_num_outputs,
        tsne_perplexity, 
        batch_size=batch_size,
        dropout=tsne_dropout,
        do_pretrain=do_pretrain)

    #if True:
    if not os.path.exists(tsne_weight_file):
        print('training tsne')
        with h5py.File(path, "r") as f:
            print(f['latent'].shape)
            print(f['label'].shape)
            tsne.fit(f['latent'],verbose=1,epochs=epochs,)
            tsne.save_model(tsne_weight_file)
            print('done training tsne...')
    else:
        print('loading pretrained tsne')
        tsne.restore_model(tsne_weight_file)

    with h5py.File(path, "r") as f:
        skip = 1
        z = f['latent'][::skip]
        l = f['label'][::skip]
        count,bins=np.histogram(l.ravel(),bins=500,range=(0,500))
        print(count)
        out = tsne.transform(z)
        skip = 100
        plt.scatter(out[::skip,0],out[::skip,1],c=l[::skip].squeeze())
        plt.grid(False)
        plt.savefig('result_latent_2d_{}{:02d}.png'.format(do_pretrain,tsne_perplexity))