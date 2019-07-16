import os, sys
from fcvaegan import (
    Model,
    MODEL_DIR,
    INPUT_TENSOR_NAME,
    SIGNATURE_NAME,
)

import numpy as np
import tensorflow as tf

import data    
data_module = data.pascal

read_and_decode = data_module.read_and_decode
TRAIN_DIR = data_module.TF_DATA_DIR
W,H,C,CY=(data_module.w,data_module.h,data_module.c,data_module.cy)
num_samples = data_module.NUM_EXAMPLES_TRAIN

training = False
epochs = 80000
batch_size = 4
params = {
    'learning_rate': 1e-6,
    'latent_dims':[10,10,10],
    'data_dims': [W,H,C],
    'is_training':False,
    'batch_size':batch_size,
    'warmup_until':1000000,
    'g_scale_factor':0.2,
    'd_scale_factor':0.2,
    'recon_const':0.0,
    'latent_factor':0.5,
    'perceptual_factor':0.25,
    'stride':[2,2,2],
}


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

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
vae = None
if training:
    tf_images,tf_labels = train_input_fn(batch_size=batch_size)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord,sess=sess)

    tsne_high_dims = params['latent_dims'][-1]
    tsne_num_outputs = 2
    tsne_perplexity = 5
    tsne_dropout = 0.3
    tsne_weight_file = os.path.join(MODEL_DIR,'tsne.hdf5')

    x = tf.placeholder(name='x', dtype=tf.float32, shape=[None, W,H,C])
    vae = Model(**params)
    vae._build(x)
    vae.restore(tf.train.latest_checkpoint(MODEL_DIR))

    # TODO: might need to save to disk as ...hdf5
    zlist=[]
    llist=[]
    #for r in range(steps_per_epoch):
    for r in range(3):
        img, lbl = sess.run([tf_images, tf_labels])
        x = img[INPUT_TENSOR_NAME]
        z = vae.transformer(x)
        zshape = np.array(z.shape)
        newshape = [np.prod(zshape[:-1]),params['latent_dims'][-1]]
        z = np.reshape(z,newshape)
        zlist.append(z)
        llist.append(lbl)

    zlist=np.array(zlist)
    llist=np.array(llist)
    z = np.concatenate(zlist,axis=0)
    l = np.concatenate(llist,axis=0)

    print(z.shape,l.shape,'^^^^^^^^^^^^^^^^^')

    print('training tsne...')
    tsne = Parametric_tSNE(tsne_high_dims, tsne_num_outputs, tsne_perplexity, dropout=tsne_dropout)
    tsne.fit(z,verbose=1,epochs=5,)
    tsne.save_model(tsne_weight_file)
    print('done training tsne...')

    coord.request_stop()
    # Wait for threads to stop
    coord.join(threads)


x = tf.placeholder(name='x', dtype=tf.float32, shape=[None, W,H,C])
if vae is None:
    vae = Model(**params)
    vae._build(x)
    vae.restore(tf.train.latest_checkpoint(MODEL_DIR))

batch_size = 4
#tf_images,tf_labels = train_input_fn(batch_size=batch_size)
tf_images,tf_labels = eval_input_fn(batch_size=batch_size)

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from skimage import exposure

n = np.sqrt(batch_size).astype(np.int32)
h = H
w = W
I_reconstructed = np.empty((h*n, 2*w*n, 3))

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord,sess=sess)

img, lbl = sess.run([tf_images, tf_labels])
x = img[INPUT_TENSOR_NAME]
x_tilde = vae.reconstructor(x)

for i in range(n):
    for j in range(n):
        tmp = np.concatenate(
            (x_tilde[i*n+j, :].reshape(h, w, C), 
             x[i*n+j, :].reshape(h, w, C)),
            axis=1
        )
        I_reconstructed[i*h:(i+1)*h, j*2*w:(j+1)*2*w, :] = tmp

plt.figure(figsize=(20, 20))
plt.imshow(I_reconstructed, cmap='gray')
plt.grid(False)
plt.savefig('result_compare.png')
print('done.')

'''

if vae.l_dim > 2:
    n = 10
    x = np.linspace(-2, 2, n)
    for latent_dim in range(params['n_z']):
        I_latent = np.empty((h*n, w))
        for i, xi in enumerate(x):
            sd = 1.
            r = np.random.normal(0.,sd,(params['n_z'],))
            r[latent_dim]=xi
            z = np.array([r])
            z1 = np.random.normal(0.,sd,(1,int(w/2),int(h/2),params['n_z_1']))
            z2 = np.random.normal(0.,sd,(1,int(w/4),int(h/4),params['n_z_2']))
            z3 = np.random.normal(0.,sd,(1,int(w/8),int(w/8),params['n_z_3']))
            #x_hat = vae.generator(z,z1,z2,z3)
            x_hat = vae.generator_only_z(z)

            I_latent[i*w:(i+1)*w,:] = x_hat[0].reshape(w, h)

        plt.figure(figsize=(8, 8))        
        plt.imshow(I_latent, cmap="gray")
        plt.grid(None)
        plt.savefig('result_latent_generate{}.png'.format(latent_dim))
else:

    n = 20
    x = np.linspace(-2, 2, n)
    y = np.linspace(-2, 2, n)

    I_latent = np.empty((h*n, w*n))
    for i, yi in enumerate(x):
        for j, xi in enumerate(y):
            z = np.array([[xi, yi]])
            x_hat = vae.generator(z)

            I_latent[(n-i-1)*W:(n-i)*W, j*H:(j+1)*H] = x_hat[0].reshape(w, h)

    plt.figure(figsize=(8, 8))        
    plt.imshow(I_latent, cmap="gray")
    plt.grid(None)
    plt.savefig('result_latent_generate.png')

# Test the trained model: transformation

zlist=[]
llist=[]
for r in range(50):
    img, lbl = sess.run([tf_images, tf_labels])
    x = img[INPUT_TENSOR_NAME]
    zlist.append(vae.transformer(x))
    llist.append(lbl)

zlist=np.array(zlist)
llist=np.array(llist)
z = np.concatenate(zlist,axis=0)
l = np.concatenate(llist,axis=0)


# save embedding
from tensorflow.contrib.tensorboard.plugins import projector

tf.reset_default_graph()
sess = tf.Session()
config = projector.ProjectorConfig()

LOG_DIR = 'emb'
summary_writer = tf.summary.FileWriter(LOG_DIR, graph=None)
path_for_metadata = os.path.join(LOG_DIR,'metadata.tsv',)
with open(path_for_metadata,'w') as f:
    f.write("Index\tLabel\n")
    for index,label in enumerate(l):
        f.write("%d\t%s\n" % (index,label))

embedding_var = tf.Variable(z, name='z',trainable=False)
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name
embedding.metadata_path = os.path.basename(path_for_metadata)

projector.visualize_embeddings(summary_writer, config)

saver = tf.train.Saver(max_to_keep=1)
sess.run(tf.global_variables_initializer())
saver.save(sess, os.path.join(LOG_DIR, 'z.ckpt'))

if vae.l_dim != 2:
    if training_success:
        print("training done.")
    print("done.")
    sys.exit(0)


print(z.shape)
plt.figure(figsize=(10, 8)) 
plt.scatter(z[:, 0], z[:, 1], c=l)
plt.colorbar()
plt.grid()
plt.savefig('result_latent_embedding.png')


# Stop the threads
coord.request_stop()
# Wait for threads to stop
coord.join(threads)       

if training_success:
    print("training done.")

'''