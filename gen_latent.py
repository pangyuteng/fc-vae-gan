import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import os, sys
import numpy as np
import tensorflow as tf
from tsne import Parametric_tSNE

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
    'is_training':True, # this impact inferences....due to batch norm...?
    #'batch_size':4, # <<--- 4 is good for 6gb gpu.
    'batch_size':1,
})

batch_size = params['batch_size']

def _input_fn(training_dir, training_filename, batch_size=batch_size):
    test_file = os.path.join(training_dir, training_filename)
    filename_queue = tf.train.string_input_producer([test_file])

    image, label = read_and_decode(filename_queue)
    images, labels = tf.train.batch(
        [image, label], batch_size=batch_size,
        capacity=1000 + 10 * batch_size)
    return {INPUT_TENSOR_NAME: images}, labels


#def eval_input_fn(training_dir=TRAIN_DIR, batch_size=batch_size, params=None):
#    return _input_fn(training_dir, 'validation.tfrecords', batch_size=batch_size)
#tf_images,tf_labels = eval_input_fn(batch_size=batch_size)

def train_input_fn(training_dir=TRAIN_DIR, batch_size=batch_size, params=None):
    return _input_fn(training_dir, 'train.tfrecords', batch_size=batch_size)
tf_images,tf_labels = train_input_fn(batch_size=batch_size)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord,sess=sess)    

# test out the input data stream.
img, lbl = sess.run([tf_images,tf_labels])
batch = [img[INPUT_TENSOR_NAME],lbl]
print(batch[0].shape,batch[1].shape)
print(batch[1][0].shape)

# load model
X = tf.placeholder(name='x', dtype=tf.float32, shape=[None, W,H,C])
model = Model(**params)
model._build(X)
model.restore(tf.train.latest_checkpoint(MODEL_DIR))

# initialize array for saving images.
n = np.sqrt(batch_size).astype(np.int32)
h = H
w = W
I_hat = np.empty((h*n, 2*w*n,3))
I_z = np.empty((h*n, 2*w*n,3))

# inference
img, lbl = sess.run([tf_images, tf_labels])
x = img[INPUT_TENSOR_NAME]
x_hat,z,x_p = model.sess.run([model.x_hat,model.z,model.x_p], feed_dict={model.x: x,})


# print stuff...
print(x.shape,np.max(x),np.min(x))
print(x_hat.shape,np.max(x_hat),np.min(x_hat))
print(x_p.shape,np.max(x_p),np.min(x_p))
print(z.shape,x_p.shape)




print(n)
for i in range(n):
    for j in range(n):
        tmp = np.concatenate(
            (x_hat[i*n+j, :].reshape(h, w,3), 
             x[i*n+j, :].reshape(h, w,3)),
            axis=1
        )
        I_hat[i*h:(i+1)*h, j*2*w:(j+1)*2*w,:] = tmp
szx,szy = 60,30
plt.figure(0,figsize=(szx,szy))
plt.imshow(I_hat, cmap='gray')
plt.grid(False)
plt.savefig('result_compare_hat.png')

for i in range(n):
    for j in range(n):
        tmp = np.concatenate(
            (x_p[i*n+j, :].reshape(h, w,3), 
             x[i*n+j, :].reshape(h, w,3)),
            axis=1
        )
        I_z[i*h:(i+1)*h, j*2*w:(j+1)*2*w,:] = tmp

plt.figure(1,figsize=(szx,szy))
plt.imshow(I_z, cmap='gray')
plt.grid(False)
plt.savefig('result_compare_z.png')

print('done.')

import yaml
with open('/media/external/scisoft/fc-vae-gan/data/label.yml','r') as f:
    label_dict = yaml.load(f.read())
num_of_interest = []
for k,v in label_dict.items():
    #if v in ['car','person','bicycle']:
    if v in ['aeroplane','bird','sky']:
        num_of_interest.append(k)
num_of_interest = set(num_of_interest)


print("OK")

import h5py
path = '/media/external/scisoft/fc-vae-gan/data/latent.h5'
if True:#not os.path.exists(path):
    os.remove(path)
    img_count = 0
    with h5py.File(path, "a") as f:
        for ind in range(NUM_EXAMPLES_TRAIN,):
            #print('index',ind)

            img, lbl = sess.run([tf_images, tf_labels])
            x = img[INPUT_TENSOR_NAME]

            intersect = num_of_interest.intersection(set(list(lbl.ravel())))
            
            if len(list(intersect))<2:
            #if len(list(intersect))==0:
                #print(ind,NUM_EXAMPLES_TRAIN,'skipped')
                continue
               
            print(ind,NUM_EXAMPLES_TRAIN,'processing')
            x_hat,z,x_p = model.sess.run([model.x_hat,model.z,model.x_p], feed_dict={model.x: x,})
            zshape = np.array(z.shape)

            skip = int(W/zshape[1])
            lbl = lbl[:,::skip,::skip,:]
            lshape = np.array(lbl.shape)
            latent_dim = params['latent_dims'][-1]
            newshape = [np.prod(zshape[:-1]),latent_dim]
            z = np.reshape(z,newshape)
            label_dim = 1
            newshape = [np.prod(lshape[:-1]),label_dim]
            lbl = np.reshape(lbl,newshape)

            if img_count == 0:
                zset = f.create_dataset('latent',
                    (newshape[0],latent_dim), maxshape=(None,latent_dim),dtype=np.float, chunks=(10**4,latent_dim))
                lset = f.create_dataset('label',
                    (newshape[0],label_dim,), maxshape=(None,label_dim),dtype=np.int, chunks=(10**4,label_dim))
                zset[:] = z
                lset = lbl
            else:
                zset = f['latent']
                lset = f['label']

                zset.resize(zset.shape[0]+newshape[0], axis=0)
                lset.resize(lset.shape[0]+newshape[0], axis=0)

                zset[-newshape[0]:,:]=z
                lset[-newshape[0]:,:]=lbl
        
            img_count+=1
            #if img_count > 5:
            #    break
    print(img_count)
    
with h5py.File(path, "r") as f:
    print(f['latent'].shape)
    print(f['label'].shape)
























# TODO traverse through latent space in one single dimention at one point?
