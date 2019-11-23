import matplotlib as mpl
mpl.use('Agg')
# coding: utf-8

# In[1]:


import sys
import os
import traceback

import numpy as np
import tensorflow as tf
from skimage import exposure

import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
session_conf.gpu_options.allow_growth=True


# In[3]:


tf.reset_default_graph()
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
init = tf.global_variables_initializer()
sess.run(init)


# In[4]:


import data
print(dir(data))
data_module = data.pascal


# In[5]:


read_and_decode = data_module.read_and_decode
TRAIN_DIR = data_module.TF_DATA_DIR
W,H,C=(data_module.w,data_module.h,data_module.c)
num_samples = data_module.NUM_EXAMPLES_TRAIN
batch_size = 4
INPUT_TENSOR_NAME = 'inputs'

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


images, labels = eval_input_fn()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord,sess=sess)    

img, lbl = sess.run([images, labels])
batch = [img[INPUT_TENSOR_NAME],lbl]
print(batch[0].shape,batch[1].shape)
print(batch[1][0].shape)


# In[6]:


batch_size


# In[7]:


img, lbl = sess.run([images, labels])
for n in range(batch_size):
    plt.figure(n)
    plt.subplot(121)
    plt.imshow(img['inputs'][n,:].squeeze(),cmap='gray')
    plt.subplot(122)
    plt.imshow(lbl[n,:].squeeze(),cmap='gray')
plt.close()

# In[8]:


# Stop the threads
#coord.request_stop()
# Wait for threads to stop
#coord.join(threads)


# In[9]:


from fcvaegan import Model, MODEL_DIR, PARAMS
params = dict(PARAMS)
params['training'] = True #<---seems to do the trick. for x_hat


# In[10]:


MODEL_DIR


# In[11]:


#tf.reset_default_graph()
#sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
#init = tf.global_variables_initializer()
#sess.run(init)


# In[12]:


x = tf.placeholder(name='x', dtype=tf.float32, shape=[None, W,H,C])
l = tf.placeholder(name='loc', dtype=tf.float32, shape=[None, 1])
model = Model(**params)
model._build(x)
model.restore(tf.train.latest_checkpoint(MODEL_DIR))


# In[13]:


n = np.sqrt(batch_size).astype(np.int32)
h = H
w = W
I_hat = np.empty((h*n, 2*w*n,3))
I_z = np.empty((h*n, 2*w*n,3))
#tf_images,tf_labels = eval_input_fn(batch_size=batch_size)
#coord = tf.train.Coordinator()
#threads = tf.train.start_queue_runners(coord=coord,sess=sess)


# In[14]:


#img, lbl = sess.run([tf_images, tf_labels])
img, lbl = sess.run([images, labels])
x = img[INPUT_TENSOR_NAME]
x_hat,z,x_p = model.sess.run([model.x_hat,model.z,model.x_p], feed_dict={model.x: x,})


# In[15]:


print(x.shape,np.max(x),np.min(x))
print(x_hat.shape,np.max(x_hat),np.min(x_hat))
print(x_p.shape,np.max(x_p),np.min(x_p))
print(z.shape,x_p.shape)


# In[16]:


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
plt.close()
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
plt.close()
print('done.')


# In[17]:


zc = np.copy(z)
print(zc.shape)
zc+=np.random.normal(loc=0.0, scale=0.001, size=z.shape)
for q in range(10):
    plt.figure(q)
    plt.imshow(zc[0,:,:,q].squeeze(),cmap='gray')
plt.close()

# In[18]:


for q,scale in enumerate([0,0.1,0.3,0.6,5,100]):
    zc = np.copy(z)
    noise=np.random.normal(loc=0.0, scale=scale, size=z.shape)
    mask=np.zeros(z.shape)
    mask[:,8:10,8:10,:]=1
    zc+=mask*noise
    #zc+=noise
    x_p = model.sess.run(model.x_p, feed_dict={model.zp:zc})
    for i in range(n):
        for j in range(n):
            tmp = np.concatenate(
                (x_hat[i*n+j, :].reshape(h, w,3), 
                 x[i*n+j, :].reshape(h, w,3)),
                axis=1
            )
            I_hat[i*h:(i+1)*h, j*2*w:(j+1)*2*w,:] = tmp
    szx,szy = 60,30
    plt.figure(q,figsize=(szx,szy))
    plt.subplot(121)
    plt.imshow(I_hat, cmap='gray')
    plt.grid(False)
    plt.savefig('result_compare_hat.png')
    plt.close()
    for i in range(n):
        for j in range(n):
            tmp = np.concatenate(
                (x_p[i*n+j, :].reshape(h, w,3), 
                 x_hat[i*n+j, :].reshape(h, w,3)),
                axis=1
            )
            I_z[i*h:(i+1)*h, j*2*w:(j+1)*2*w,:] = tmp

    plt.subplot(122)
    plt.imshow(I_z, cmap='gray')
    plt.grid(False)
    plt.savefig('result_compare_z.png')
    plt.close()
    print('done.')


# In[19]:


num_samples/batch_size


# In[20]:


# TODO: might need to save to disk as ...hdf5
zlist=[]
llist=[]
#for r in range(steps_per_epoch):
for r in range(100):
    img, lbl = sess.run([images, labels])
    x = img[INPUT_TENSOR_NAME]
    x_hat,z,x_p = model.sess.run([model.x_hat,model.z,model.x_p], feed_dict={model.x: x,})
    zshape = np.array(z.shape)
    
    skip = int(W/zshape[1])
    lbl = lbl[:,::skip,::skip,:]
    lshape = np.array(lbl.shape)
    
    newshape = [np.prod(zshape[:-1]),params['latent_dims'][-1]]
    z = np.reshape(z,newshape)

    newshape = [np.prod(lshape[:-1]),1]
    lbl = np.reshape(lbl,newshape)

    zlist.append(z)
    llist.append(lbl)

zlist=np.array(zlist)
llist=np.array(llist)
z = np.concatenate(zlist,axis=0)
l = np.concatenate(llist,axis=0)

print(z.shape,l.shape,'^^^^^^^^^^^^^^^^^')
print(len(np.unique(l)))


# In[21]:


from tsne import Parametric_tSNE


# In[22]:


print('training tsne...')
tsne_weight_file = os.path.join(MODEL_DIR,'tsne.hdf5')
tsne_high_dims = params['latent_dims'][-1]
tsne_num_outputs = 2
tsne_perplexity = 30
tsne_dropout=0.5

tsne = Parametric_tSNE(tsne_high_dims, tsne_num_outputs, tsne_perplexity, dropout=tsne_dropout)
tsne.fit(z,verbose=1,epochs=5,)
tsne.save_model(tsne_weight_file)
print('done training tsne...')


# In[23]:


l.shape,z.shape


# In[24]:


out = tsne.transform(z)


# In[25]:

plt.close()

skip = 1000
plt.figure(100)
plt.scatter(out[::skip,0],out[::skip,1],c=l[::skip].squeeze())
plt.savefig('tsne.png',)
plt.close()

# In[26]:




coord.request_stop()
## Wait for threads to stop
coord.join(threads)

