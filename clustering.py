import os
import sys
import tempfile
import numpy as np
import tensorflow as tf
import random
import pandas as pd
import SimpleITK as sitk
from tensorflow import keras
from skimage.transform import resize 

from data_gen import resample_img


import matplotlib.pyplot as plt
import matplotlib.cm as cm
from dipy.data import fetch_tissue_data, read_tissue_data
from dipy.segment.tissue import TissueClassifierHMRF
from openTSNE import TSNE

def read_image(myfolder):

    subject_id = os.path.basename(myfolder)
    flair_path = os.path.join(myfolder,f'{subject_id}_flair.nii.gz')
    t1_path = os.path.join(myfolder,f'{subject_id}_t1.nii.gz')
    t1ce_path = os.path.join(myfolder,f'{subject_id}_t1ce.nii.gz')
    t2_path = os.path.join(myfolder,f'{subject_id}_t2.nii.gz')

    seg_path = os.path.join(myfolder,f'{subject_id}_seg.nii.gz')

    x_list = []
    spacing=(1,1,1)
    t1, seg = None, None
    for file_path in [flair_path,t1_path,t2_path,seg_path]:

        reader= sitk.ImageFileReader()
        reader.SetFileName(file_path)
        img_obj = reader.Execute()
        img_obj = resample_img(img_obj, out_spacing=spacing, is_label=False)

        spacing = img_obj.GetSpacing()
        origin = img_obj.GetOrigin()
        size = img_obj.GetSize()
        direction = img_obj.GetDirection()

        x = sitk.GetArrayFromImage(img_obj)

        if file_path == t1_path:
            t1 = sitk.GetArrayFromImage(img_obj).copy()
        if file_path == seg_path:
            seg = sitk.GetArrayFromImage(img_obj).copy()

        if file_path in [flair_path,t1_path,t2_path]:
            mu = np.mean(x[x>0])
            sd = np.std(x[x>0])
            x = (x-mu)/(3*sd)
            x = x.clip(-1,1)
            x_list.append(x)

    img = np.array(x_list)
    img = np.moveaxis(img, 0, -1)

    return img, t1, seg

def visualize_cluster(mymodel,myfolder,workdir,batch_size=4):
    
    x, t1, tumor = read_image(myfolder)
    x = np.expand_dims(x,axis=1)

    latent_file = os.path.join(workdir,'latent.npy')
    if not os.path.exists(latent_file):
    
        print(x.shape)
        z_mean, z_log_var, latent = mymodel.encoder.predict(x,batch_size=batch_size)
        print(latent.shape)
        x_hat = mymodel.decoder.predict(latent,batch_size=batch_size)
        print(x_hat.shape)
        np.save(latent_file,latent)
    
    latent = np.load(latent_file)
    latent = latent.squeeze()

    seg_file = os.path.join(workdir,'seg.npy')
    if not os.path.exists(seg_file):
        # 
        # https://dipy.org/documentation/1.0.0./examples_built/tissue_classification
        #
        # segment CSF,WM,GM , values 1,2,3
        #
        
        nclass = 3
        beta = 0.1
        hmrf = TissueClassifierHMRF()
        initial_segmentation, final_segmentation, PVE = hmrf.classify(t1, nclass, beta)
        if False:
            fig = plt.figure()
            a = fig.add_subplot(1, 2, 1)
            img_ax = np.rot90(final_segmentation[..., 89])
            imgplot = plt.imshow(img_ax)
            a.axis('off')
            a.set_title('Axial')
            a = fig.add_subplot(1, 2, 2)
            img_cor = np.rot90(final_segmentation[:, 128, :])
            imgplot = plt.imshow(img_cor)
            a.axis('off')
            a.set_title('Coronal')
            plt.savefig('final_seg.png', bbox_inches='tight', pad_inches=0)
            plt.close()
        np.save(seg_file,final_segmentation)
    
    seg = np.load(seg_file)
    
    mask = np.zeros_like(t1)
    for x in [1,2,3]:
        mask[seg==x]=x

    # `seg` 0,1,2,3,4 - 4 classes of tumors
    # edema,
    # non-enhancing (solid) core
    # necrotic (or fluid-filled) core        
    # non-enhancing core
    offset= 3
    for x in [1,2,3,4]:
        mask[tumor==x]=offset+x
    if latent.shape[1] != mask.shape[1]:
        print(latent.shape)
        print(mask.shape)
        target_shape = list(latent.shape)
        target_shape[1]=mask.shape[1]
        target_shape[2]=mask.shape[2]
        latent = resize(latent,target_shape,order=0,mode='edge',cval=-1)

    print(latent.shape)
    print(mask.shape)

    if False:
        mylist = []
        for l in range(target_shape[-1]):
            for x in [0,1,2,3,4,5,6,7]:
                tmp = latent[:,:,:,l].squeeze()
                vals = tmp[mask==x]
                mu = np.mean(vals)
                sd = np.std(vals)
                n = len(vals)
                prct = np.percentile(vals,[5,95])
                print(f'latent ind {l}, kind {x} mean(sd) {mu:1.2f}({sd:1.2f}) {prct} n {n}')
                mydict = dict(
                    latent_dim=l,
                    kind=x,
                    val_mean=mu,
                    val_sd=sd,
                    val_05prct=prct[0],
                    val_95prct=prct[1],
                    val_n=n,
                )
                mylist.append(mydict)

        pd.DataFrame(mylist).to_csv(os.path.join(workdir,"latent.csv"),index=False)

    #
    # https://opentsne.readthedocs.io/en/latest
    #
    # dimention reduction.
    # use tsne to visualize multi-d latent variable in 2d
    # ensure no posterior collapse
    # https://www.youtube.com/watch?v=oHtqlRIsXcQ
    # https://yixinwang.github.io/papers/collapse-id-slides-public.pdf
    # 

    original_shape = latent.shape
    num = np.prod(mask.shape)
    latent_dim = 10
    X = latent.reshape((num,latent_dim))
    labels = mask.ravel()

    print(labels.shape)
    non_bkgd = np.where(labels!=0)[0]
    X = X[non_bkgd,:]
    labels = labels[non_bkgd]
    print(labels.shape)
    print(X.shape)

    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    
    idx = idx[:10000]
    
    X_subsampled = X[idx,:]
    labels_subsampled = labels[idx]
    
    print('tsne fit')
    X_embedded = TSNE(
        perplexity=30,
        metric="euclidean",
        n_jobs=8,
        random_state=42,
        verbose=True,
    ).fit(X_subsampled)

    print(X_embedded.shape)
    cmap = cm.get_cmap('PiYG', 7)
    plt.scatter(
        X_embedded[:, 0],X_embedded[:, 1],
        c=labels_subsampled,cmap=cmap,alpha=0.5)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.grid(True)
    tsne_file = os.path.join(workdir,'tsne.png')
    plt.savefig(tsne_file, bbox_inches='tight', pad_inches=0)
    plt.close()


def main(myfolder):

    from models import VAEGAN

    batch_size = 4
    input_dim=(1,240,240,3)
    latent_dim=(1,240,240,10)
    num_list=[16,32]
    dis_num_list=[16,32,64]
    mystrides=(1,1,1)
    mykernel=(1,15,15)
    
    mymodel = VAEGAN(
        input_dim=input_dim,latent_dim=latent_dim,
        num_list=num_list,dis_num_list=dis_num_list,
        mystrides=mystrides,mykernel=mykernel,
    )
    mymodel.compile(optimizer=keras.optimizers.Adam(0.01),run_eagerly=True)
    mymodel.encoder.load_weights('saved_modelsL10/enc.h5')
    mymodel.decoder.load_weights('saved_modelsL10/dec.h5')
    mymodel.discr.load_weights('saved_modelsL10/discr.h5')
    
    visualize_cluster(mymodel,myfolder,".")

if __name__ == "__main__":

    myfolder = sys.argv[1]
    
    main(myfolder)

'''

python clustering.py /mnt/hd2/data/brats2019/MICCAI_BraTS_2019_Data_Training/LGG/BraTS19_TCIA13_653_1

'''
