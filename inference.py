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

import hdbscan

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

    return img, t1, seg, t1_path

def segment_via_clustering(mymodel,myfolder,workdir,batch_size=4):
    
    x, t1, tumor, t1_path = read_image(myfolder)
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

    print(latent.shape)
    print(t1.shape)
    target_shape = list(latent.shape)
    target_shape[1]=t1.shape[1]
    target_shape[2]=t1.shape[2]
    latent = resize(latent,target_shape,order=0,mode='edge',cval=-1)

    original_shape = latent.shape
    num = np.prod(t1.shape)
    latent_dim = 10
    X = latent.reshape((num,latent_dim))
    non_bkgd = np.where(t1!=0)[0]
    X = X[non_bkgd,:]
    
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    
    idx = idx[:1000]
    X_subsampled = X[idx,:]
    # random_state=0
    clusterer = hdbscan.HDBSCAN(min_cluster_size=15, prediction_data=True).fit(X_subsampled)
    
    X = latent.reshape((num,latent_dim))
    pred, strengths = hdbscan.approximate_predict(clusterer, X)    
    pred+=1
    print(np.unique(pred))
    
    pred = np.reshape(pred,t1.shape)
    pred[t1==0]=0
    pred = pred.astype(np.int16)

    reader= sitk.ImageFileReader()
    reader.SetFileName(t1_path)
    img_obj = reader.Execute()

    mask_obj = sitk.GetImageFromArray(pred)
    mask_obj.SetSpacing(img_obj.GetSpacing())
    mask_obj.SetOrigin(img_obj.GetOrigin())
    mask_obj.SetDirection(img_obj.GetDirection())

    writer = sitk.ImageFileWriter()    
    writer.SetFileName("prediction.nii.gz")
    writer.SetUseCompression(True)
    writer.Execute(mask_obj)



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
    
    segment_via_clustering(mymodel,myfolder,".")

if __name__ == "__main__":

    myfolder = sys.argv[1]
    
    main(myfolder)

'''

python inference.py /mnt/hd2/data/brats2019/MICCAI_BraTS_2019_Data_Training/LGG/BraTS19_TCIA13_653_1

'''
