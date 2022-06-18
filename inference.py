import os
import sys
import numpy as np
import tensorflow as tf
import random
import pandas as pd
import SimpleITK as sitk
from tensorflow import keras
from skimage.transform import resize 

from data_gen import resample_img
from models import VAEGAN

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

def main(myfolder):
    
    batch_size = 4
    input_dim=(1,240,240,3)
    latent_dim=(1,240,240,10)
    num_list=[16,16]
    dis_num_list=[16,32,64]
    mystrides=(1,1,1)
    mykernel=(1,7,7)
    
    mymodel = VAEGAN(
        input_dim=input_dim,latent_dim=latent_dim,
        num_list=num_list,dis_num_list=dis_num_list,
        mystrides=mystrides,mykernel=mykernel,
    )
    mymodel.compile(optimizer=keras.optimizers.Adam(0.01),run_eagerly=True)
    mymodel.encoder.load_weights('saved_modelsL10/enc.h5')
    mymodel.decoder.load_weights('saved_modelsL10/dec.h5')
    mymodel.discr.load_weights('saved_modelsL10/discr.h5')    

    x, t1, seg = read_image(myfolder)
    x = np.expand_dims(x,axis=1)

    print(x.shape)
    z_mean, z_log_var, latent = mymodel.encoder.predict(x,batch_size=batch_size)
    print(latent.shape)
    x_hat = mymodel.decoder.predict(latent,batch_size=batch_size)
    print(x_hat.shape)
    
    bkgd=(t1==0).astype(np.int16)
    mask = seg.astype(np.int16)
    mask[bkgd==1]=-1

    target_shape = list(latent.shape)
    target_shape[2]=x.shape[2]
    target_shape[3]=x.shape[3]
    resized_latent = resize(latent,target_shape,order=0,mode='edge',cval=-1).squeeze()

    print(resized_latent.shape)
    print(mask.shape)
    mylist = []
    for l in range(target_shape[-1]):
        for x in [-1,0,1,2,3,4]:
            tmp = resized_latent[:,:,:,l].squeeze()
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

    pd.DataFrame(mylist).to_csv("latent.csv",index=False)

    #
    # TODO: 
    # + segment CSF,WM,GM
    # + dimension reduction, plot classification as color overlay.
    #
    
if __name__ == "__main__":

    myfolder = sys.argv[1]
    main(myfolder)

'''

python inference.py /mnt/hd2/data/brats2019/MICCAI_BraTS_2019_Data_Training/LGG/BraTS19_TCIA13_653_1

'''