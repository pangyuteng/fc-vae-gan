
import os
import sys
import time
import zlib
import logging
logger = logging.getLogger('data_gen')
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

import numpy as np
import tensorflow as tf
import random
import pandas as pd

from tensorflow.keras.utils import Sequence

import SimpleITK as sitk
from skimage.transform import resize 
from scipy import ndimage as ndi

import albumentations as A

def seed_everything(seed=4269):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  # new flag present in tf 2.0+
    np.random.seed(seed)
    tf.random.set_seed(seed)
seed_everything()

    
# https://gist.github.com/mrajchl/ccbd5ed12eb68e0c1afc5da116af614a
def resample_img(itk_image, out_spacing=[2.0, 2.0, 2.0], is_label=False):
    
    # Resample images to 2mm spacing with SimpleITK
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)

def read_image(row): # responsible for reading, resampling, scaling intensity to (-1,1)
    
    file_path = row.file_path
    if row.dataset == 'ped-ct-seg':
        
        spacing=(2.0,2.0,2.0)

        reader= sitk.ImageFileReader()
        reader.SetFileName(file_path)
        img_obj = reader.Execute()   
        img_obj = resample_img(img_obj, out_spacing=spacing, is_label=False)

        spacing = img_obj.GetSpacing()
        origin = img_obj.GetOrigin()
        size = img_obj.GetSize()
        direction = img_obj.GetDirection()

        img = sitk.GetArrayFromImage(img_obj)

        logger.debug(f'{origin},{direction}')
        logger.debug(f'{spacing},{size}')
        logger.debug(f'img.shape {img.shape}')

        MIN_VAL,MAX_VAL = -1000,1000
        img = img.astype(np.float16)
        img = ((img-MIN_VAL)/(MAX_VAL-MIN_VAL))
        img = (img-0.5)*2
        img = img.clip(-1,1)

        img = np.expand_dims(img,axis=-1)

    elif row.dataset == 'brats19':

        subject_id = os.path.basename(row.file_path)
        flair_path = os.path.join(row.file_path,f'{subject_id}_flair.nii.gz')
        t1_path = os.path.join(row.file_path,f'{subject_id}_t1.nii.gz')
        t1ce_path = os.path.join(row.file_path,f'{subject_id}_t1ce.nii.gz')
        t2_path = os.path.join(row.file_path,f'{subject_id}_t2.nii.gz')

        x_list = []
        spacing=(1,1,1)
        for file_path in [flair_path,t1_path,t2_path]:

            reader= sitk.ImageFileReader()
            reader.SetFileName(file_path)
            img_obj = reader.Execute()   
            img_obj = resample_img(img_obj, out_spacing=spacing, is_label=False)

            spacing = img_obj.GetSpacing()
            origin = img_obj.GetOrigin()
            size = img_obj.GetSize()
            direction = img_obj.GetDirection()

            x = sitk.GetArrayFromImage(img_obj)

            logger.debug(f'{origin},{direction}')
            logger.debug(f'{spacing},{size}')
            logger.debug(f'x.shape {x.shape}')

            mu = np.mean(x[x>0])
            sd = np.std(x[x>0])
            x = (x-mu)/(3*sd)
            x = x.clip(-1,1)
            x_list.append(x)

        img = np.array(x_list)
        img = np.moveaxis(img, 0, -1)

    else:
        raise NotImplementedError()

    return img

MIN_VAL = -1
aug_pipeline = A.Compose([
    A.ShiftScaleRotate(value=MIN_VAL,border_mode=0),
])

FILL_VAL = 0
cutout_aug_pipeline = A.Compose([
    A.Cutout(p=0.5, num_holes=1,
        max_h_size=120, max_w_size=120, fill_value=FILL_VAL),
])

def augment_2d(img,min_val):
    
    img = img.squeeze()

    assert(min_val==MIN_VAL)

    augmented = aug_pipeline(
        image=img,
    )
    img = augmented['image']

    cut_augmented = cutout_aug_pipeline(
        image=img,
    )
    aug_img = cut_augmented['image']

    img = np.expand_dims(img,axis=0)
    aug_img = np.expand_dims(aug_img,axis=0)

    return img,aug_img

def augment_3d(img,min_val):
    
    mydim = [6,8,8] # random rectangle cutouts
    np.random.shuffle(mydim)

    tmp = np.expand_dims(np.random.rand(*mydim),axis=-1)
    cutout = (tmp>0.9).astype(np.float) # cut out 10% of spaces.
    cutout = resize(cutout>0,img.shape,order=0,mode='edge',cval=min_val)

    aug_img = img.copy() # copy!!!
    aug_img[cutout==1] = min_val

    return img,aug_img

def augment(img,min_val):

    if img.shape[0]>1: # leverage albumentation if 1st dim == 1
        return augment_3d(img,min_val)
    else:
        return augment_2d(img,min_val)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class DataGenerator(Sequence):
    def __init__(self,df,batch_size=8,shuffle=False,augment=False,output_shape=(32,128,128,1)):
        
        self.df = df.copy().reset_index()        
        self.indices = np.arange(len(self.df))

        self.min_val = -1
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.output_shape = output_shape
        

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def dataread(self, row):

        img = read_image(row)
        if self.output_shape:            
            # orignal image shape
            i0,i1,i2,_ = img.shape

            # target image shape
            o0,o1,o2,_ = self.output_shape

            # we pad some values 
            diff = np.array([i0-o0,i1-o1,i2-o2,0])
            if any(diff<0):
                padding = [(0,0) if x>=0 else (np.abs(x),np.abs(x)) for x in diff]
                img = np.pad(img,padding,'constant',constant_values=(self.min_val,self.min_val))
            
            i0,i1,i2,_ = img.shape

            # starting coordinate
            if i0-o0 == 0:
                s0 = 0
            else:
                s0 = random.choice(list(range(i0-o0))) 
            if i1-o1 == 0:
                s1 = 0
            else:
                s1 = random.choice(list(range(i1-o1)))
            if i2-o2 == 0:
                s2 = 0
            else:
                s2 = random.choice(list(range(i2-o2)))

            img = img[s0:s0+o0,s1:s1+o1,s2:s2+o2,:]

        if self.augment:
            img, aug_img = augment(img,self.min_val)
        else:
            aug_img = img.copy()
        
        logger.debug(f'{img.shape},{aug_img.shape}')

        return img,aug_img

    def __len__(self):
        return int(np.floor(len(self.indices) / float(self.batch_size)))

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_rows = self.df.iloc[inds,:]

        x_arr = []
        cutout_x_arr = []
        for n,row in batch_rows.iterrows():
            img, cutout_img = self.dataread(row)
            x_arr.append(img)
            cutout_x_arr.append(cutout_img)
            
        return np.array(cutout_x_arr), np.array(x_arr)

if __name__ == "__main__":
    logging.basicConfig( level="DEBUG" )
    
    df = pd.read_csv(sys.argv[1])
    mygen = DataGenerator(
        df,
        batch_size=8,output_shape=(1,240,240,4),
        shuffle=True,augment=True,
    )
    mygen.on_epoch_end()
    print(len(mygen))
    for n,(x,y) in zip(range(2),mygen):
        print(n,x.shape,y.shape)

'''
python data_gen.py ped-ct-seg.csv
'''
