
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

def read_image(row,spacing=(2.0,2.0,2.0)): # responsible for reading, resampling, scaling intensity to (-1,1)
    
    file_path = row.file_path
    assert(row.dataset == 'ped-ct-seg')

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

    return img


#A.GridDistortion(p=0.5, num_steps=5),
A.RandomBrightnessContrast(),
aug_pipeline = A.Compose([
    A.ShiftScaleRotate(),
])
cutout_aug_pipeline = A.Compose([
    A.Cutout(p=0.5, num_holes=8, max_h_size=32, max_w_size=32, fill_value=-1),
])

def augment(img,min_val):

    cutout = (np.random.rand(8,8,8)>0.9).astype(np.float) # cut out 10% of spaces.
    cutout = resize(cutout,img.shape,order=0)
    
    img[cutout==1] = min_val

    return img

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
            i0,i1,i2 = img.shape

            # target image shape
            o0,o1,o2,_ = self.output_shape

            # we pad some values 
            diff = np.array([i0-o0,i1-o1,i2-o2])
            if any(diff<0):
                padding = [(0,0) if x>=0 else (np.abs(x),np.abs(x)) for x in diff]
                img = np.pad(img,padding,'constant',constant_values=(self.min_val,self.min_val))
            
            i0,i1,i2 = img.shape

            # starting coordinate
            s0 = random.choice(list(range(i0-o0))) 
            s1 = random.choice(list(range(i1-o1)))
            s2 = random.choice(list(range(i2-o2)))

            img = img[s0:s0+o0,s1:s1+o1,s2:s2+o2]

        if self.augment and np.random.rand()>0.5:
            aug_img = augment(img,self.min_val)
        else:
            aug_img = img
            
        img = np.expand_dims(img,axis=-1)
        aug_img = np.expand_dims(aug_img,axis=-1)
        
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

    
    df = pd.read_csv(sys.argv[1])
    mygen = DataGenerator(
        df,
        batch_size=8,output_shape=(32,128,128,1),
        shuffle=True,augment=True,
    )
    mygen.on_epoch_end()
    print(len(mygen))
    for n,(x,y) in zip(range(2),mygen):
        print(n,x.shape,y.shape)

'''
python data_gen.py ped-ct-seg.csv
'''