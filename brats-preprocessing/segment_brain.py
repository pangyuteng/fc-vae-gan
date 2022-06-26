import os
import sys
import numpy as np
import SimpleITK as sitk

from dipy.data import fetch_tissue_data, read_tissue_data
from dipy.segment.tissue import TissueClassifierHMRF


def segment(myfolder):
    
    subject_id = os.path.basename(myfolder)
    flair_path = os.path.join(myfolder,f'{subject_id}_flair.nii.gz')
    t1_path = os.path.join(myfolder,f'{subject_id}_t1.nii.gz')
    t1ce_path = os.path.join(myfolder,f'{subject_id}_t1ce.nii.gz')
    t2_path = os.path.join(myfolder,f'{subject_id}_t2.nii.gz')
    tumor_path = os.path.join(myfolder,f'{subject_id}_seg.nii.gz')

    seg_path = os.path.join(myfolder,f'{subject_id}_sawtelle.nii.gz')

    reader= sitk.ImageFileReader()
    reader.SetFileName(t1_path)
    img_obj = reader.Execute()
    spacing = img_obj.GetSpacing()
    origin = img_obj.GetOrigin()
    size = img_obj.GetSize()
    direction = img_obj.GetDirection()

    t1 = sitk.GetArrayFromImage(img_obj)

    reader.SetFileName(tumor_path)
    tumor_obj = reader.Execute()
    tumor = sitk.GetArrayFromImage(tumor_obj)
    # 
    # https://dipy.org/documentation/1.0.0./examples_built/tissue_classification
    #
    # segment CSF,GM,WM values 1,2,3 ?
    #
    
    nclass = 3
    beta = 0.1
    hmrf = TissueClassifierHMRF()
    initial_segmentation, final_segmentation, PVE = hmrf.classify(t1, nclass, beta)
    
    mask = np.zeros_like(t1)
    for x in [1,2,3]:
        mask[final_segmentation==x]=x+5

    # `seg` 0,1,2,3,4 - 4 classes of tumors
    # edema,
    # non-enhancing (solid) core
    # necrotic (or fluid-filled) core        
    # non-enhancing core
    offset= 3
    for x in [1,2,3,4]:
        mask[tumor==x]=x

    # 0 bkgd, 1-4, tumour, 5,6,7 - csf/wm/gm
    mask_obj = sitk.GetImageFromArray(mask)
    mask_obj.SetSpacing(img_obj.GetSpacing())
    mask_obj.SetOrigin(img_obj.GetOrigin())
    mask_obj.SetDirection(img_obj.GetDirection())

    writer = sitk.ImageFileWriter()    
    writer.SetFileName(seg_path)
    writer.SetUseCompression(True)
    writer.Execute(mask_obj)


if __name__ == "__main__":

    myfolder = sys.argv[1]
    
    segment(myfolder)

'''

python segment_brain.py /mnt/hd2/data/brats2019/MICCAI_BraTS_2019_Data_Training/LGG/BraTS19_TCIA13_653_1

'''
