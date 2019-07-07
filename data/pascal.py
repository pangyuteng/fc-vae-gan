import subprocess
import numpy as np
import os
import hashlib
import PIL
from PIL import Image
from scipy.io import loadmat
import yaml 

import sys
import os
import warnings
import tensorflow as tf
import numpy as np
import scipy
from sklearn.model_selection import train_test_split

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TF_DATA_DIR = os.path.join(THIS_DIR,'tf_data')
train_filename = os.path.join(TF_DATA_DIR,'train.tfrecords')
validation_filename = os.path.join(TF_DATA_DIR,'validation.tfrecords')
test_filename = os.path.join(TF_DATA_DIR,'test.tfrecords')
expected_file_list = [train_filename,validation_filename,test_filename]

w = h = 256
c = 3
cy = 1

NUM_EXAMPLES_PATH = os.path.join(THIS_DIR,'num_examples.yml')

def set_num_examples(data_sets):
    num_examples_dict = {
        'train':data_sets.train.num_examples,
        'validation':data_sets.validation.num_examples,
        'test':data_sets.test.num_examples,
    }
    with open(NUM_EXAMPLES_PATH,'w') as f:
        f.write(yaml.dump(num_examples_dict,default_flow_style=False))

def load_num_examples():
    with open(NUM_EXAMPLES_PATH,'r') as f:
        num_examples_dict = yaml.load(f.read())
    return num_examples_dict['train'],num_examples_dict['validation'], num_examples_dict['test']

if os.path.exists(NUM_EXAMPLES_PATH):
    NUM_EXAMPLES_TRAIN,NUM_EXAMPLES_VALIDATION,NUM_EXAMPLES_TEST = load_num_examples()


class Dataset(object):
    def __init__(self,image_path_list,label_path_list):
        self.images=image_path_list
        self.labels=label_path_list
        self.num_examples=len(image_path_list)
        self.image_shape=[w,h,c]
        self.label_shape=[w,h,cy]
        
    def read_image(self,image_path):
        img = Image.open(image_path)
        img = img.resize((w,h), resample=PIL.Image.BICUBIC)
        img = np.array(img).astype(np.uint8)
        return img
    
    def read_label(self,label_path):
        label = loadmat(label_path)
        array = label['LabelMap']
        array = array.astype(np.int32)
        img = Image.fromarray(array, mode='I')
        img = img.resize((w,h), resample=PIL.Image.NEAREST)
        img = np.array(img).astype(np.uint16)
        img = np.expand_dims(img,axis=-1)
        return img

class Pascal(object):
    def __init__(self,image_folder_path,label_folder_path,label_txt_path):
        self.image_folder_path = image_folder_path
        self.label_folder_path = label_folder_path
        self.label_txt_path = label_txt_path
        self.data_yml_path = os.path.join(THIS_DIR,'data.yml')
        self.label_yml_path = os.path.join(THIS_DIR,'label.yml')
        self.prepare()
        self.load()
        
    def load(self):
        
        with open(self.data_yml_path,'r') as f:
            self.df = yaml.load(f.read())
            
        X_comb, X_test, y_comb, y_test = train_test_split(
            self.df['data_list'],self.df['label_list'],test_size=0.33,random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(
            X_comb,y_comb,test_size=0.33,random_state=42)

        self.train = Dataset(X_train,y_train)
        self.validation = Dataset(X_val,y_val)
        self.test = Dataset(X_test,y_test)
    
    def prepare(self):
        
        if os.path.exists(self.data_yml_path):
            return
        
        _ = self.get_label_dict()
        
        data = {
            'data_list':[],
            'label_list':[],
        }
        
        img_list = [x.split('.')[0] for x in os.listdir(self.image_folder_path)]
        label_list = [x.split('.')[0] for x in os.listdir(self.label_folder_path)]
        
        intersection_list = sorted(list(set(img_list).intersection(set(label_list))))
        intersection_list = intersection_list[:1000]
        c=0
        for n,basename in enumerate(intersection_list):
            img_basename = basename+'.jpg'
            label_basename = basename+'.mat'
            img_path = os.path.join(self.image_folder_path,img_basename)
            label_path = os.path.join(self.label_folder_path,label_basename)
            data['data_list'].append(img_path)
            data['label_list'].append(label_path)
            c+=1
    
        with open(self.data_yml_path,'w') as f:
            f.write(yaml.dump(data))

    def get_label_dict(self):
        
        if os.path.exists(self.label_yml_path):
            with open(self.label_yml_path,'r') as f:
                return yaml.load(f.read())
        
        def to_dict(x):
            num,name = x.split(':')
            num = int(num)
            name = name.strip(' ')
            return {num:name}
        
        label_dict={}
        with open(self.label_txt_path,'r') as f:
            label_txt=f.read()
            for x in label_txt.split('\n'):
                if len(x) < 1:
                    continue
                label_dict.update(to_dict(x))
                
        with open(self.label_yml_path,'w') as f:
            f.write(yaml.dump(label_dict))
            
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(data_set, name, directory):
    """Converts a dataset to tfrecords."""
    images = data_set.images
    labels = data_set.labels
    num_examples = data_set.num_examples

    rows,cols,depth = data_set.image_shape
    _,_,depth_y = data_set.label_shape

    if len(images) != num_examples:
        raise ValueError('Images size %d does not match label size %d.' %
                         (images.shape[0], num_examples))

    filename = os.path.join(directory, name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        img = data_set.read_image(images[index])
        label = data_set.read_label(labels[index])
        
        if index%100 == 0:
            _img = Image.fromarray(img.squeeze().astype(np.uint8),mode='RGB')
            _label = Image.fromarray(label.squeeze().astype(np.int32),mode='I')
            preview=os.path.join(THIS_DIR,'preview')
            if not os.path.exists(preview):
                os.makedirs(preview)
            _img.save(os.path.join(preview,str(index)+'_img.png'))
            _label.save(os.path.join(preview,str(index)+'_label.png'))

        image_raw = img.tostring()
        label_raw = label.tostring()
        example = tf.train.Example(
            features=tf.train.Features(feature={
                'height': _int64_feature(rows),
                'width': _int64_feature(cols),
                'depth': _int64_feature(depth),
                'depth_y': _int64_feature(depth_y),
                'label_raw': _bytes_feature(label_raw),
                'image_raw': _bytes_feature(image_raw),
            }))
        writer.write(example.SerializeToString())
    writer.close()
    
INPUT_TENSOR_NAME = 'inputs'
def serving_input_fn(params): # for deployment
    inputs = {INPUT_TENSOR_NAME: tf.placeholder(tf.float32, [None, w,h,c])}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label_raw': tf.FixedLenFeature([], tf.string),
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.reshape(image, [w,h,c])
    image = tf.cast(image, tf.float32) * (1. / 255)
    
    label = tf.decode_raw(features['label_raw'], tf.uint16)
    label = tf.reshape(label, [w,h,cy])
    label = tf.cast(label, tf.uint16)
    
    return image, label

'''
python pascal.py /media/external/Downloads/data/pascal/VOCdevkit/VOC2012/JPEGImages \
/media/external/Downloads/data/pascal/trainval \
/media/external/Downloads/data/pascal/labels.txt
'''
if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image_folder_path',type=str)
    parser.add_argument('label_folder_path',type=str)
    parser.add_argument('label_txt_path',type=str)
    
    args = parser.parse_args()
    
    image_folder_path = args.image_folder_path
    label_folder_path = args.label_folder_path
    label_txt_path = args.label_txt_path

    if all([os.path.exists(x) for x in expected_file_list]):
        warnings.warn('data exists already!')
        sys.exit(0)

    if not os.path.exists(TF_DATA_DIR):
        os.makedirs(TF_DATA_DIR)

    data_sets = Pascal(image_folder_path,label_folder_path,label_txt_path)
        
    convert_to(data_sets.train, 'train', TF_DATA_DIR)
    convert_to(data_sets.validation, 'validation', TF_DATA_DIR)
    convert_to(data_sets.test, 'test', TF_DATA_DIR)

    set_num_examples(data_sets)

    print('done generating dataset.')
























