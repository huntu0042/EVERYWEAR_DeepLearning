

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import time
import numpy as np
import scipy.io as sio
import scipy.misc
import pickle
import tensorflow as tf

from scipy.misc import imresize

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("text_of_prod_dir", "prod_id.txt",
                       "directory text of product")
tf.flags.DEFINE_string("pkl_dir", "dopkl/",
                       "directory product pickle")
tf.flags.DEFINE_string("prod_image_dir", "dopkl/",
                       "directory product image")

def _process_prod(prod_name, sess, resize_width=192,resize_height=256):
    prod_id = prod_name[:-4]
    print(FLAGS.prod_image_dir + prod_name)
    prod_image = scipy.misc.imread(FLAGS.prod_image_dir + prod_name)
    prod_image = tf.image.convert_image_dtype(prod_image, dtype=tf.float32)
    prod_image = tf.image.resize_images(prod_image,
                                        size=[resize_height, resize_width],
                                        method=tf.image.ResizeMethod.BILINEAR)
    prod_image = (prod_image - 0.5) * 2.0
    prod_image = sess.run(prod_image)

    return prod_image
def _load_and_process_data(prod_name, prod_image):
    
    info_data = {}
    info_data['prod_image'] = prod_image
    return info_data
def main(unused_argv):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    test_info = open(FLAGS.text_of_prod_dir).read().splitlines()
    
    
    with tf.Session() as sess:
        for i in range(len(test_info)):
            prod_name = test_info[i]
            
            prod_image = _process_prod(prod_name, sess)
##
            data = _load_and_process_data(prod_name, prod_image)
            with open(FLAGS.pkl_dir + str(prod_name[:-4]) + '.pkl', 'wb') as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        
if __name__=="__main__":
  '''
  f = open("prod_id.txt", "w")
  
  for i in range(7001, 7008):
      f.write(str(i).zfill(6) + "_1.jpg")
      f.write("\n")
  f.close()
  '''
  tf.app.run()
    