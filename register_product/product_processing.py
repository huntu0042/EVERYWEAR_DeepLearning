from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import time
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf

from scipy.misc import imresize
import socket
import queue
import threading

FLAGS = tf.app.flags.FLAGS
tf.flags.DEFINE_string("prod_dir", "../data/",
                       "directory product image")
                       
wait_count = 0
wait_queue = queue.Queue()
def GetFlask():
  print("GETFLASK")
  s = socket.socket()
  host = socket.gethostname()
  port = 12226
  s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
  s.bind((host, port))
  s.setblocking(1)
  s.listen(5)
  c = None
  a = 1
  
  while True:
    print("#")
    print(a)
    print(c)
    if c is None or a == 1:
      print("[Waiting for connection...]")
      c, addr = s.accept()
      print("Got connection from", addr)
      a = 0
    else:
      print('[Waiting for response...]')
      wait_str = (c.recv(1024)).decode('utf-8')
      print(wait_str)
      print(len(wait_str))

      wait_list = wait_str.split()
      if len(wait_list) < 2:
        print("continue")
        c = None
        a = 1
        continue
      
      global wait_count
      wait_queue.put(wait_list)
      wait_count = wait_count + 1

      if wait_str == '0':
        print("shutdown")
        return 1
      c = None
      a = 1
def load_image(prodId):
  file_name = prodId + "_1.png"
  prod_image = cv2.imread(FLAGS.prod_dir + "/images/" + file_name)
  '''
  cv2.imshow('img', prod_image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  '''
  return prod_image
def process_prod(prodId, resize_width=192, resize_height=256):
  prod_image = scipy.misc.imread(FLAGS.prod_dir + "/images/" + prodId + "_1.jpg")
  with tf.Session() as sess:
    prod_image = tf.image.convert_image_dtype(prod_image, dtype=tf.float32)
    prod_image = tf.image.resize_images(prod_image,
                                        size=[resize_height, resize_width],
                                        method=tf.image.ResizeMethod.BILINEAR)
    prod_image = (prod_image - 0.5) * 2.0
    prod_image = sess.run(prod_image)
  
  print(prod_image.shape)
  return prod_image
def main(unused_argv):
  threading._start_new_thread(GetFlask, ())

  while True:
    while wait_queue.qsize() != 0:
      queue_data = wait_queue.get()
      print(queue_data)
      prodId = queue_data[0]
      category = queue_data[1]
      mall_name = "test_img"
      input_dir = "../data/" + mall_name
      if category == "1001":
        FLAGS.prod_dir = input_dir + "/men_tshirts"
      elif category == "1002":
        FLAGS.prod_dir = input_dir + "/men_nambang"
      elif category == "1003":
        FLAGS.prod_dir = input_dir + "/men_long"
      elif category == "1101":
        FLAGS.prod_dir = input_dir + "/men_pants"
      try:
        os.mkdir(FLAGS.prod_dir)
      except:
        pass
      
      prod_image = process_prod(prodId)
      info_data = {}
      info_data['prod_image'] = prod_image
      
      with open(FLAGS.prod_dir + "/pkl/" + prodId + '_1.pkl', 'wb') as f:
        pickle.dump(info_data, f, pickle.HIGHEST_PROTOCOL)
      print("Complete to register product")
      
      #scipy.misc.imsave(FLAGS.prod_dir + "/images/" + prodId + ".jpg", prod_image)

if __name__ == "__main__":
  tf.app.run()
    