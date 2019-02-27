from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from subprocess import call

import collections
import os
import time

import numpy as np
import scipy.io as sio
import scipy.misc
import pickle
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from configobj import ConfigObj
from scipy.misc import imresize


# JPPNet_dir

text_dir = "testdata/profile.txt"
image_dir = ""
JPPNet_dir = "LIP_JPPNet/evaluate_parsing_JPPNet-s2.py"
extract_part_dir = "extract_part.py"
pose_est_dir = "Pose_Estimation/demo_image.py"
image_write_pkl_dir = "CoCo_processing_image_write_pkl.py"
def write_id(userId, imageId):
  with open(text_dir, "a") as f:
    f.write(userId + '\n')
  input_dir = "testdata/" + userId + "/input/"
  with open(input_dir + "image.txt", "w") as f:
    f.write(imageId + "_0.jpg")
  
  return input_dir + "body_images/" + imageId + "_0.jpg"
    

'''
    수정사항
    0. 코드를 실행할 때마다 tf.reset_default_graph()로 그래프 초기화 해준다.
    1. LIPJPPNet에 쓰이는 모든 코드 파일 경로를 이쪽으로 설정
    2. extract_part.py 경로 수정
    3. pose_estimation에 있는 config파일을 현재 경로에 복사
    4. body_images를 제외한 폴더 모두 삭제해도 가능

'''

def main(unused_argv):
  #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  #os.environ["CUDA_VISIBLE_DEVICE"] = "-1"
  my_env = os.environ.copy()
  my_env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  my_env["CUDA_VISIBLE_DEVICE"] = "-1"
  #userid,image = queuedata4개
  
  p1 = time.time()
  userId = "category39"
  imageId = "000001"
  input_dir = "testdata/" + userId + "/input/"
  try:
    os.mkdir(input_dir + "/body_resized")
    os.mkdir(input_dir + "/body_pose")
    os.mkdir(input_dir + "/body_segment")
    
    os.mkdir(input_dir + "/upper_images")
    os.mkdir(input_dir + "/upper_pose")
    os.mkdir(input_dir + "/upper_segment")

    os.mkdir(input_dir + "/upper_pickle")
    os.mkdir(input_dir + "/body_pickle")
  except:
    pass
  body_image_dir = write_id(userId,imageId)
  
  while(True):
    start1 = time.time()
    call("python " + JPPNet_dir, shell=False)
    start2 = time.time()
    '''
    start3 = time.time()
    call("python " + extract_part_dir, shell=False)
    start4 = time.time()
    
    start5 = time.time()
    call("python " + pose_est_dir, shell=False)
    start6 = time.time()
    
    start7 = time.time()
    call("python " + image_write_pkl_dir, shell=False)
    start8 = time.time()
    '''
    print("Segment parsing complete: " + str(start2 - start1))
    #print("Extract part complete: " + str(start4 - start3))
    #print("Pose Estimation complete: " + str(start6 - start5))
    #print("write pkl complete: " + str(start8 - start7))
  p2 = time.time()
  print("Image preprocessing complete: " + str(p2 - p1))
  #os.system("python " + JPPNet_dir)

  '''
  test_info = open(text_dir).read().splitlines()
  for i in range(len(test_info)):
      userId = test_info[i]
  '''    
      
    

if __name__ == "__main__":
  tf.app.run()
