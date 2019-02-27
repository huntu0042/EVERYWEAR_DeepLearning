from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import collections
import os
import numpy as np
import pickle as pkl
import scipy.io as sio
import scipy.misc
import matplotlib.pyplot as plt
import cv2

from scipy.misc import imresize
tshirts_switch = True
nambang_switch = False



def _load_image(img_name_dir):
  img = cv2.imread(img_name_dir)
  #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  return img

def _cvt_gray(img):
  return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def process_raw_image(img):
  gray_img = _cvt_gray(img)
  ret, raw = cv2.threshold(gray_img, 100, 255, cv2.THRESH_BINARY)
  return raw

def final_process():
  '''
  post_info(dict):
    - "userId"
    - "model_image"
    - "image_name"
    - "composition_name" : image_name
    - "fg" : upper_Id
    - "bg" : lower_Id
    - "interval" : isUpper
    - "isUpper" : input_dir
    - "final" : resized interval
  '''
  userId = "testuser2"
  image_name = "000001_0"
  upperId = "000043"
  lowerId = "000000"
  upper_name = upperId + "_1"
  lower_name = lowerId + "_1"
  interval = 57

  isUpper = 1

  upper_composition_name = image_name + "_" + upper_name + "_"
  lower_composition_name = image_name + "_" + lower_name + "_"
  full_composition_name = image_name + "_" + upper_name + "_" + lower_name + "_"

  middle_composition_name = ""
  if(isUpper == 0):
    print("Lower")
    middle_composition_name = lower_composition_name
    product_image_name = lowerId + '_1'
  elif(isUpper == 1):
    print("Upper")
    middle_composition_name = upper_composition_name
    product_image_name = upperId + '_1'
    
  input_dir = "testdata/" + userId + "/input"
  output_dir = "testdata/" + userId + "/output"
  stage_dir = "testdata/" + userId + "/stage/"
  result_dir_stage2 = "testdata/" + userId + "/output/composed_images/"

  with open(result_dir_stage2 + middle_composition_name + "0_pkl.pkl", "rb") as f1:
    try:
      model_mask = pkl.load(f1, encoding = "latin-1")
    except EOFError:
      print("pkl.load fail")
    

  with open(result_dir_stage2 + middle_composition_name + "1_pkl.pkl", "rb") as f1:
    try:
      prod_mask = pkl.load(f1, encoding = "latin-1")
    except EOFError:
      print("pkl.load fail")
  print(middle_composition_name)
  coarse_mask = _load_image(stage_dir + middle_composition_name + "mask.png")
  #coarse_mask = model_mask['model_mask']
  binary_segment_upper = _load_image(input_dir + "/upper_segment/" + image_name + ".jpg")
  model_image = _load_image(input_dir + "/body_resized/" + image_name + ".jpg")
  resized_binary_segment_upper = cv2.resize(binary_segment_upper, (192, 256), interpolation=cv2.INTER_AREA)
  #mask = process_raw_image(resized_binary_segment_upper)
  mask = cv2.add(resized_binary_segment_upper, coarse_mask)
  final_mask = process_raw_image(mask)
  #fg = cv2.cvtColor(fg, cv2.COLOR_BGR2RGB)
  #bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
  composition_raw = _load_image(result_dir_stage2 + middle_composition_name + "final.png")
  print(composition_raw.shape) 
  
  final_mask_reverse = cv2.bitwise_not(final_mask)
  fg = cv2.bitwise_and(composition_raw, composition_raw, mask=final_mask)
  
  roi = model_image[:256, interval:interval + 192]
  print(roi.shape)
  
  bg = cv2.bitwise_and(roi, roi, mask=final_mask_reverse)
  final_image = cv2.add(fg, bg)
  model_image[:256, interval:interval + 192] = final_image
  cv2.imshow('img', fg)
  cv2.imshow('img2', bg)
  cv2.imshow('img3', model_image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  # bg = cv2.bitwise_and()
  cv2.imwrite(output_dir + "/final_images/" + middle_composition_name + "final3.png", model_image)
  '''
  coarse_image = cv2.add(fg, bg)
  binary_segment_model = _load_image(input_dir + "/body_segment/" + image_name + ".png")
  #binary_segment_upper = _load_image(result_dir_stage2 + image_name + "_" + product_image_name + "_mask.png")

  
  process_binary_segment_model = process_raw_image(update_binary_segment_model)
  reverse_binary_segment_model = cv2.bitwise_not(process_binary_segment_model)
  
    #coarse_image = cv2.add(binary_segment, bg)
  #final_image = cv2.add(fg, coarse_image)
  set_roi = cv2.bitwise_and(model_image, model_image, mask=reverse_binary_segment_model)

  print(set_roi.shape)
  print(update_binary_segment_model.shape)
  white_roi_image = cv2.add(update_binary_segment_model, set_roi)
  
  # image_preprocessing에서 segment를 자를 때 binary_segment도 같은 모양으로 잘라서
  # 예측된 세그먼트가 아닌 원본 세그먼트에서 잘라서 이미지를 얻어보자. 12/27에 한다. ㅅㅂ
  '''
  
  
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  
  
  
if __name__ == "__main__":
  final_process()
