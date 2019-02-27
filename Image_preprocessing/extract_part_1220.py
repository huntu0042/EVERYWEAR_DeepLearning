from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import numpy as np
import scipy.io as sio
import scipy.misc
from scipy.misc import imresize
from PIL import Image
import matplotlib.pyplot as plt
import cv2

def load_image(data_list_dir, body_image_dir):
  image_list = open(data_list_dir, "r").read().splitlines()
  file_name = image_list[0] # 000001_0.jpg
  image_id = file_name[:-4] # 000001_0
  body_image_cv = cv2.imread(body_image_dir + file_name)
  body_image_cv = cv2.resize(body_image_cv, (480, 640), interpolation = cv2.INTER_AREA)
  return file_name, image_id, body_image_cv

def load_segment_info(body_segment_dir, image_id):
  segment_mat = sio.loadmat(os.path.join(
      body_segment_dir, image_id + ".mat"))["segment"]
  segment_image = cv2.imread(body_segment_dir + image_id + "_vis.png")
  binary_segment = cv2.imread(body_segment_dir + image_id + ".png")
  return segment_mat, segment_image, binary_segment

def cropping_upper_segment_height(segment_mat, segment_image):
  height = segment_image.shape[0] # 640
  width = segment_image.shape[1] # 480
  cropping_lower_height = height / 10 # 64
  cropped_height = 0 # 자르는 높이
  find_pants = False
  cropped_height_segment_mat = np.zeros((1, width))

  for i in range(height):
    if cropping_lower_height <= 0:
      break
    cropped_height += 1
    check = segment_mat[i][int(width / 2)]
    check_list = segment_mat[i][:]
    cropped_height_segment_mat = np.vstack(
        [cropped_height_segment_mat, check_list])

    if check == 9:
      find_pants = True
    if find_pants == True:
      cropping_lower_height -= 1
  '''
  print(cropped_height)
  segment_img_trim = segment_image[:cropped_height, :]
  print(segment_img_trim.shape)
  '''
  
  return cropped_height_segment_mat, cropped_height

def cropping_upper_segment_width(segment_mat, segment_image, cropped_height):
  height = segment_image.shape[0] # 640
  width = segment_image.shape[1] # 480
  # 640 : 480 = 3 : 4 = x : cropped_height
  cropped_width = int(cropped_height * 0.75)
  print("cropped_height: " +str(cropped_width))
  print("cropped_ratio: " + str(cropped_width / cropped_height))
  # 잘라야 하는 너삐
  cropping_width = int((width - cropped_width) * 0.5)
  upper_segment = np.zeros((1, cropped_height + 1))
  segment_mat = np.transpose(segment_mat)
  
  for i in range(cropping_width, width - cropping_width + 1):
    upper_segment = np.vstack([upper_segment, segment_mat[i]])
  upper_segment = np.transpose(upper_segment)
  return upper_segment, cropping_width

def image_trim(body_image_cv, segment_image, binary_segment, 
                cropped_height, cropping_width):
  
  print(cropping_width)
  trimmed_body_image_cv = body_image_cv[:cropped_height + 1, cropping_width - 1 : -cropping_width + 1]
  trimmed_segment_image = segment_image[:cropped_height + 1, cropping_width - 1 : -cropping_width + 1]
  trimmed_binary_segment = binary_segment[:cropped_height + 1, cropping_width - 1 : -cropping_width + 1]

  print(trimmed_body_image_cv.shape)
  print(trimmed_segment_image.shape)
  return trimmed_body_image_cv, trimmed_segment_image, trimmed_binary_segment
  '''
  cv2.imshow('img', trimmed_segment_image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  '''

def resizing_body_image(body_image_cv, upper_image):
  body_image_height = body_image_cv.shape[0] # 640
  body_image_width = body_image_cv.shape[1]  # 480
  
  # upper_image.shape[0] : 256 = 640 : x = 480 : y
  ratio = 256 / upper_image.shape[0]

  update_height = int(body_image_height * ratio)
  update_width = int(body_image_width * ratio)

  update_body_image = cv2.resize(body_image_cv, (update_width, update_height), interpolation = cv2.INTER_AREA)
  return update_body_image

def image_control_process(userId):
  # start
  data_list_dir = '../testdata/' + userId + '/input/image.txt'
  body_image_dir = "../testdata/" + userId + "/input/body_images/"
  body_resized_dir = "../testdata/" + userId + "/input/body_resized/"
  body_segment_dir = "../testdata/" + userId + "/input/body_segment/"
  result_upper_dir = "../testdata/" + userId + "/input/upper_images/"
  result_upper_segment_dir = "../testdata/" + userId + "/input/upper_segment/"
  interval_upper_dir = "../testdata/" + userId + "/input/interval_upper_data.txt"

  file_name, image_id, body_image_cv = load_image(data_list_dir, body_image_dir)
  segment_mat, segment_image, binary_segment = load_segment_info(body_segment_dir, image_id)
  cropped_height_segment_mat, cropped_height = cropping_upper_segment_height(segment_mat, segment_image)
  upper_segment, cropping_width = cropping_upper_segment_width(
      cropped_height_segment_mat, segment_image, cropped_height)
  
  #interval_upper_dir 써야함
  #cropped_height : 잘려진 높이
  #cropping_width : interval
  print(upper_segment.shape)
  print(cropping_width)
  upper_image, upper_segment_image, upper_binary_segment = image_trim(body_image_cv, 
      segment_image, binary_segment, cropped_height, cropping_width)
  # resize image
  resized_body_image = resizing_body_image(body_image_cv, upper_image)
  # save
  cv2.imwrite(result_upper_dir + file_name, upper_image)
  cv2.imwrite(result_upper_segment_dir + "segment_" + file_name, upper_segment_image)
  cv2.imwrite(result_upper_segment_dir + image_id + ".png", upper_binary_segment)
  cv2.imwrite(body_resized_dir + file_name, resized_body_image)
  sio.savemat('{}/{}.mat' .format(result_upper_segment_dir, image_id), {
      'segment':upper_segment}, do_compression=True)

  # 리사이징 된 interval, 256, 192에 맞추기 위해
  resized_cropping_width = int(cropping_width * (256 / cropped_height))
  f_1 = open(interval_upper_dir, "w")
  f_1.write(image_id + " " + str(cropping_width) + " " + str(resized_cropping_width) + '\n')
  f_1.close()
  



  
  

# debugging
'''
if __name__ == "__main__":
    image_control_process("testuser2")
'''
