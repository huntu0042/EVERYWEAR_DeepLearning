from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import collections
import os
import numpy as np
import scipy.io as sio
import scipy.misc
import matplotlib.pyplot as plt
import cv2

from scipy.misc import imresize
composed_upper = True
pants_switch = True
shorts_switch = False
userID = "ksp0519"
model_image_dir = "../testdata/" + userID + "/body_images/"
composed_upper_final_dir = "../testdata/" + userID + "/final_upper_images/"
composed_lower_image_dir = "../testdata/" + userID + "/composed_lower_images/"
composed_image_dir = ""
result_body_dir = "../testdata/" + userID + "/final_images/"
text_of_image_dir = "../testdata/" + userID + "/composed_lower_test.txt"

def _product_dir_change():
    if pants_switch:
        composed_image_dir = composed_lower_image_dir + "pants/"
    elif shorts_switch:
        composed_image_dir = composed_lower_image_dir + "shorts/"
    return composed_image_dir
def _process_ratio(h, w, composition_mask):
    #height_ratio = h / composition_raw.shape[0]
    #width_ratio = w / composition_raw.shape[1]
    #update_h = int(h * height_ratio)
    #update_w = int(w * width_ratio)
    
    #resized_composition_raw = cv2.resize(composition_raw, (w, h), interpolation = cv2.INTER_CUBIC)
    resized_composition_mask = cv2.resize(composition_mask, (w, h), interpolation = cv2.INTER_CUBIC)
    return resized_composition_mask

def _model_dir_change():
    if composed_upper:
        model_image_dir = composed_upper_final_dir
    return model_image_dir

def _load_image(img_name_dir):
    img = cv2.imread(img_name_dir)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def _original_image_resize(img):
    resized_image = imresize(img, (256, 192), interp = 'nearest')
    return resized_image

def _cvt_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def process_raw_mask(mask):
    gray_mask = _cvt_gray(mask)
    ret, raw = cv2.threshold(gray_mask, 100, 255, cv2.THRESH_BINARY)
    return raw
'''
def _compose_lower_image22(test):
    model_image_name = test[0] # 합성된 이미지
    composed_model_image_name = model_image_name # 상의가 합성된 이미지
    # original에는 상의 이름도 포함되어 있음.
    if composed_upper:
        model_image_name = model_image_name[:12] # 최초 원본 이미지

    product_image_name = test[1]
    composition_name = model_image_name + "_" + product_image_name + "_"
    print(model_image_dir + composed_model_image_name)
    model_image = _load_image(model_image_dir + composed_model_image_name)
    model_image_height = model_image.shape[0]
    model_image_width = model_image.shape[1]
    #model_image = _original_image_resize(model_image)

    composition_raw = _load_image(composed_image_dir + composition_name + "final.png")
    composition_mask = _load_image(composed_image_dir + composition_name + "mask.png")

    composition_raw, composition_mask = _process_ratio(model_image_height, model_image_width, composition_raw, composition_mask)
    composition_mask = process_raw_mask(composition_mask)
    composition_mask_reverse = cv2.bitwise_not(composition_mask)
  
    fg = cv2.bitwise_and(composition_raw, composition_raw, mask = composition_mask)
    bg = cv2.bitwise_and(model_image, model_image, mask = composition_mask_reverse)

    final_image = cv2.add(fg, bg)
    final_name = composed_model_image_name[:-10] + "_" + product_image_name + "_"
    
    final_image_name = result_body_dir + final_name + "final.png"
    scipy.misc.imsave(final_image_name, final_image)
    

if __name__ == "__main__":
    try:
        os.mkdir(result_body_dir)
    except:
        pass
    composed_image_dir = _product_dir_change()
    model_image_dir = _model_dir_change()
    print(composed_image_dir)
    print(model_image_dir)
    test_info = open(text_of_image_dir).read().splitlines()
    for test in test_info:
        print(test)
        _compose_lower_image22(test.split())
    '''    

