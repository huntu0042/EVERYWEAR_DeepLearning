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
tshirts_switch = True
nambang_switch = False


def _load_image(img_name_dir):
    img = cv2.imread(img_name_dir)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def _cvt_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def process_raw_image(img):
    gray_img = _cvt_gray(img)
    ret, raw = cv2.threshold(gray_img, 100, 255, cv2.THRESH_BINARY)
    return raw


'''
def process_image_ratio(model_image, cropped_image, composed_image, width_interval):
    # model_image의 비율을 cropped_image : composed_image에 맞춰 resize 진행
    height_ratio = composed_image.shape[0] / cropped_image.shape[0]
    width_ratio = composed_image.shape[1] / cropped_image.shape[1]
    update_model_height = int(model_image.shape[0] * height_ratio)
    update_model_width = int(model_image.shape[1] * width_ratio)

    resized_model_image = imresize(model_image, (update_model_height, update_model_width), interp = 'nearest')
    width_interval = int(width_ratio * width_interval)
    
    return resized_model_image, width_interval
'''
'''
def _compose_upper_image22(model_image_name, product_image_name, width_interval):
    
    composition_name = model_image_name + "_" + product_image_name + "_"
    
    composition_raw = _load_image(composed_upper_image_dir + composition_name + "final.png")
    composition_mask = _load_image(composed_upper_image_dir + composition_name + "mask.png")
    
    model_image = _load_image(model_image_dir + model_image_name)
    #plt.imshow(model_image)
    #plt.show()
    cropped_image = _load_image(cropped_image_dir + model_image_name)

    # 잘린 이미지와 합성된 이미지 비율 맞추기
    composition_mask = process_raw_image(composition_mask)
    composition_mask_reverse = cv2.bitwise_not(composition_mask)
    #original_height = person_body_image.shape[0]
    #original_width = person_body_image.shape[1]
    composition_height = composition_raw.shape[0]
    composition_width = composition_raw.shape[1]
    
    resized_model_image, resized_interval = process_image_ratio(model_image, cropped_image, composition_raw, width_interval)
    print("resized model image shape: " + str(resized_model_image.shape))
    # roi : 자를 영역 설정 256 * 192
    roi = resized_model_image[:composition_height, resized_interval:resized_interval + composition_width]

    # mask 조작
    
    
    
    fg = cv2.bitwise_and(composition_raw, composition_raw, mask = composition_mask)
    bg = cv2.bitwise_and(roi, roi, mask = composition_mask_reverse)
    
    
    
    cropped_final_image = cv2.add(fg, bg)
    #resized_model_image[:composition_height, resized_interval:resized_interval + composition_width] = composition_raw
    resized_model_image[:composition_height, resized_interval:resized_interval + composition_width] = cropped_final_image

    

    scipy.misc.imsave(result_body_dir + composition_name + "final.png", resized_model_image)
'''
'''    
    
if __name__ == "__main__":
    try:
        os.mkdir(result_body_dir)
    except:
        pass
    test_info = open(text_of_image_dir).read().splitlines()
    print(len(test_info))
    interval_list = open(text_of_interval_dir).read().splitlines()
    composed_image_dir = _dir_change()
    for num in range(len(test_info)):
        info = test_info[num].split()
        print(info)
        print(composed_image_dir)
        interval = int(interval_list[num])
        _compose_upper_image(info, interval)
        
'''