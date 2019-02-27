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
import cv2


def image_trim_width(image_raw, cropping_width):
    img_trim = image_raw[:,cropping_width - 1 : -cropping_width + 1]
    
    return img_trim

def image_trim_height(image_raw, cropping_height, tag = ""):
    if tag == "upper":
        img_trim = image_raw[:cropping_height + 1,:]
    elif tag == "lower":
        img_trim = image_raw[-cropping_height:,:]
    
    return img_trim

def cropping_upper_segment_height(segment, height, width, crop_size):
    """
        Args:
            segment : 모델의 전신 segment(height * width)
            height : 모델의 전신 이미지 height
            width : 모델의 전신 이미지 width
            half : 모델의 전신 이미지 width / 2
            crop_size : 모델의 전신 이미지 height / 10
        Returns:
            update_seg : 사진의 1행 half열에서 탐색을 시작하여 바지를 발견한 영역으로부터 crop_size만큼 자름
            croped_height : crop된 이미지의 높이
    """
    half = int(width / 2)
    update_seg = np.zeros((1, width))
    
    croped_height = 0
    find_pants = False
    print("image_height: " + str(height))
    print("image_width: " + str(width))
    print("plus lower crop_size: " + str(crop_size))

    for i in range(height):
        
        croped_height += 1
        check = segment[i][half]
        
        check_list = segment[i][:]
        update_seg = np.vstack([update_seg, check_list])
        
        if crop_size <= 0:
            break
        if check == 9:
            find_pants = True
        #if croped_height == 625:
        #    break
        
        if find_pants == True:
            crop_size -= 1
    print(croped_height)

    return update_seg, croped_height
def cropping_lower_segment_height(segment, height, width, half, crop_size):
    update_seg = np.zeros((1, width))

    croped_height = 0
    find_clothes = False
    for i in range(height - 1, 1, -1):
        croped_height += 1
        check = segment[i][half]
        check_list = segment[i][:]
        update_seg = np.vstack([update_seg, check_list])

        if crop_size == 0:
            break
        if check == 5:
            find_clothes = True
        if find_clothes:
            crop_size -= 1
    
    update_seg2 = np.zeros((1, width))
    # reverse
    for i in range(update_seg.shape[0] - 1, 0, -1):
        update_seg2 = np.vstack([update_seg2, update_seg[i][:]])
    
    return update_seg2, croped_height
def cropping_segment_width(image_id, segment, croped_height, interval_upper_dir, tag=""):
    """
        Args:
            segment: 높이를 허벅지까지 자른 update segment
            croped_height: 허벅지까지 자른 segement의 높이(결국 imresize시켜거 잘린 이미지의 높이와 같음)
        Returns:
            update_seg: 잘린 height의 3/4로 너비를 맞춘 segment
            cropping_width: 잘라야 하는 width
    """
    # real_width: 만들어져야 하는 너비
    # cropping_width: 잘라야 하는 너비
    real_width = int(croped_height * 0.75) # 0.75, 3:4
    print("real_width: " + str(real_width))
    
    cropping_width = int((segment.shape[1] - real_width) * 0.5)
    #cropping_width = 141

    # resize_cropping_width = resized_interval
    resized_cropping_width = int(cropping_width * (256 / croped_height))
    #resized_cropping_width = 57

    if tag == "upper":
        f = open(interval_upper_dir, "w")
        
        f.write(image_id + " " + str(cropping_width) + " " + str(resized_cropping_width))
        
        
        f.write("\n")
        f.close()
    segment = np.transpose(segment)
    
    update_seg = np.zeros((1, croped_height + 1))
    for i in range (cropping_width, segment.shape[1] - cropping_width + 1):
        update_seg = np.vstack([update_seg, segment[i]])
    update_seg = np.transpose(update_seg)
    return update_seg, cropping_width


def process_segment_map(segment, height, width, isSegment=True):
    """
        Args:
            segment: 모델의 전신 segment 파일(644 * 483)
            height: 모델의 전신 이미지 height
            width: 모델의 전신 이미지 width
        Returns:
            644 * 483 -> 모델의 전신 이미지의 height * width 크기로 바뀐 segment
    """
    
    if(isSegment == True):
        segment = np.asarray(segment)
        # segment = imresize(segment, (height, width), interp = 'nearest')
        segment = cv2.resize(segment, (width, height), interpolation = cv2.INTER_AREA)
        
    else:
        segment = np.asarray(segment)
        # segment = imresize(segment, (height, width), interp = 'nearest')
        segment = cv2.resize(segment, (width, height), interpolation = cv2.INTER_AREA)

    return segment

def image_resize(image_raw, upper_raw):
    image_height = image_raw.shape[0]
    image_width = image_raw.shape[1]

    update_width2 = image_height * 0.75
    if(image_width > update_width2):
        image_raw = image_trim_width(image_raw, int((image_width - update_width2) / 2))
    
    ratio = 256 / upper_raw.shape[0]

    update_height = int(image_height * ratio)
    update_width = int(image_width * ratio)

    update_image = cv2.resize(image_raw, (update_width, update_height), interpolation = cv2.INTER_AREA)
    #update_image = imresize(image_raw, (update_height, update_width), interp = 'nearest')

    return update_image
def _load_image(img_name_dir):
    img = cv2.imread(img_name_dir)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def _cropping_process_image(userId):
    data_list_dir = '../testdata/' + userId + '/input/image.txt'
    resized_dir = "../testdata/" + userId + "/input/body_resized/"
    segment_dir = "../testdata/" + userId + "/input/body_segment/"
    result_upper_dir = "../testdata/" + userId + "/input/upper_images/"
    result_upper_segment_dir = "../testdata/" + userId + "/input/upper_segment/"
    image_dir = "../testdata/" + userId + "/input/body_images/"
    interval_upper_dir = "../testdata/" + userId + "/input/interval_upper_data.txt"

    image_list = open(data_list_dir, "r").read().splitlines()
    file_name = image_list[0] # 000001_0.jpg
    image_id = file_name[:-4] # 000001_0
    image_raw = scipy.misc.imread(image_dir + file_name)
    image_cv_raw = _load_image(image_dir + file_name)

    image_height = int(image_raw.shape[0]) # original image's height
    image_width = int(image_raw.shape[1]) # original image's width

    # segment matlab file load
    segment_raw = sio.loadmat(os.path.join(
        segment_dir, image_id + ".mat"))["segment"]

    # segment colorful image load
    segment_image_raw = scipy.misc.imread(segment_dir + image_id + "_vis.png")
    
    segment_raw = process_segment_map(segment_raw, image_height, image_width)
    
    
    crop_size = int(image_height / 10)
    '''
        Extraction upper part
    '''
    update_upper_segment, cropped_height = cropping_upper_segment_height(segment_raw, image_height, image_width, crop_size)
    
    img_trim_height = image_trim_height(image_raw, cropped_height, tag="upper")
  
    
    update_upper_segment, cropping_width = cropping_segment_width(image_id, 
    update_upper_segment, cropped_height, interval_upper_dir,tag="upper")
    
    update_upper_image = image_trim_width(img_trim_height, cropping_width)
    
    crop_size = int(crop_size * 3)
    '''
    update_lower_segment, cropped_height2 = cropping_lower_segment_height(segment_raw, image_height, image_width, half, crop_size)
    
    img_trim_height2 = image_trim_height(image_raw, cropped_height2, tag="lower")
    
    update_lower_segment, cropping_width2 = cropping_segment_width(update_lower_segment, cropped_height2, tag="lower")
    update_lower_image = image_trim_width(img_trim_height2, cropping_width2)
    '''
    #update_segment = process_segment_map(update_segment, 644, 483, isSegment=True)
    #update_image = process_segment_map(update_image, 644, 483, isSegment=False)
    
    scipy.misc.imsave(result_upper_segment_dir + "segment_" + file_name, update_upper_segment)
    
    
    
    scipy.misc.imsave(result_upper_dir + file_name, update_upper_image)

    resized_body_image = image_resize(image_cv_raw, update_upper_image)
    scipy.misc.imsave(resized_dir + file_name, resized_body_image)
    print("save cropped upper image!" + str(file_name))
    #scipy.misc.imsave(result_lower_dir + "segment_" + file_name, update_lower_segment / 2.0 + 0.5)
    #scipy.misc.imsave(result_lower_dir + file_name, update_lower_image)
    #print("save cropped lower image!"+ str(file_name))
    sio.savemat('{}/{}.mat'.format(result_upper_segment_dir, image_id), {'segment':update_upper_segment}, do_compression=True)
    print("save upper segment!")
    #sio.savemat('{}/{}.mat'.format(result_lower_segment_dir, image_id), {'segment':update_lower_segment}, do_compression=True)
    #print("save lower segment!")



'''
if __name__ == "__main__":
    try:
        os.mkdir(result_upper_dir)
    except:
        pass
    file_name_list = open("testdata/" + userId + "/input/image.txt").read().splitlines()
    for image_name in file_name_list:
        _process_image(image_name)
'''