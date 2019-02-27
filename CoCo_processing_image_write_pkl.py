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
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

from scipy.misc import imresize

FLAGS = tf.app.flags.FLAGS
userId = ""
userId_text_dir = "testdata/profile.txt"
test_info = open(userId_text_dir).read().splitlines()
for i in range(len(test_info)):
    userId = test_info[i]
tf.flags.DEFINE_string("text_of_image_dir", "testdata/" + userId + "/input/image.txt",
                       "Directory text file")
tf.flags.DEFINE_string("text_of_interval_dir", "testdata/" + userId + "/input/interval_upper_data.txt",
                       "Directory text file")
tf.flags.DEFINE_string("body_pkl_dir", "testdata/" + userId + "/input/body_pickle/",
                       "Directory body pickle file")
tf.flags.DEFINE_string("body_image_dir", "testdata/" + userId + "/input/body_images/",
                       "")
tf.flags.DEFINE_string("body_segment_dir", "testdata/" + userId + "/input/body_segment/",
                       "Directory segment file")
tf.flags.DEFINE_string("body_pose_dir", "testdata/" + userId + "/input/body_pose/",
                       "Directory pose file")

tf.flags.DEFINE_string("upper_pkl_dir", "testdata/" + userId + "/input/upper_pickle/",
                       "Directory upper pickle file file")
tf.flags.DEFINE_string("upper_image_dir", "testdata/" + userId + "/input/upper_images/",
                       "")
tf.flags.DEFINE_string("upper_segment_dir", "testdata/" + userId + "/input/upper_segment/",
                       "Directory segment file")
tf.flags.DEFINE_string("upper_pose_dir", "testdata/" + userId + "/input/upper_pose/",
                       "Directory pose file")

def extract_pose_keypoints(pose):
    pose_keypoints = - np.ones((18, 2), dtype=int)
    for i in range(18):
        if pose['subset'][0, i] != -1:
            pose_keypoints[i, :] = pose['candidate'][int(pose['subset'][0, i]), :2]
    return pose_keypoints
def extract_pose_map(pose_keypoints, h, w, resize_h=256.0, resize_w=192.0):
    pose_keypoints = np.asarray(pose_keypoints, np.float32)
    pose_keypoints[:, 0] = pose_keypoints[:, 0] * resize_w / float(w)
    pose_keypoints[:, 1] = pose_keypoints[:, 1] * resize_h / float(h)
    pose_keypoints = np.asarray(pose_keypoints, np.int)
    pose_map = np.zeros((int(resize_h), int(resize_w), 18), np.bool)
    for i in range(18):
        if pose_keypoints[i, 0] < 0:
            continue
        t = np.max((pose_keypoints[i, 1] - 5, 0))
        b = np.min((pose_keypoints[i, 1] + 5, h - 1))
        l = np.max((pose_keypoints[i, 0] - 5, 0))
        r = np.min((pose_keypoints[i, 0] + 5, w - 1))
        pose_map[t:b+1, l:r+1, i] = True
    return pose_map
def process_segment_map(segment, h, w):
    segment = np.asarray(segment, dtype=np.uint8)
    segment = imresize(segment, (h, w), interp='nearest')
    return segment
def extract_segmentation(segment, label):
    """
        # 0 : 배경
    # 1 : 모자, 2 : 머리, 3 : 장갑, 4 : 선글라스, 5 : 상의
    # 6 : 드레스, 7 : 코트, 8 : 양말, 9 : 바지, 10 : 원피스형 수트
    # 11 : 스카프, 12 : 치마, 13 : 얼굴, 14 : 왼쪽 팔, 15 : 오른팔
    # 16 : 왼쪽 다리, 17 : 오른쪽 다리, 18 : 왼쪽 신발, 18 : 오른쪽 신발
    """
    product_segmentation = tf.cast(tf.equal(segment, label), tf.float32)
    skin_segmentation = (tf.cast(tf.equal(segment, 1), tf.float32) +
                         tf.cast(tf.equal(segment, 2), tf.float32) +
                         tf.cast(tf.equal(segment, 4), tf.float32) +
                         tf.cast(tf.equal(segment, 13), tf.float32))
    body_segmentation = (1.0 - tf.cast(tf.equal(segment, 0), tf.float32) - skin_segmentation)
    # Extend the axis
    product_segmentation = tf.expand_dims(product_segmentation, -1)
    body_segmentation = tf.expand_dims(body_segmentation, -1)
    skin_segmentation = tf.expand_dims(skin_segmentation, -1)

    body_segmentation = tf.image.resize_images(body_segmentation,
                                    size=[16, 12],
                                    method=tf.image.ResizeMethod.AREA,
                                    align_corners=False)
    
    return body_segmentation, product_segmentation, skin_segmentation
def _process_image(image_name, sess, tag="", resize_width=192,resize_height=256):
    image_id = image_name[:-4]
    print(FLAGS.upper_image_dir + image_name)
    if tag=="upper":
        image = scipy.misc.imread(FLAGS.upper_image_dir + image_name)
        segment_raw = sio.loadmat(os.path.join(FLAGS.upper_segment_dir, image_id))["segment"]
        pose_raw = sio.loadmat(os.path.join(FLAGS.upper_pose_dir, image_id))
    elif tag=="lower":
        image = scipy.misc.imread(FLAGS.body_image_dir + image_name)
        segment_raw = sio.loadmat(os.path.join(FLAGS.body_segment_dir, image_id))["segment"]
        pose_raw = sio.loadmat(os.path.join(FLAGS.body_pose_dir, image_id))

    segment_raw = process_segment_map(segment_raw, image.shape[0], image.shape[1])
    pose_raw = extract_pose_keypoints(pose_raw)
    pose_raw = extract_pose_map(pose_raw, image.shape[0], image.shape[1])
    pose_raw = np.asarray(pose_raw, np.float32)

    if tag=="upper":
        body_segment, prod_segment, skin_segment = extract_segmentation(segment_raw, label=5)
    elif tag=="lower":
        body_segment, prod_segment, skin_segment = extract_segmentation(segment_raw, label=9)

    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    image = tf.image.resize_images(image, size=[resize_height, resize_width],
                                          method=tf.image.ResizeMethod.BILINEAR)
    body_segment = tf.image.resize_images(body_segment,
                                          size=[resize_height, resize_width],
                                          method=tf.image.ResizeMethod.BILINEAR,
                                          align_corners=False)
    skin_segment = tf.image.resize_images(skin_segment,
                                          size=[resize_height, resize_width],
                                          method=tf.image.ResizeMethod.BILINEAR,
                                          align_corners=False)
    prod_segment = tf.image.resize_images(prod_segment,
                                          size=[resize_height, resize_width],
                                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = (image - 0.5) * 2.0
    
    skin_segment = skin_segment * image

    [image, body_segment, prod_segment, skin_segment] = sess.run(
        [image, body_segment, prod_segment, skin_segment])
    
    return image, pose_raw, body_segment, prod_segment, skin_segment
def _load_and_process_data(image, pose_raw, body_segment, prod_segment, skin_segment, interval, resized_interval, tag=""):
    
    info_data = {}
    info_data['image'] = image
    info_data['pose_raw'] = pose_raw
    info_data['body_seg'] = body_segment
    info_data['prod_seg'] = prod_segment
    info_data['skin_seg'] = skin_segment
    
    info_data['interval'] = interval
    info_data['resized_interval'] = resized_interval
    return info_data
def main(unused_argv):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    tf.reset_default_graph()
    test_info = open(FLAGS.text_of_image_dir).read().splitlines()
    interval_info_list = open(FLAGS.text_of_interval_dir).read().splitlines()

    size = 1
    with tf.Session() as sess:
        for i in range(len(test_info)):
            image_name = test_info[i]
            interval_info = interval_info_list[i].split()
            interval = interval_info[1]
            resized_interval = interval_info[2]
            (image, pose_raw, body_segment,
                prod_segment, skin_segment) = _process_image(image_name, sess, tag="upper")
            upper_data = _load_and_process_data(image, pose_raw, body_segment,
                                          prod_segment, skin_segment, interval, resized_interval, tag="upper")
            (image2, pose_raw2, body_segment2,
                prod_segment2, skin_segment2) = _process_image(image_name, sess, tag="lower")
            lower_data = _load_and_process_data(image2, pose_raw2, body_segment2,
                                          prod_segment2, skin_segment2, interval, resized_interval, tag="lower")
            with open(FLAGS.upper_pkl_dir + str(image_name[:-4]) + '.pkl', 'wb') as f:
                pickle.dump(upper_data, f, pickle.HIGHEST_PROTOCOL)
            with open(FLAGS.body_pkl_dir + str(image_name[:-4]) + '.pkl', 'wb') as f:
                pickle.dump(lower_data, f, pickle.HIGHEST_PROTOCOL)
    
        
if __name__=="__main__":
    tf.app.run()
    