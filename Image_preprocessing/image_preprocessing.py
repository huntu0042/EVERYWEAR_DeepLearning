from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import scipy.misc

import scipy.io as sio

import argparse
import configobj

import math
import pickle
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf


from configobj import ConfigObj
from scipy.misc import imresize
from LIP_utils import *
from LIP_model import *
from extract_part_1220 import image_control_process
from scipy.ndimage.filters import gaussian_filter

import Pose_Estimation.util

from PIL import Image

from Pose_Estimation.model.cmu_model import get_testing_model
from keras.models import Model
from keras.layers.merge import Concatenate
from keras.layers import Activation, Input, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Multiply
from keras.regularizers import l2
from keras.initializers import random_normal,constant

import threading
import socket
import queue
import requests

wait_count = 0
wait_queue = queue.Queue()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

text_dir = "../testdata/profile.txt"
image_dir = ""


#리스트를 받아서
def GetFlask():
    print("GETFLASK")
    s = socket.socket()
    host = socket.gethostname()
    port = 12224
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
             # Halts
            print( '[Waiting for connection...]')
            c, addr = s.accept() #  (socket object, address info) return
            print( 'Got connection from', addr)
            a = 0
        else:
             # Halts
            #time.sleep(3)
            print( '[Waiting for response...]')
            wait_str = (c.recv(1024)).decode('utf-8') #여기서 멈춘다
            print(wait_str)
            print(len(wait_str))
            
            wait_list = wait_str.split()
            if len(wait_list) < 2 :
                print("continue")
                c=None
                a=1
                continue
            

            global wait_count
            
            wait_queue.put(wait_list)
            wait_count = wait_count+1
                        
            if wait_str=='0':
                print("shutdown")
                return 1
            c = None    
            a=1
             #c.send(q.encode('utf-8'))

label_colours = [(0,0,0)
                # 0=Background
                ,(128,0,0),(255,0,0),(0,85,0),(170,0,51),(255,85,0)
                # 1=Hat,  2=Hair,    3=Glove, 4=Sunglasses, 5=UpperClothes
                ,(0,0,85),(0,119,221),(85,85,0),(0,85,85),(85,51,0)
                # 6=Dress, 7=Coat, 8=Socks, 9=Pants, 10=Jumpsuits
                ,(52,86,128),(0,128,0),(0,0,255),(51,170,221),(0,255,255)
                # 11=Scarf, 12=Skirt, 13=Face, 14=LeftArm, 15=RightArm
                ,(85,255,170),(170,255,85),(255,255,0),(255,170,0)]
                # 16=LeftLeg, 17=RightLeg, 18=LeftShoe, 19=RightShoe
limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
           [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
           [1, 16], [16, 18], [3, 17], [6, 18]]

# the middle joints heatmap correpondence
mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22], \
          [23, 24], [25, 26], [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52], \
          [55, 56], [37, 38], [45, 46]]

# visualize
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
          [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255],
          [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
def write_id(userId, imageId):
  with open(text_dir, "w") as f:
    f.write(userId + '\n')
  input_dir = "../testdata/" + userId + "/input/"
  with open(input_dir + "image.txt", "w") as f:
    f.write(imageId + "_0.jpg")
  
  return input_dir + "body_images/" + imageId + "_0.jpg", input_dir + "image.txt"
def decode_labels(mask, img_id, output_dir, num_images=1, num_classes=20):
    """Decode batch of segmentation masks.
    
    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
  
    Returns:
      A batch with num_images RGB images of the same size as the input. 
    """
    resize_height = 640
    resize_width = 480
    n, h, w, c = mask.shape
    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    load_segment = np.zeros((resize_height, resize_width), dtype=np.uint8)
    bw_upper_segment = np.zeros((resize_height, resize_width), dtype=np.uint8)
    bw_lower_segment = np.zeros((resize_height, resize_width), dtype=np.uint8)
    for i in range(num_images):
      img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
      pixels = img.load()
      
      for j_, j in enumerate(mask[i, :, :, 0]): #j_ : row, j : enumerate(mask[0,:,:,0])
          for k_, k in enumerate(j):            #k_ : col, k : label num
              if k < num_classes:
                  pixels[k_,j_] = label_colours[k]
                  load_segment[j_,k_] = k
                  if k == 5:
                    bw_upper_segment[j_,k_] = 255
                  elif k == 9:
                    bw_lower_segment[j_,k_] = 255
                  else:
                    bw_upper_segment[j_,k_] = 0
                    bw_lower_segment[j_,k_] = 0

                  
      sio.savemat('{}/{}.mat'.format(output_dir, img_id), {'segment':load_segment}, do_compression=True)
      img_id2 = img_id[:-1] + "1"
      scipy.misc.imsave(output_dir + "/" + img_id + ".png", bw_upper_segment)
      scipy.misc.imsave(output_dir + "/" + img_id2 + ".png", bw_lower_segment)
      
      outputs[i] = np.array(img)  
    return outputs
def evaluate_segment_parsing(userId):
  segment_graph = tf.Graph()
  with segment_graph.as_default() as graph1:
    body_image_dir = '../testdata/' + userId + '/input/body_images/'
    data_list_dir = '../testdata/' + userId + '/input/image.txt'
    restore_from = 'LIP_JPPNet/checkpoint/JPPNet-s2'
    output_dir = "../testdata/" + userId + '/input/body_segment/'

    steps = 1
    resize_width = 480
    resize_height = 640
    input_size = (640, 480)
    N_CLASSES = 20

    #test_sess = tf.Session()

    coord = tf.train.Coordinator()
    h, w = input_size
    with tf.name_scope("create_inputs"):
      reader = ImageReader(body_image_dir, data_list_dir, None, False, False, coord)
      image = reader.image
      image = tf.image.resize_images(image, size=[resize_height, resize_width], method=tf.image.ResizeMethod.BILINEAR)
      image_rev = tf.reverse(image, tf.stack([1]))
      image_list = reader.image_list

    image_batch_origin = tf.stack([image, image_rev])
    image_batch = tf.image.resize_images(image_batch_origin, [int(h), int(w)])
    image_batch075 = tf.image.resize_images(image_batch_origin, [int(h * 0.75), int(w * 0.75)])
    image_batch125 = tf.image.resize_images(image_batch_origin, [int(h * 1.25), int(w * 1.25)])

    with tf.variable_scope('', reuse=tf.AUTO_REUSE):
        net_100 = JPPNetModel({'data': image_batch}, is_training=False, n_classes=N_CLASSES)
    with tf.variable_scope('', reuse=True):
      net_075 = JPPNetModel({'data': image_batch075}, is_training=False, n_classes=N_CLASSES)
    with tf.variable_scope('', reuse=True):
      net_125 = JPPNetModel({'data': image_batch125}, is_training=False, n_classes=N_CLASSES)

    
    # parsing net
    parsing_fea1_100 = net_100.layers['res5d_branch2b_parsing']
    parsing_fea1_075 = net_075.layers['res5d_branch2b_parsing']
    parsing_fea1_125 = net_125.layers['res5d_branch2b_parsing']

    parsing_out1_100 = net_100.layers['fc1_human']
    parsing_out1_075 = net_075.layers['fc1_human']
    parsing_out1_125 = net_125.layers['fc1_human']

    # pose net
    
    resnet_fea_100 = net_100.layers['res4b22_relu']
    resnet_fea_075 = net_075.layers['res4b22_relu']
    resnet_fea_125 = net_125.layers['res4b22_relu']
    
    
    with tf.variable_scope('', reuse=tf.AUTO_REUSE):
      pose_out1_100, pose_fea1_100 = pose_net(resnet_fea_100, 'fc1_pose')
      pose_out2_100, pose_fea2_100 = pose_refine(pose_out1_100, parsing_out1_100, pose_fea1_100, name='fc2_pose')
      parsing_out2_100, parsing_fea2_100 = parsing_refine(parsing_out1_100, pose_out1_100, parsing_fea1_100, name='fc2_parsing')
      parsing_out3_100, parsing_fea3_100 = parsing_refine(parsing_out2_100, pose_out2_100, parsing_fea2_100, name='fc3_parsing')

    with tf.variable_scope('', reuse=True):
      pose_out1_075, pose_fea1_075 = pose_net(resnet_fea_075, 'fc1_pose')
      pose_out2_075, pose_fea2_075 = pose_refine(pose_out1_075, parsing_out1_075, pose_fea1_075, name='fc2_pose')
      parsing_out2_075, parsing_fea2_075 = parsing_refine(parsing_out1_075, pose_out1_075, parsing_fea1_075, name='fc2_parsing')
      parsing_out3_075, parsing_fea3_075 = parsing_refine(parsing_out2_075, pose_out2_075, parsing_fea2_075, name='fc3_parsing')

    with tf.variable_scope('', reuse=True):
      pose_out1_125, pose_fea1_125 = pose_net(resnet_fea_125, 'fc1_pose')
      pose_out2_125, pose_fea2_125 = pose_refine(pose_out1_125, parsing_out1_125, pose_fea1_125, name='fc2_pose')
      parsing_out2_125, parsing_fea2_125 = parsing_refine(parsing_out1_125, pose_out1_125, parsing_fea1_125, name='fc2_parsing')
      parsing_out3_125, parsing_fea3_125 = parsing_refine(parsing_out2_125, pose_out2_125, parsing_fea2_125, name='fc3_parsing')


    parsing_out1 = tf.reduce_mean(tf.stack([tf.image.resize_images(parsing_out1_100, tf.shape(image_batch_origin)[1:3,]),
                          tf.image.resize_images(parsing_out1_075, tf.shape(image_batch_origin)[1:3,]),
                          tf.image.resize_images(parsing_out1_125, tf.shape(image_batch_origin)[1:3,])]), axis=0)
    parsing_out2 = tf.reduce_mean(tf.stack([tf.image.resize_images(parsing_out2_100, tf.shape(image_batch_origin)[1:3,]),
                          tf.image.resize_images(parsing_out2_075, tf.shape(image_batch_origin)[1:3,]),
                          tf.image.resize_images(parsing_out2_125, tf.shape(image_batch_origin)[1:3,])]), axis=0)
    parsing_out3 = tf.reduce_mean(tf.stack([tf.image.resize_images(parsing_out3_100, tf.shape(image_batch_origin)[1:3,]),
                          tf.image.resize_images(parsing_out3_075, tf.shape(image_batch_origin)[1:3,]),
                          tf.image.resize_images(parsing_out3_125, tf.shape(image_batch_origin)[1:3,])]), axis=0)

    raw_output = tf.reduce_mean(tf.stack([parsing_out1, parsing_out2, parsing_out3]), axis=0)
    head_output, tail_output = tf.unstack(raw_output, num=2, axis=0)
    tail_list = tf.unstack(tail_output, num=20, axis=2)
    tail_list_rev = [None] * 20
    for xx in range(14):
      tail_list_rev[xx] = tail_list[xx]
    tail_list_rev[14] = tail_list[15]
    tail_list_rev[15] = tail_list[14]
    tail_list_rev[16] = tail_list[17]
    tail_list_rev[17] = tail_list[16]
    tail_list_rev[18] = tail_list[19]
    tail_list_rev[19] = tail_list[18]
    tail_output_rev = tf.stack(tail_list_rev, axis=2)
    tail_output_rev = tf.reverse(tail_output_rev, tf.stack([1]))

    
    raw_output_all = tf.reduce_mean(tf.stack([head_output, tail_output_rev]), axis=0)
    raw_output_all = tf.expand_dims(raw_output_all, dim=0)
    raw_output_all = tf.argmax(raw_output_all, dimension=3)
    pred_all = tf.expand_dims(raw_output_all, dim=3) # Create 4-d tensor.

    # Which variables to load.
    restore_var = tf.global_variables()
    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    init = tf.global_variables_initializer()
    local = tf.local_variables_initializer()
  
  sess = tf.Session(config=config, graph=graph1)
  
  sess.run(init)
  sess.run(local)

  loader = tf.train.Saver(var_list=restore_var)
  if restore_from is not None:
    if load(loader, sess, restore_from):
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")
  threads = tf.train.start_queue_runners(coord=coord, sess=sess)

  for step in range(steps):
    parsing_ = sess.run(pred_all)
    
    img_split = image_list[step].split('/')
    image_id = img_split[-1][:-4]
    msk = decode_labels(parsing_, image_id, output_dir, num_classes=N_CLASSES)
    parsing_im = Image.fromarray(msk[0]) # parse to image from array
    
    parsing_im.save('{}/{}_vis.png'.format(output_dir, image_id))
    #cv2.imwrite('{}/{}.png'.format(output_dir, image_id), parsing_[0,:,:,0])
  
  sess.close()
  coord.request_stop()
  coord.join(threads)
def config_reader():
    config = configobj.ConfigObj('config')

    param = config['param']
    model_id = param['modelID']
    model = config['models'][model_id]
    model['boxsize'] = int(model['boxsize'])
    model['stride'] = int(model['stride'])
    model['padValue'] = int(model['padValue'])
    #param['starting_range'] = float(param['starting_range'])
    #param['ending_range'] = float(param['ending_range'])
    param['octave'] = int(param['octave'])
    param['use_gpu'] = int(param['use_gpu'])
    param['starting_range'] = float(param['starting_range'])
    param['ending_range'] = float(param['ending_range'])
    param['scale_search'] = list(map(float, param['scale_search']))
    param['thre1'] = float(param['thre1'])
    param['thre2'] = float(param['thre2'])
    param['thre3'] = float(param['thre3'])
    param['mid_num'] = int(param['mid_num'])
    param['min_num'] = int(param['min_num'])
    param['crop_ratio'] = float(param['crop_ratio'])
    param['bbox_ratio'] = float(param['bbox_ratio'])
    param['GPUdeviceNumber'] = int(param['GPUdeviceNumber'])

    return param, model
def process (img_id, input_image, params, model_params, userId, tag=""):

    oriImg = cv2.imread(input_image)  # B,G,R order
    multiplier = [x * model_params['boxsize'] / oriImg.shape[0] for x in params['scale_search']]

    heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
    paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))

    for m in range(len(multiplier)):
        scale = multiplier[m]

        imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        imageToTest_padded, pad = Pose_Estimation.util.padRightDownCorner(imageToTest, model_params['stride'],
                                                          model_params['padValue'])

        input_img = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,0,1,2)) # required shape (1, width, height, channels)

        output_blobs = pose_estimation_model.predict(input_img)

        # extract outputs, resize, and remove padding
        heatmap = np.squeeze(output_blobs[1])  # output 1 is heatmaps
        heatmap = cv2.resize(heatmap, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                             interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3],
                  :]
        heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        paf = np.squeeze(output_blobs[0])  # output 0 is PAFs
        paf = cv2.resize(paf, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                         interpolation=cv2.INTER_CUBIC)
        paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        paf = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        heatmap_avg = heatmap_avg + heatmap / len(multiplier)
        paf_avg = paf_avg + paf / len(multiplier)

    all_peaks = []
    peak_counter = 0

    for part in range(18):
        map_ori = heatmap_avg[:, :, part]
        map = gaussian_filter(map_ori, sigma=3)

        map_left = np.zeros(map.shape)
        map_left[1:, :] = map[:-1, :]
        map_right = np.zeros(map.shape)
        map_right[:-1, :] = map[1:, :]
        map_up = np.zeros(map.shape)
        map_up[:, 1:] = map[:, :-1]
        map_down = np.zeros(map.shape)
        map_down[:, :-1] = map[:, 1:]

        peaks_binary = np.logical_and.reduce(
            (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > params['thre1']))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
        peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)

    connection_all = []
    special_k = []
    mid_num = 10

    for k in range(len(mapIdx)):
        score_mid = paf_avg[:, :, [x - 19 for x in mapIdx[k]]]
        candA = all_peaks[limbSeq[k][0] - 1]
        candB = all_peaks[limbSeq[k][1] - 1]
        nA = len(candA)
        nB = len(candB)
        indexA, indexB = limbSeq[k]
        if (nA != 0 and nB != 0):
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                    # failure case when 2 body parts overlaps
                    if norm == 0:
                        continue
                    vec = np.divide(vec, norm)

                    startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                   np.linspace(candA[i][1], candB[j][1], num=mid_num)))

                    vec_x = np.array(
                        [score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                         for I in range(len(startend))])
                    vec_y = np.array(
                        [score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                         for I in range(len(startend))])

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                        0.5 * oriImg.shape[0] / norm - 1, 0)
                    criterion1 = len(np.nonzero(score_midpts > params['thre2'])[0]) > 0.8 * len(
                        score_midpts)
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append([i, j, score_with_dist_prior,
                                                     score_with_dist_prior + candA[i][2] + candB[j][2]])

            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            connection = np.zeros((0, 5))
            for c in range(len(connection_candidate)):
                i, j, s = connection_candidate[c][0:3]
                if (i not in connection[:, 3] and j not in connection[:, 4]):
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if (len(connection) >= min(nA, nB)):
                        break

            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])

    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall configuration
    subset = -1 * np.ones((0, 20))
    candidate = np.array([item for sublist in all_peaks for item in sublist])

    for k in range(len(mapIdx)):
        if k not in special_k:
            partAs = connection_all[k][:, 0]
            partBs = connection_all[k][:, 1]
            indexA, indexB = np.array(limbSeq[k]) - 1

            for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)):  # 1:size(subset,1):
                    if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1

                if found == 1:
                    j = subset_idx[0]
                    if (subset[j][indexB] != partBs[i]):
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2:  # if found 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else:  # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(20)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + \
                              connection_all[k][i][2]
                    subset = np.vstack([subset, row])

    # delete some rows of subset which has few parts occur
    deleteIdx = [];
    for i in range(len(subset)):
        if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)

    canvas = cv2.imread(input_image)  # B,G,R order
    for i in range(18):
        for j in range(len(all_peaks[i])):
            cv2.circle(canvas, all_peaks[i][j][0:2], 4, colors[i], thickness=-1)

    stickwidth = 4

    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0,
                                       360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    if tag=="upper":
        sio.savemat('{}/{}.mat'.format("../testdata/" + userId + "/input/upper_pose", img_id), {'candidate':candidate, 'subset':subset}, do_compression=True)
        print("Save upper pose success")
    elif tag == "lower":
        sio.savemat('{}/{}.mat'.format("../testdata/" + userId + "/input/body_pose", img_id), {'candidate':candidate, 'subset':subset}, do_compression=True)
        print("Save body pose success")
    
    
    return canvas
def pose_estimation(userId, pose_estimation_model, params, model_params):
  
  UPPER_IMAGE_PATH = "../testdata/" + userId + "/input/upper_images/"
  UPPER_RESULT_PATH = "../testdata/" + userId + "/input/upper_pose/"
  BODY_IMAGE_PATH = "../testdata/" + userId + "/input/body_images/"
  BODY_RESULT_PATH = "../testdata/" + userId + "/input/body_pose/"
  

  image_name_list = open("../testdata/" + userId + "/input/image.txt").read().splitlines()
  
  print('start processing...')
  for image_name in image_name_list:
    img1 = UPPER_IMAGE_PATH + image_name
    img2 = BODY_IMAGE_PATH + image_name
    img_id = image_name[:-4]
    canvas = process(img_id, img1, params, model_params, userId, tag="upper")
    canvas2 = process(img_id, img2, params, model_params, userId, tag="lower")
    output = UPPER_RESULT_PATH + img_id + ".png"
    cv2.imwrite(output, canvas)
    output = BODY_RESULT_PATH + img_id + ".png"
    cv2.imwrite(output, canvas2)
def _process_image(FLAGS, image_name, sess, tag="", resize_width=192,resize_height=256):
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
def write_image_pkl(userId, imageId):
  FLAGS = tf.app.flags.FLAGS
  tf.flags.DEFINE_string("text_of_image_dir", "../testdata/" + userId + "/input/image.txt",
                        "Directory text file")
  tf.flags.DEFINE_string("text_of_interval_dir", "../testdata/" + userId + "/input/interval_upper_data.txt",
                        "Directory text file")
  tf.flags.DEFINE_string("body_pkl_dir", "../testdata/" + userId + "/input/body_pickle/",
                        "Directory body pickle file")
  tf.flags.DEFINE_string("body_image_dir", "../testdata/" + userId + "/input/body_images/",
                        "")
  tf.flags.DEFINE_string("body_segment_dir", "../testdata/" + userId + "/input/body_segment/",
                        "Directory segment file")
  tf.flags.DEFINE_string("body_pose_dir", "../testdata/" + userId + "/input/body_pose/",
                        "Directory pose file")

  tf.flags.DEFINE_string("upper_pkl_dir", "../testdata/" + userId + "/input/upper_pickle/",
                        "Directory upper pickle file file")
  tf.flags.DEFINE_string("upper_image_dir", "../testdata/" + userId + "/input/upper_images/",
                        "")
  tf.flags.DEFINE_string("upper_segment_dir", "../testdata/" + userId + "/input/upper_segment/",
                        "Directory segment file")
  tf.flags.DEFINE_string("upper_pose_dir", "../testdata/" + userId + "/input/upper_pose/",
                        "Directory pose file")
  write_pkl_graph = tf.Graph()
  with write_pkl_graph.as_default() as graph2:
      interval_info = open(FLAGS.text_of_interval_dir).read().split()
      image_name_list = open(FLAGS.text_of_image_dir).read().split()
      interval = interval_info[1]
      resized_interval = interval_info[2]
      image_name = image_name_list[0]
      with tf.Session() as sess1:
        (image, pose_raw, body_segment,
            prod_segment, skin_segment) = _process_image(FLAGS, image_name, sess1, tag="upper")
        upper_data = _load_and_process_data(image, pose_raw, body_segment,
                                      prod_segment, skin_segment, interval, resized_interval, tag="upper")
      sess1.close()
      with tf.Session() as sess2:                                      
        (image2, pose_raw2, body_segment2,
            prod_segment2, skin_segment2) = _process_image(FLAGS, image_name, sess2, tag="lower")
        lower_data = _load_and_process_data(image2, pose_raw2, body_segment2,
                                      prod_segment2, skin_segment2, interval, resized_interval, tag="lower")
      sess2.close()
      with open(FLAGS.upper_pkl_dir + str(image_name[:-4]) + '.pkl', 'wb') as f:
          pickle.dump(upper_data, f, pickle.HIGHEST_PROTOCOL)
      with open(FLAGS.body_pkl_dir + str(image_name[:-4]) + '.pkl', 'wb') as f:
          pickle.dump(lower_data, f, pickle.HIGHEST_PROTOCOL)
  flags_dict = FLAGS._flags()  
  keys_list = [keys for keys in flags_dict]
  for keys in keys_list:
    FLAGS.__delattr__(keys)
	
  
def main(unused_argv):
  threading._start_new_thread(GetFlask,())
  '''
    make folder, write text file
  '''

  while True:
    while wait_queue.qsize() != 0:

      queue_data = wait_queue.get()
      print(queue_data)

      userId = queue_data[0]
      imageId = queue_data[1]
      
      input_dir = "../testdata/" + userId + "/input/"
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
      
      p1 = time.time()
      body_image_dir, data_list_dir = write_id(userId,imageId)
      p2 = time.time()
      evaluate_segment_parsing(userId)  
      p3 = time.time()
      image_control_process(userId)
      p4 = time.time()
      #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
      #os.environ["CUDA_VISIBLE_DEVICES"]="-1"
      pose_estimation(userId, pose_estimation_model, params, model_params)
      p5 = time.time()
      #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
      #os.environ["CUDA_VISIBLE_DEVICES"]="0"
      write_image_pkl(userId, imageId)
      p6 = time.time()
      print("       write id time: " + str(p2 - p1))
      print("segment parsing time: " + str(p3 - p2))
      print(" cropping image time: " + str(p4 - p3))
      print("pose estimation time: " + str(p5 - p4))
      print("write image pkl time: " + str(p6 - p5))

if __name__ == "__main__":
  userId = "category39"
  keras_weights_file = 'Pose_Estimation/model/keras/model.h5'
  pose_estimation_model = get_testing_model()
  pose_estimation_model.load_weights(keras_weights_file)
  params, model_params = config_reader()
  
  tf.app.run()