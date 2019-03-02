# Copyright 2017 Xintong Han. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

""" Test for Stage 1: from product image + body segment +
		pose + face/hair predict a coarse result and product segment.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import time

import numpy as np
import scipy.io as sio
import scipy.misc
import tensorflow as tf

from utils import *
from model_zalando_mask_content import create_model

import threading
import socket
import queue
import requests


import matlab.engine
from tps_transformer import tps_stn
from PIL import Image


### tps 합침
import tensorflow.contrib.slim as slim


wait_count = 0
wait_queue = queue.Queue()


total_start_time = time.time()
FLAGS = tf.app.flags.FLAGS
IMGCOUNT = 1
model_one = "12000"
model_two = "5000"
checkpoint_one = "model/stage1/model-" + model_one
checkpoint_two = "model/stage2/model-" + model_two
datadir = "data/women_top/"
resultdir_one = "results/stage1/"

wait_count = 0
wait_data = 0
sema = threading.Semaphore(2)


#g1 = tf.get_default_graph()
#g1.get_operations()
tf.flags.DEFINE_string("pose_dir", "data/pose/",
											 "Directory containing poses.")
tf.flags.DEFINE_string("segment_dir", "data/segment/",
											 "Directory containing human segmentations.")
tf.flags.DEFINE_string("image_dir", datadir,
											 "Directory containing product andto person images.")
tf.flags.DEFINE_string("test_label",
											 "data/viton_test_pairs.txt",
											 "File containing labels for testing.")
tf.flags.DEFINE_string("result_dir", resultdir_one,
											 "Folder containing the results of testing.")
tf.flags.DEFINE_string("coarse_result_dir", "results/stage1",
									"Folder containing the results of stage1 (coarse) results.")

tf.flags.DEFINE_integer("begin", "0", "")
tf.flags.DEFINE_integer("end", IMGCOUNT, "")




tf.logging.set_verbosity(tf.logging.INFO)




eng = matlab.engine.start_matlab()



# preprocess images for testing
def _process_image(image_name, product_image_name, sess,
									 resize_width=192, resize_height=256):
	image_id = image_name[:-4]
	
	step_one_before_time = time.time()
	

	image = scipy.misc.imread(FLAGS.image_dir + image_name)
	prod_image = scipy.misc.imread(FLAGS.image_dir + product_image_name)
	segment_raw = sio.loadmat(os.path.join(
			FLAGS.segment_dir, image_id))["segment"]
	segment_raw = process_segment_map(segment_raw, image.shape[0], image.shape[1])
	pose_raw = sio.loadmat(os.path.join(FLAGS.pose_dir, image_id))
	pose_raw = extract_pose_keypoints(pose_raw)
	pose_raw = extract_pose_map(pose_raw, image.shape[0], image.shape[1])
	pose_raw = np.asarray(pose_raw, np.float32)

	body_segment, prod_segment, skin_segment = extract_segmentation(segment_raw)



	step_two_after_time = time.time()
	print(str( step_two_after_time - step_one_before_time ) + "초 걸렸습니다 [step1]")

	image = tf.image.convert_image_dtype(image, dtype=tf.float32)
	prod_image = tf.image.convert_image_dtype(prod_image, dtype=tf.float32)

	image = tf.image.resize_images(image,
																 size=[resize_height, resize_width],
																 method=tf.image.ResizeMethod.BILINEAR)
	prod_image = tf.image.resize_images(prod_image,
																			size=[resize_height, resize_width],
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
	prod_image = (prod_image - 0.5) * 2.0

	step_three_after_time = time.time()
	print(str( step_three_after_time - step_two_after_time ) + "초 걸렸습니다 [step3]")

	# using skin rbg
	skin_segment = skin_segment * image

	[image, prod_image, body_segment, prod_segment, skin_segment] = sess.run(
			[image, prod_image, body_segment, prod_segment, skin_segment])

	step_four_after_time = time.time()
	print(str( step_four_after_time - step_three_after_time ) + "초 걸렸습니다 [step4]")

	return image, prod_image, pose_raw, body_segment, prod_segment, skin_segment

#*************************step2를 위해*******************************

def deprocess_image(image, mask01=False):
  if not mask01:
    image = image / 2 + 0.5
  return image

def process_one_image(image, resize_height, resize_width, if_zero_one=False):
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  if if_zero_one:
    return image
  image = tf.image.resize_images(image,
                                 size=[resize_height, resize_width],
                                 method=tf.image.ResizeMethod.BILINEAR)
  return (image - 0.5) * 2.0

def _process_image_2(image_name, product_image_name, sess,
                   resize_width=192, resize_height=256):
  image_id = image_name[:-4]
  image = scipy.misc.imread(FLAGS.image_dir + image_name)
  prod_image = scipy.misc.imread(FLAGS.image_dir + product_image_name)
  # sorry for the hard coded file path.
  print(FLAGS.coarse_result_dir)
  print(image_name)
  print(product_image_name)
  print("!@#!@#\n")
  coarse_image = scipy.misc.imread(FLAGS.coarse_result_dir +
                                   "/images/00012000_" +
                                   image_name + "_" +
                                   product_image_name + ".png")
  #scipy.misc.imshow(coarse_image)
  print(type(coarse_image))
  print(coarse_image.shape)
  print(coarse_image.dtype)
  mask_output = scipy.misc.imread(FLAGS.coarse_result_dir +
                                  "/images/00012000_" +
                                  image_name + "_" +
                                  product_image_name + "_mask.png")
  print(mask_output.shape)
  print(mask_output.dtype)
  #why black????

  image = process_one_image(image, resize_height, resize_width)
  prod_image = process_one_image(prod_image, resize_height, resize_width)
  coarse_image = process_one_image(coarse_image, resize_height, resize_width)
  mask_output = process_one_image(mask_output, resize_height,
                                  resize_width, True)
  # TPS transform
  # Here we use control points to generate 
  # We tried to learn the control points, but the network refuses to converge.
  tps_control_points = sio.loadmat(FLAGS.coarse_result_dir +
                                   "/tps/00012000_" +
                                   image_name + "_" +
                                   product_image_name +
                                   "_tps.mat")

  v = tps_control_points["control_points"]
  print(len(v))
  nx = v.shape[1]
  ny = v.shape[2]
  v = np.reshape(v, -1)
  v = np.transpose(v.reshape([1,2,nx*ny]), [0,2,1]) * 2 -1
  p = tf.convert_to_tensor(v, dtype=tf.float32)
  img = tf.reshape(prod_image, [1,256,192,3])

  tps_image = tps_stn(img, nx, ny, p, [256,192,3])

  tps_mask = tf.cast(tf.less(tf.reduce_sum(tps_image, -1), 3*0.95), tf.float32)

  [image, prod_image, coarse_image, tps_image, mask_output, tps_mask] = sess.run(
              [image, prod_image, coarse_image, tps_image, mask_output, tps_mask])

  return image, prod_image, coarse_image, tps_image, mask_output, tps_mask

#********************************************************


#리스트를 받아서
def GetFlask():
	print("GETFLASK")
	s = socket.socket()
	host = socket.gethostname()
	port = 12222
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


#가져옴
def create_refine_generator(stn_image_outputs, gen_image_outputs):
  generator_input = tf.concat([stn_image_outputs, gen_image_outputs],
                               axis=-1)

  downsampled = tf.image.resize_area(generator_input,
                                     (256, 192),
                                     align_corners=False)
  net = slim.conv2d(downsampled, 64, [3, 3], rate=1, normalizer_fn=slim.layer_norm,
                      activation_fn=lrelu, scope='g_256_conv1')
  net = slim.conv2d(net, 64, [3, 3], rate=1, normalizer_fn=slim.layer_norm,
                      activation_fn=lrelu, scope='g_256_conv2')
  net = slim.conv2d(net, 64, [3, 3], rate=1, normalizer_fn=slim.layer_norm,
                      activation_fn=lrelu, scope='g_256_conv3')

  # output a selection mask:
  net = slim.conv2d(net, 1, [1, 1], rate=1,
                    activation_fn=None, scope='g_1024_final')
  net = tf.sigmoid(net)
  return net  


def main(unused_argv):

	



	try:
		os.mkdir(FLAGS.result_dir)
	except:
		pass
	try:
		os.mkdir(FLAGS.result_dir + "/images/")
	except:
		pass
	try:
		os.mkdir(FLAGS.result_dir + "/tps/")
	except:
		pass

	# batch inference, can also be done one image per time.
	batch_size = 1
	
	g1 = tf.Graph()
	with g1.as_default() as graph:
		image_holder_one = tf.placeholder(tf.float32, shape=[batch_size, 256, 192, 3],name="Placeholder_9")
		prod_image_holder_one = tf.placeholder(
				tf.float32, shape=[batch_size, 256, 192, 3],name="Placeholder_6")
		body_segment_holder_one = tf.placeholder(
				tf.float32, shape=[batch_size, 256, 192, 1])
		prod_segment_holder_one = tf.placeholder(
				tf.float32, shape=[batch_size, 256, 192, 1])
		skin_segment_holder_one = tf.placeholder(
				tf.float32, shape=[batch_size, 256, 192, 3])
		pose_map_holder_one = tf.placeholder(tf.float32, shape=[batch_size, 256, 192, 18])

		model = create_model(prod_image_holder_one, body_segment_holder_one,
												 skin_segment_holder_one, pose_map_holder_one,
												 prod_segment_holder_one, image_holder_one)

		images_one = np.zeros((batch_size, 256, 192, 3))
		prod_images_one = np.zeros((batch_size, 256, 192, 3))
		body_segments_one = np.zeros((batch_size, 256, 192, 1))
		prod_segments_one = np.zeros((batch_size, 256, 192, 1))
		skin_segments_one = np.zeros((batch_size, 256, 192, 3))
		pose_raws_one = np.zeros((batch_size, 256, 192, 18))
	


	#tf.train.update_checkpoint_state("model/stage1","model/stage1")
		saver_one = tf.train.Saver()
		sess_one = tf.Session(graph = g1)
		print("loading model from checkpoint")
		checkpoint = tf.train.latest_checkpoint(checkpoint_one)
		if checkpoint == None:
			checkpoint = checkpoint_one
		print(checkpoint)

		saver_one.restore(sess_one, checkpoint)
	
	print("EEE")
	# reading input data
	#test_info = open(FLAGS.test_label).read().splitlines()

	##########stage2 Model Load###############
	#from model_zalando_tps_warp import create_refine_generator
	flags_dict = FLAGS._flags()    
	keys_list = [keys for keys in flags_dict]    
	for keys in keys_list:    
		FLAGS.__delattr__(keys)

	#############tps wrap

	tf.flags.DEFINE_string("input_file_pattern",
	                       "./prepare_data/tfrecord/zalando-train-?????-of-00032",
	                       "File pattern of sharded TFRecord input files.")
	tf.flags.DEFINE_string("mode", "train", "Training or testing")
	tf.flags.DEFINE_string("checkpoint", "", "Checkpoint path to resume training.")
	tf.flags.DEFINE_string("gen_checkpoint", "",
	                       "Checkpoint path to the initial generative model.")
	tf.flags.DEFINE_string("output_dir", "model/stage2/",
	                       "Output directory of images.")
	tf.flags.DEFINE_string("vgg_model_path", "./model/imagenet-vgg-verydeep-19.mat",
	                       "model of the trained vgg net.")

	tf.flags.DEFINE_integer("number_of_steps", 100000,
	                        "Number of training steps.")
	tf.flags.DEFINE_integer("log_every_n_steps", 10,
	                        "Frequency at which loss and global step are logged.")
	tf.flags.DEFINE_integer("batch_size", 8, "Size of mini batch.")
	tf.flags.DEFINE_integer("num_preprocess_threads", 1, "")
	tf.flags.DEFINE_integer("values_per_input_shard", 433, "")
	tf.flags.DEFINE_integer("ngf", 64,
	                        "number of generator filters in first conv layer")
	tf.flags.DEFINE_integer("ndf", 64,
	                        "number of discriminator filters in first conv layer")
	# Summary
	tf.flags.DEFINE_integer("summary_freq", 50, #100
	                        "update summaries every summary_freq steps")
	tf.flags.DEFINE_integer("progress_freq", 10, #100
	                        "display progress every progress_freq steps")
	tf.flags.DEFINE_integer("trace_freq", 0,
	                        "trace execution every trace_freq steps")
	tf.flags.DEFINE_integer("display_freq", 50, #300
	                        "write current training images every display_freq steps")
	tf.flags.DEFINE_integer("save_freq", 5000,
	                        "save model every save_freq steps, 0 to disable")

	tf.flags.DEFINE_float("number_of_samples", 1500.0, "Samples in training set.")
	tf.flags.DEFINE_float("lr", 0.0002, "Initial learning rate.")
	tf.flags.DEFINE_float("beta1", 0.5, "momentum term of adam")
	tf.flags.DEFINE_float("content_l1_weight", 0.2, "Weight on L1 term of content.")
	tf.flags.DEFINE_float("perceptual_weight", 0.8, "weight on GAN term.")
	tf.flags.DEFINE_float("tv_weight", 0.000005, "weight on TV term.")
	tf.flags.DEFINE_float("mask_weight", 0.1, "weight on the selection mask.")

	tf.flags.DEFINE_string("image_dir", datadir,
													 "Directory containing product and person images.")
	tf.flags.DEFINE_string("test_label",
												 "data/viton_test_pairs.txt",
												 "File containing labels for testing.")
	tf.flags.DEFINE_string("result_dir", "results/stage2/",
												 "Folder containing the results of testing.")
	tf.flags.DEFINE_string("coarse_result_dir", "results/stage1",
										"Folder containing the results of stage1 (coarse) results.")

	tf.flags.DEFINE_integer("begin", "0", "")
	tf.flags.DEFINE_integer("end", IMGCOUNT, "")

	g2 = tf.Graph()


	with g2.as_default() as graph:
		image_holder = tf.placeholder(tf.float32, shape=[batch_size,256,192,3])
		prod_image_holder = tf.placeholder(tf.float32, shape=[batch_size,256,192,3])
		prod_mask_holder = tf.placeholder(tf.float32, shape=[batch_size,256,192,1])
		coarse_image_holder = tf.placeholder(tf.float32, shape=[batch_size,256,192,3])
		tps_image_holder = tf.placeholder(tf.float32, shape=[batch_size,256,192,3])
		#'''

		#1016 옮김

		with tf.variable_scope("refine_generator") as scope:
			select_mask = create_refine_generator(tps_image_holder,coarse_image_holder)
			select_mask = select_mask * prod_mask_holder
			model_image_outputs = (select_mask * tps_image_holder + (1 - select_mask) * coarse_image_holder)

		saver_two = tf.train.Saver(var_list=[var for var in tf.trainable_variables()  if var.name.startswith("refine_generator")])
		sess_two = tf.Session(graph = g2)
		print("loading model from checkpoint")

			#checkpoint 오류를 없애기 위해 추가
			#tf.train.update_checkpoint_state("model/stage2","model/stage2")

		checkpoint = tf.train.latest_checkpoint(checkpoint_two)
		if checkpoint == None:
			checkpoint = checkpoint_two
		print(checkpoint)
		
		saver_two.restore(sess_two, checkpoint)




	threading._start_new_thread(GetFlask,())
	is_run=0

	

	count = 0
	while True: #기다리며 합성할 거 생기면 계속 처리


		#try:
		while wait_queue.qsize() != 0: # 합성할 게 없을 경우 위의 루프로 넘어감

			count = count + 1			
			if count > 0:
				print("초기화")
				flags_dict = FLAGS._flags()    
				keys_list = [keys for keys in flags_dict]    
				for keys in keys_list:    
					FLAGS.__delattr__(keys)
				#tf.reset_default_graph()
				###P1


				tf.flags.DEFINE_string("pose_dir", "data/pose/",
												 "Directory containing poses.")
				tf.flags.DEFINE_string("segment_dir", "data/segment/",
															 "Directory containing human segmentations.")
				tf.flags.DEFINE_string("image_dir", datadir,
															 "Directory containing product andto person images.")
				tf.flags.DEFINE_string("test_label",
															 "data/viton_test_pairs.txt",
															 "File containing labels for testing.")
				tf.flags.DEFINE_string("result_dir", resultdir_one,
															 "Folder containing the results of testing.")
				tf.flags.DEFINE_string("coarse_result_dir", "results/stage1",
													"Folder containing the results of stage1 (coarse) results.")

				#####
				
				tf.flags.DEFINE_string("input_file_pattern",
	                       "./prepare_data/tfrecord/zalando-train-?????-of-00032",
	                       "File pattern of sharded TFRecord input files.")
				tf.flags.DEFINE_string("mode", "train", "Training or testing")
				tf.flags.DEFINE_string("checkpoint", "", "Checkpoint path to resume training.")
				tf.flags.DEFINE_string("output_dir", "model/stage1/",
				                       "Output directory of images.")
				tf.flags.DEFINE_string("vgg_model_path", "./model/imagenet-vgg-verydeep-19.mat",
				                       "model of the trained vgg net.")

				tf.flags.DEFINE_integer("number_of_steps", 100000,
				                        "Number of training steps.")
				tf.flags.DEFINE_integer("log_every_n_steps", 10,
				                        "Frequency at which loss and global step are logged.")
				tf.flags.DEFINE_integer("batch_size", 8, "Size of mini batch.")
				tf.flags.DEFINE_integer("num_preprocess_threads", 1, "")
				tf.flags.DEFINE_integer("values_per_input_shard", 443, "")
				tf.flags.DEFINE_integer("ngf", 64,
				                        "number of generator filters in first conv layer")
				tf.flags.DEFINE_integer("ndf", 64,
				                        "number of discriminator filters in first conv layer")
				# Summary
				tf.flags.DEFINE_integer("summary_freq", 100,
				                        "update summaries every summary_freq steps")
				tf.flags.DEFINE_integer("progress_freq", 10,
				                        "display progress every progress_freq steps")
				tf.flags.DEFINE_integer("trace_freq", 0,
				                        "trace execution every trace_freq steps")
				tf.flags.DEFINE_integer("display_freq", 300,
				                        "write current training images every display_freq steps")
				tf.flags.DEFINE_integer("save_freq", 3000,
				                        "save model every save_freq steps, 0 to disable")

				# Weights
				tf.flags.DEFINE_float("mask_offset", 1.0, "Weight mask is emphasized.")
				tf.flags.DEFINE_float("number_of_samples", 1500.0, "Samples in training set.")
				tf.flags.DEFINE_float("lr", 0.0002, "Initial learning rate.")
				tf.flags.DEFINE_float("beta1", 0.5, "momentum term of adam")
				tf.flags.DEFINE_float("mask_l1_weight", 1.0, "Weight on L1 term of product mask.")
				tf.flags.DEFINE_float("content_l1_weight", 1.0, "Weight on L1 term of content.")
				tf.flags.DEFINE_float("perceptual_weight", 3.0, "weight on GAN term.")
				


				###

				tf.flags.DEFINE_integer("begin", "0", "")
				tf.flags.DEFINE_integer("end", IMGCOUNT, "")
				tf.logging.set_verbosity(tf.logging.INFO)


			with g1.as_default() as graph:


				print("StarT")
				step_one_before_time = time.time()

				print(wait_queue.qsize())
		# loading batch data
				
				queue_data = wait_queue.get()
				if queue_data == "exit":
					return 0

				test_info = open(FLAGS.test_label).read().splitlines()
				for i in range(FLAGS.begin, FLAGS.end, batch_size):
					image_names = []
					product_image_names = []

					print("1")
					for j in range(i, i + batch_size):
						info = test_info[j].split()
						print(queue_data)
						image_name = queue_data[0] + "_0.jpg"
						product_image_name = queue_data[1]  + "_1.jpg"#큐의 데이터를 이미지 각각 넣어줌
						image_names.append(image_name)
						product_image_names.append(product_image_name)
						print("2")
						(image_one, prod_image_one, pose_raw_one,
						 body_segment_one, prod_segment_one,
						 skin_segment_one) = _process_image(image_name,product_image_name, sess_one)
						images_one[j-i] = image_one
						prod_images_one[j-i] = prod_image_one
						body_segments_one[j-i] = body_segment_one
						prod_segments_one[j-i] = prod_segment_one
						skin_segments_one[j-i] = skin_segment_one
						pose_raws_one[j-i] = pose_raw_one
					print("3")
					# inference

					feed_dict_one = {
							image_holder_one: images_one,
							prod_image_holder_one: prod_images_one,
							body_segment_holder_one: body_segments_one,
							skin_segment_holder_one: skin_segments_one,
							prod_segment_holder_one: prod_segments_one,
							pose_map_holder_one: pose_raws_one,
					}

					

					[image_output, mask_output, loss, step] = sess_one.run(
							[model.image_outputs,
							 model.mask_outputs,
							 model.gen_loss_content_L1,
							 model.global_step],
							feed_dict=feed_dict_one)
				###P1

				# write results
				for j in range(batch_size):
					scipy.misc.imsave(FLAGS.result_dir + ("images/%08d_" % step) +
														image_names[j] + "_" + product_image_names[j] + '.png',
														(image_output[j] / 2.0 + 0.5))
					scipy.misc.imsave(FLAGS.result_dir + ("images/%08d_" % step) +
														image_names[j] + "_" + product_image_names[j] + '_mask.png',
														np.squeeze(mask_output[j]))
					scipy.misc.imsave(FLAGS.result_dir + "images/" +
														image_names[j], (images_one[j] / 2.0 + 0.5))
					scipy.misc.imsave(FLAGS.result_dir + "images/" +
														product_image_names[j], (prod_images_one[j] / 2.0 + 0.5))
					sio.savemat(FLAGS.result_dir + "/tps/" + ("%08d_" % step) +
											image_names[j] + "_" + product_image_names[j] + "_mask.mat",
											{"mask": np.squeeze(mask_output[j])})

				# write html
				index_path = os.path.join(FLAGS.result_dir, "index.html")
				if os.path.exists(index_path):
					index = open(index_path, "a")
				else:
					index = open(index_path, "w")
					index.write("<html><body><table><tr>")
					index.write("<th>step</th>")
					index.write("<th>name</th><th>input</th>"
											"<th>output</th><th>target</th></tr>")
				for j in range(batch_size):
					index.write("<tr>")
					index.write("<td>%d %d</td>" % (step, i + j))
					index.write("<td>%s %s</td>" % (image_names[j], product_image_names[j]))
					index.write("<td><img src='images/%s'></td>" % image_names[j])
					index.write("<td><img src='images/%s'></td>" % product_image_names[j])
					index.write("<td><img src='images/%08d_%s'></td>" %
											(step, image_names[j] + "_" + product_image_names[j] + '.png'))
					index.write("<td><img src='images/%08d_%s'></td>" %
											(step, image_names[j] + "_" + product_image_names[j] + '_mask.png'))
					index.write("</tr>")
				index.close()
				print("End")
				print(wait_queue.qsize())
				#except:
				#	print("ERROR")
				##웹 서버로 결과 이미지 전송

				result_file = open(FLAGS.result_dir + ("images/%08d_" % step) + image_names[0] + "_" + product_image_names[0] + '.png','rb')
				upload = {'fileToUpload':result_file}
				user_test = "3"
				product_test = "1535187668"


				#obj = {'userid':image_name[:-5], 'productid':product_image_name[:-5]}
				obj = {'userid':user_test, 'productid':product_test}
				print(obj)
				res = requests.post('http://211.253.229.68/get_res.php',files=upload,data=obj)
				print(str(res)+"######")
				print(res.text)
			f = open("data/viton_test_pairs.txt",'w')
			f.write(image_name + " " + product_image_name)
			f.close()

			print("1차 처리 완료")

			# 메인 1차 끝
			step_one_after_time = time.time()
			print(str( step_one_after_time - step_one_before_time ) + "초 걸렸습니다 [step1]")


			eng.shape_context_warp(nargout=0)
			print("매트랩 처리 완료")	
			matlab_after_time = time.time()
			print(str(matlab_after_time- step_one_after_time) + "초 걸렸습니다 [matlab]" )
			
			flags_dict = FLAGS._flags()    
			keys_list = [keys for keys in flags_dict]    
			for keys in keys_list:    
				FLAGS.__delattr__(keys)
			
			#tf.reset_default_graph()
			###P1

			

			tf.flags.DEFINE_string("image_dir", datadir,
													 "Directory containing product and person images.")
			tf.flags.DEFINE_string("test_label",
														 "data/viton_test_pairs.txt",
														 "File containing labels for testing.")
			tf.flags.DEFINE_string("result_dir", "results/stage2/",
														 "Folder containing the results of testing.")
			tf.flags.DEFINE_string("coarse_result_dir", "results/stage1",
												"Folder containing the results of stage1 (coarse) results.")

			tf.flags.DEFINE_integer("begin", "0", "")
			tf.flags.DEFINE_integer("end", IMGCOUNT, "")

			if count > 1:
				tf.flags.DEFINE_string("checkpoint", "", "Checkpoint path to resume training.")
			#####Q1
			

			#tf.flags.DEFINE_string("checkpoint", "", "Checkpoint path to resume training.")
			'''
			tf.flags.DEFINE_string("gen_checkpoint", "",
			                       "Checkpoint path to the initial generative model.")
			tf.flags.DEFINE_string("output_dir", "model/stage2/",
			                       "Output directory of images.")
			tf.flags.DEFINE_string("vgg_model_path", "./model/imagenet-vgg-verydeep-19.mat",
			                       "model of the trained vgg net.")

			tf.flags.DEFINE_integer("number_of_steps", 100000,
			                        "Number of training steps.")
			tf.flags.DEFINE_integer("log_every_n_steps", 10,
			                        "Frequency at which loss and global step are logged.")
			tf.flags.DEFINE_integer("batch_size", 8, "Size of mini batch.")
			tf.flags.DEFINE_integer("num_preprocess_threads", 1, "")
			tf.flags.DEFINE_integer("values_per_input_shard", 433, "")
			tf.flags.DEFINE_integer("ngf", 64,
			                        "number of generator filters in first conv layer")
			tf.flags.DEFINE_integer("ndf", 64,
			                        "number of discriminator filters in first conv layer")
			# Summary
			tf.flags.DEFINE_integer("summary_freq", 50, #100
			                        "update summaries every summary_freq steps")
			tf.flags.DEFINE_integer("progress_freq", 10, #100
			                        "display progress every progress_freq steps")
			tf.flags.DEFINE_integer("trace_freq", 0,
			                        "trace execution every trace_freq steps")
			tf.flags.DEFINE_integer("display_freq", 50, #300
			                        "write current training images every display_freq steps")
			tf.flags.DEFINE_integer("save_freq", 5000,
			                        "save model every save_freq steps, 0 to disable")

			tf.flags.DEFINE_float("number_of_samples", 1500.0, "Samples in training set.")
			tf.flags.DEFINE_float("lr", 0.0002, "Initial learning rate.")
			tf.flags.DEFINE_float("beta1", 0.5, "momentum term of adam")
			tf.flags.DEFINE_float("content_l1_weight", 0.2, "Weight on L1 term of content.")
			tf.flags.DEFINE_float("perceptual_weight", 0.8, "weight on GAN term.")
			tf.flags.DEFINE_float("tv_weight", 0.000005, "weight on TV term.")
			tf.flags.DEFINE_float("mask_weight", 0.1, "weight on the selection mask.")


			###'''
			

			with g2.as_default() as graph:

				tf.logging.set_verbosity(tf.logging.INFO)
				

				#safe
				try:
					os.mkdir(FLAGS.result_dir)
				except:
					pass
				try:
						os.mkdir(FLAGS.result_dir + "/images/")
				except:
					pass


				batch_size = 1
				#here is problem
				# Feed into the refine module
				'''
				image_holder = tf.placeholder(tf.float32, shape=[batch_size,256,192,3])
				prod_image_holder = tf.placeholder(tf.float32, shape=[batch_size,256,192,3])
				prod_mask_holder = tf.placeholder(tf.float32, shape=[batch_size,256,192,1])
				coarse_image_holder = tf.placeholder(tf.float32, shape=[batch_size,256,192,3])
				tps_image_holder = tf.placeholder(tf.float32, shape=[batch_size,256,192,3])
				#

				#1016 옮김

				with tf.variable_scope("refine_generator") as scope:
					select_mask = create_refine_generator(tps_image_holder,coarse_image_holder)
					select_mask = select_mask * prod_mask_holder
					model_image_outputs = (select_mask * tps_image_holder + (1 - select_mask) * coarse_image_holder)

				saver = tf.train.Saver(var_list=[var for var in tf.trainable_variables()  if var.name.startswith("refine_generator")])
				with tf.Session() as sess:
					print("loading model from checkpoint")

					#checkpoint 오류를 없애기 위해 추가
					#tf.train.update_checkpoint_state("model/stage2","model/stage2")

					checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint)
					if checkpoint == None:
						checkpoint = FLAGS.checkpoint
					print(checkpoint)
					
					saver.restore(sess, checkpoint)'''
				step_two_before_time = time.time()

				# reading input data
				test_info = open(FLAGS.test_label).read().splitlines()
				for i in range(FLAGS.begin, FLAGS.end, batch_size):
					# loading batch data
					print(i)
					images = np.zeros((batch_size,256,192,3))
					prod_images = np.zeros((batch_size,256,192,3))
					coarse_images = np.zeros((batch_size,256,192,3))
					tps_images = np.zeros((batch_size,256,192,3))
					mask_outputs = np.zeros((batch_size,256,192,1))

					image_names = []
					product_image_names = []

					for j in range(i, i + batch_size):
						info = test_info[j].split()
						print(info)
						image_name = info[0]
						product_image_name = info[1]
						image_names.append(image_name)
						product_image_names.append(product_image_name)
						#try:
						(image, prod_image, coarse_image,
							 tps_image, mask_output, tps_mask) = _process_image_2(image_name,
																											 product_image_name, sess_two)
						#except:
						#	print("PROCESS ERROR")
						#	continue

						images[j-i] = image
						prod_images[j-i] = prod_image
						coarse_images[j-i] = coarse_image
						tps_images[j-i] = tps_image
						mask_outputs[j-i] = np.expand_dims(mask_output, -1)

					# inference
					feed_dict2 = {
						image_holder: images,
						prod_image_holder: prod_images,
						coarse_image_holder: coarse_images,
						tps_image_holder: tps_images,
						prod_mask_holder: mask_outputs,
					}

					[image_output, sel_mask] = sess_two.run([model_image_outputs, select_mask],
																		feed_dict=feed_dict2)

				# write results
				for j in range(batch_size):
					step = 0
					scipy.misc.imsave(FLAGS.result_dir + "images/" + image_names[j] +
														"_" + product_image_names[j] + '_tps.png',
														(tps_images[j] / 2.0 + 0.5))
					scipy.misc.imsave(FLAGS.result_dir + "images/" + image_names[j] +
														"_" + product_image_names[j] + '_coarse.png',
														(coarse_images[j] / 2.0 + 0.5))
					scipy.misc.imsave(FLAGS.result_dir + "images/" + image_names[j] +
														"_" + product_image_names[j] + '_mask.png',
														np.squeeze(mask_outputs[j]))
					scipy.misc.imsave(FLAGS.result_dir + "images/" + image_names[j] +
														"_" + product_image_names[j] + '_final.png',
														(image_output[j]) / 2.0 + 0.5)
					scipy.misc.imsave(FLAGS.result_dir + "images/" + image_names[j] +
														"_" + product_image_names[j] + '_sel_mask.png',
														np.squeeze(sel_mask[j]))
					scipy.misc.imsave(FLAGS.result_dir + "images/" + image_names[j],
														(images[j] / 2.0 + 0.5))
					scipy.misc.imsave(FLAGS.result_dir + "images/"+ product_image_names[j],
														(prod_images[j] / 2.0 + 0.5))

				# write html
				index_path = os.path.join(FLAGS.result_dir, "index.html")
				if os.path.exists(index_path):
					index = open(index_path, "a")
				else:
					index = open(index_path, "w")
					index.write("<html><body><table><tr>")
					index.write("<th>step</th>")
					index.write("<th>name</th><th>input</th>"
						"<th>output</th><th>target</th></tr>")
				for j in range(batch_size):
					index.write("<tr>")
					index.write("<td>%d %d</td>" % (step, i + j))
					index.write("<td>%s %s</td>" % (image_names[j],
																						product_image_names[j]))
					index.write("<td><img src='images/%s'></td>" % image_names[j])
					index.write("<td><img src='images/%s'></td>" % product_image_names[j])
					index.write("<td><img src='images/%s'></td>" % 
						 (image_names[j] + "_" + product_image_names[j] + '_tps.png'))
					index.write("<td><img src='images/%s'></td>" % 
						(image_names[j] + "_" + product_image_names[j] + '_coarse.png'))
					index.write("<td><img src='images/%s'></td>" % 
						 (image_names[j] + "_" + product_image_names[j] + '_mask.png'))
					index.write("<td><img src='images/%s'></td>" % 
						 (image_names[j] + "_" + product_image_names[j] + '_final.png'))
					index.write("<td><img src='images/%s'></td>" % 
						 (image_names[j] + "_" + product_image_names[j] + '_sel_mask.png'))
					index.write("</tr>")
				index.close()
			step_two_after_time = time.time()
			print(str(step_two_after_time- step_two_before_time) + "초 걸렸습니다 [step2]")
			print(str(step_two_after_time - step_one_before_time) + "초 걸렸습니다 [step1~2]")
			print(str( step_one_after_time- total_start_time) + "초 걸렸습니다 [통합]")  
		#'''

				
		#except Exception as ex:
		#	print("ERROR",ex)
		#	return 0



if __name__ == "__main__":
	tf.app.run()
