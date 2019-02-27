from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import math
import os
import time
import datetime

from utils import *

import numpy as np
import scipy.io as sio
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("mode", "test", "Training or testing")
tf.flags.DEFINE_string("vgg_model_path", "./model/imagenet-vgg-verydeep-19.mat",
                       "model of the trained vgg net.")

tf.flags.DEFINE_integer("ngf", 64,
                        "number of generator filters in first conv layer")

tf.flags.DEFINE_float("lr", 0.0002, "Initial learning rate.")
tf.flags.DEFINE_float("beta1", 0.5, "momentum term of adam")
tf.flags.DEFINE_float("mask_l1_weight", 1.0, "Weight on L1 term of product mask.")
tf.flags.DEFINE_float("content_l1_weight", 1.0, "Weight on L1 term of content.")
tf.flags.DEFINE_float("perceptual_weight", 3.0, "weight on GAN term.")

tf.logging.set_verbosity(tf.logging.INFO)


Model = collections.namedtuple("Model",
                               "mask_outputs, image_outputs,"
                               "gen_loss_GAN, gen_loss_mask_L1,"
                               "gen_loss_content_L1, perceptual_loss,"
                               "train, global_step")
def is_training():
    return FLAGS.mode == "train"
def create_generator(product_image, body_seg, skin_seg,
                     pose_map, generator_outputs_channels):
  """ Generator from product images, segs, poses to a segment map"""
  # Build inputs
  generator_inputs = tf.concat([product_image, body_seg, skin_seg, pose_map],
                                axis=-1)
  layers = []

  # encoder_1: [batch, 256, 192, in_channels] => [batch, 128, 96, ngf]
  with tf.variable_scope("encoder_1"):
    output = conv(generator_inputs, FLAGS.ngf, stride=2)
    layers.append(output)

  layer_specs = [
      # encoder_2: [batch, 128, 96, ngf] => [batch, 64, 48, ngf * 2]
      FLAGS.ngf * 2,
      # encoder_3: [batch, 64, 48, ngf * 2] => [batch, 32, 24, ngf * 4]
      FLAGS.ngf * 4,
      # encoder_4: [batch, 32, 24, ngf * 4] => [batch, 16, 12, ngf * 8]
      FLAGS.ngf * 8,
      # encoder_5: [batch, 16, 12, ngf * 8] => [batch, 8, 6, ngf * 8]
      FLAGS.ngf * 8,
      # encoder_6: [batch, 8, 6, ngf * 8] => [batch, 4, 3, ngf * 8]
      FLAGS.ngf * 8,
  ]

  for out_channels in layer_specs:
    with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
      rectified = lrelu(layers[-1], 0.2)
      # [batch, in_height, in_width, in_channels]
      # => [batch, in_height/2, in_width/2, out_channels]
      convolved = conv(rectified, out_channels, stride=2)
      output = batch_norm(convolved, is_training())
      layers.append(output)

  layer_specs = [
      # decoder_6: [batch, 4, 3, ngf * 8 * 2] => [batch, 8, 6, ngf * 8 * 2]
      (FLAGS.ngf * 8, 0.5),
      # decoder_5: [batch, 8, 12, ngf * 8 * 2] => [batch, 16, 12, ngf * 8 * 2]
      (FLAGS.ngf * 8, 0.0),
      # decoder_4: [batch, 16, 12, ngf * 8 * 2] => [batch, 32, 24, ngf * 4 * 2]
      (FLAGS.ngf * 4, 0.0),
      # decoder_3: [batch, 32, 24, ngf * 4 * 2] => [batch, 64, 48, ngf * 2 * 2]
      (FLAGS.ngf * 2, 0.0),
      # decoder_2: [batch, 64, 48, ngf * 2 * 2] => [batch, 128, 96, ngf * 2]
      (FLAGS.ngf, 0.0),
  ]

  num_encoder_layers = len(layers)
  for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
    skip_layer = num_encoder_layers - decoder_layer - 1
    with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
      if decoder_layer == 0:
        # first decoder layer doesn't have skip connections
        # since it is directly connected to the skip_layer
        input = layers[-1]
      else:
        input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

      rectified = tf.nn.relu(input)
      # [batch, in_height, in_width, in_channels]
      # => [batch, in_height*2, in_width*2, out_channels]
      output = deconv(rectified, out_channels)
      output = batch_norm(output, is_training())

      if dropout > 0.0 and is_training():
        output = tf.nn.dropout(output, keep_prob=1 - dropout)

      layers.append(output)

  # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256,
  # generator_outputs_channels]
  with tf.variable_scope("decoder_1"):
    input = tf.concat([layers[-1], layers[0]], axis=3)
    rectified = tf.nn.relu(input)
    output = deconv(rectified, generator_outputs_channels)
    output = tf.tanh(output)
    layers.append(output)

  return layers[-1]


def create_model(product_image, body_seg, skin_seg, pose_map, prod_seg, image, category_num):
  """Build the model given product image, skin/body segments, pose
     predict the product segmentation.
  """
  
  with tf.variable_scope("generator") as scope:
    out_channels = int(prod_seg.get_shape()[-1] + image.get_shape()[-1])
    outputs = create_generator(product_image, body_seg, skin_seg,
                               pose_map, out_channels)

  # No discriminator.
  with tf.name_scope("generator_loss"):
    # output mask
    mask_outputs = outputs[:,:,:,:prod_seg.get_shape()[-1]]
    # output image
    image_outputs = outputs[:,:,:,prod_seg.get_shape()[-1]:]
    
    # losses
    gen_loss_mask_L1 = tf.reduce_mean(tf.abs(prod_seg - mask_outputs))
    gen_loss_content_L1 = tf.reduce_mean(tf.abs(image - image_outputs))
    
    if FLAGS.perceptual_weight > 0.0:
      with tf.variable_scope("vgg_19"):
        vgg_real = build_vgg19(image, FLAGS.vgg_model_path)
        vgg_fake = build_vgg19(image_outputs, FLAGS.vgg_model_path, reuse=True)
        # p0 = compute_error(vgg_real['input'],
        #                    vgg_fake['input'],
        #                    prod_segment)  # 256*256*3
        p1 = compute_error(vgg_real['conv1_2'],
                           vgg_fake['conv1_2']) / 5.3 * 2.5  # 128*128*64
        p2 = compute_error(vgg_real['conv2_2'],
                           vgg_fake['conv2_2']) / 2.7  / 1.2 # 64*64*128
        p3 = compute_error(vgg_real['conv3_2'],
                           vgg_fake['conv3_2']) / 1.35 / 2.3 # 32*32*256
        p4 = compute_error(vgg_real['conv4_2'],
                           vgg_fake['conv4_2']) / 0.67 / 8.2 # 16*16*512
        p5 = compute_error(vgg_real['conv5_2'],
                           vgg_fake['conv5_2']) / 0.16  # 8*8*512
        perceptual_loss = (p1 + p2 + p3 + p4 + p5) / 5.0 / 128.0
        # 128.0 for normalize to [0.1]
        
      gen_loss = (FLAGS.mask_l1_weight * gen_loss_mask_L1 +
                  FLAGS.content_l1_weight * gen_loss_content_L1 +
                  FLAGS.perceptual_weight * perceptual_loss)
    else:
      perceptual_loss = tf.get_variable('perceptual_loss', dtype=tf.float32, 
                                        initializer=tf.constant(0.0))
      gen_loss = (FLAGS.mask_l1_weight * gen_loss_mask_L1 +
                  FLAGS.content_l1_weight * gen_loss_content_L1)

  with tf.name_scope("generator_train"):
    # with tf.control_dependencies([discrim_train]):
    gen_tvars = [var for var in tf.trainable_variables()
                 if var.name.startswith("generator")]
    gen_optim = tf.train.AdamOptimizer(FLAGS.lr, FLAGS.beta1)
    gen_train = gen_optim.minimize(gen_loss, var_list=gen_tvars)
    print("create_model !!")

  global_step = tf.contrib.framework.get_or_create_global_step()
  incr_global_step = tf.assign(global_step, global_step+1)

  return Model(
      gen_loss_GAN=gen_loss,
      gen_loss_mask_L1=gen_loss_mask_L1,
      gen_loss_content_L1=gen_loss_content_L1,
      perceptual_loss=perceptual_loss,
      mask_outputs=mask_outputs,
      image_outputs=image_outputs,
      train=tf.group(incr_global_step, gen_train),
      global_step=global_step)
