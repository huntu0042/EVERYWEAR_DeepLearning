from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from SEGMENTutil import JPPNetModel
from SEGMENTutil import pose_net
from SEGMENTutil import pose_refine
from SEGMENTutil import parsing_refine

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("model_name", "SEGMENT", "body-parsing-segment")
tf.flags.DEFINE_string("checkpoint_dir", "./segment/", "model directory")


def export_segment_graph(model_name):
    segment_graph = tf.Graph()
    with segment_graph.as_default():
        N_CLASSES = 20

        image_holder_bytes = tf.placeholder(tf.string, shape=[], name="image_holder_bytes")
        resize_height = 640
        resize_width = 480
        input_size = (640, 480)

        with tf.name_scope("create_inputs"):
            image_holder_bytes = tf.image.decode_png(image_holder_bytes, channels=3)
            image_holder_bytes = tf.image.resize_images(image_holder_bytes,
                                size=[resize_height, resize_width],
                                method=tf.image.ResizeMethod.BILINEAR)
            image_holder_rev = tf.reverse(image_holder_bytes, tf.stack([1]))
        
        image_batch_origin = tf.stack([image_holder_bytes, image_holder_rev])
        image_batch = tf.image.resize_images(image_batch_origin, [resize_height, resize_width])
        image_batch075 = tf.image.resize_images(image_batch_origin, [int(resize_height * 0.75), int(resize_width * 0.75)])
        image_batch125 = tf.image.resize_images(image_batch_origin, [int(resize_height * 1.25), int(resize_width * 1.25)])

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
        segment_output_bytes = tf.expand_dims(raw_output_all, dim=3) # Create 4-d tensor. # Create 4-d tensor
        print(segment_output_bytes)
        segment_output_bytes = tf.identity(segment_output_bytes, name="segment_output")

        
        # Which variables to load.
        restore_var = tf.global_variables()
        # Set up tf session and initialize variables.

        restore_saver = tf.train.Saver(var_list=restore_var)
    
    with tf.Session(graph=segment_graph) as sess:
        sess.run(tf.global_variables_initializer())
        print("loading model from checkpoint")
        """
        checkpoint = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            checkpoint_name = os.path.basename(checkpoint.model_checkpoint_path)
            restore_saver.restore(sess, os.path.join(FLAGS.checkpoint_dir, checkpoint_name))
            print("Restored model parameters from {}".format(checkpoint_name))
        """
        checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        print("Restored model parameters from ", checkpoint)
        restore_saver.restore(sess, checkpoint)
        for var in tf.trainable_variables():
            print(var)
        
        builder = tf.compat.v1.saved_model.builder.SavedModelBuilder("savedmodel")
        builder.add_meta_graph_and_variables(sess, [tf.compat.v1.saved_model.tag_constants.SERVING],
                signature_def_map={
                    "model": tf.compat.v1.saved_model.signature_def_utils.predict_signature_def(
                        inputs={"image_holder_bytes": image_holder_bytes},
                        outputs={"segment_output": segment_output_bytes})
                })
    builder.save()

    
def main(unused_argv):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print("Export SEGMENT model..")
    export_segment_graph(FLAGS.model_name)


if __name__ == "__main__":
    tf.app.run()