from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("image_dir", "data/men_tshirts/",
                       "Directory containing product and person images.")
tf.flags.DEFINE_string("model_name", "RM", "RefineMent Network")                
tf.flags.DEFINE_string("test_label",
                       "data/viton_test_pairs.txt",
                       "File containing labels for testing.")

tf.flags.DEFINE_string("result_dir", "results/stage2/",
                       "Folder containing the results of testing.")
tf.flags.DEFINE_string("coarse_result_dir", "results/stage1",
                  "Folder containing the results of stage1 (coarse) results.")
tf.flags.DEFINE_string("checkpoint_dir", "model/stage2/",
                       "model directory")
tf.flags.DEFINE_integer("height", 256, "")
tf.flags.DEFINE_integer("width", 192, "")


def convert2float(image, resize_height=256, resize_width=192, if_zero_one=False):
    """ Transfrom from int image ([0,255]) to float tensor ([-1.,1.])
    """
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    if if_zero_one:
        return image
    image = tf.image.resize_images(image,
                                size=[resize_height, resize_width],
                                method=tf.image.ResizeMethod.BILINEAR)
    return (image - 0.5) * 2.0


def lrelu(x, a=0.2):
   with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def create_refine_generator(stn_image_outputs, gen_image_outputs):
    generator_input = tf.concat([stn_image_outputs, gen_image_outputs],
                                axis=-1)
    print(type(generator_input))
    print(generator_input.shape)
    downsampled = tf.image.resize_area(generator_input, (256, 192), align_corners=False)
    net = slim.conv2d(downsampled, 64, [3, 3], rate=1, normalizer_fn=slim.layer_norm,
                        activation_fn=lrelu, scope="g_256_conv1")
    net = slim.conv2d(net, 64, [3, 3], rate=1, normalizer_fn=slim.layer_norm,
                        activation_fn=lrelu, scope='g_256_conv2')
    net = slim.conv2d(net, 64, [3, 3], rate=1, normalizer_fn=slim.layer_norm,
                        activation_fn=lrelu, scope='g_256_conv3')
    net = slim.conv2d(net, 1, [1, 1], rate=1,
                        activation_fn=None, scope='g_1024_final')
    net = tf.sigmoid(net)
    return net


def export_graph(model_name):
    batch_size = 1
    graph = tf.Graph()

    with graph.as_default():
        
        prod_image_holder_bytes = tf.placeholder(tf.string, shape=[], name="prod_image_holder_bytes")
        prod_mask_holder_bytes = tf.placeholder(tf.string, shape=[], name="prod_mask_holder_bytes")
        coarse_image_holder_bytes = tf.placeholder(tf.string, shape=[], name="coarse_image_holder_bytes")
        tps_image_holder_bytes = tf.placeholder(tf.string, shape=[], name="tps_image_holder_bytes")
        
        
        prod_image_holder = tf.image.decode_png(prod_image_holder_bytes, channels=3)
        prod_mask_holder = tf.image.decode_png(prod_mask_holder_bytes, channels=1)
        coarse_image_holder = tf.image.decode_png(coarse_image_holder_bytes, channels=3)
        tps_image_holder = tf.image.decode_png(tps_image_holder_bytes, channels=3)
        
        
        
        prod_image_holder = convert2float(prod_image_holder)
        prod_mask_holder = convert2float(prod_mask_holder, if_zero_one=True)
        coarse_image_holder = convert2float(coarse_image_holder)
        tps_image_holder = convert2float(tps_image_holder)

        
        prod_image_holder = tf.expand_dims(prod_image_holder, 0)
        prod_mask_holder = tf.expand_dims(prod_mask_holder, 0)
        coarse_image_holder = tf.expand_dims(coarse_image_holder, 0)
        tps_image_holder = tf.expand_dims(tps_image_holder, 0)
        

        with tf.variable_scope("refine_generator") as scope:
            select_mask_bytes = create_refine_generator(tps_image_holder,
                                                  coarse_image_holder)
            select_mask_bytes = select_mask_bytes * prod_mask_holder
            model_image_outputs_bytes = (select_mask_bytes * tps_image_holder +
                                   (1 - select_mask_bytes) * coarse_image_holder)


        model_image_outputs_bytes = tf.image.convert_image_dtype(model_image_outputs_bytes / 2.0 + 0.5, tf.uint8)
        select_mask_bytes = tf.image.convert_image_dtype(select_mask_bytes, tf.uint8)
        model_image_outputs_bytes = tf.image.encode_png(tf.squeeze(model_image_outputs_bytes, [0]))
        select_mask_bytes = tf.image.encode_png(tf.squeeze(select_mask_bytes, [0]))
        model_image_outputs_bytes = tf.identity(model_image_outputs_bytes, name="model_image_outputs_bytes")
        select_mask_bytes = tf.identity(select_mask_bytes, name="select_mask_bytes")
        
        restore_saver = tf.train.Saver(var_list=[var for var in tf.trainable_variables()
                                            if var.name.startswith("refine_generator")])
        
        #restore_saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        print("loading model from checkpoint")
        checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        print(checkpoint)
        
        restore_saver.restore(sess, checkpoint)
        for var in tf.trainable_variables():
            if var.name.startswith("refine_generator"):
                print(var)
        
        

        """
        output_graph_def = tf.graph_util.convert_variables_to_constants(
                           sess, graph.as_graph_def(), [model_image_outputs_bytes.op.name, select_mask_bytes.op.name])
        """
        # Savedmodel을 저장할 경로
        builder = tf.compat.v1.saved_model.builder.SavedModelBuilder("savedmodel")
        builder.add_meta_graph_and_variables(sess, [tf.compat.v1.saved_model.tag_constants.SERVING],
                signature_def_map={
                    "model": tf.compat.v1.saved_model.signature_def_utils.predict_signature_def(
                        inputs={"prod_image_holder_bytes": prod_image_holder_bytes,
                                "prod_mask_holder_bytes": prod_mask_holder_bytes,
                                "coarse_image_holder_bytes": coarse_image_holder_bytes,
                                "tps_image_holder_bytes": tps_image_holder_bytes},
                        outputs={"model_image_outputs_bytes": model_image_outputs_bytes,
                                 "select_mask_bytes": select_mask_bytes})
                })
    builder.save()
    #tf.train.write_graph(output_graph_def, 'pretrained', model_name, as_text=False)


def main(unused_argv):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print("Export model..")
    export_graph(FLAGS.model_name)


if __name__ == "__main__":
    tf.app.run()