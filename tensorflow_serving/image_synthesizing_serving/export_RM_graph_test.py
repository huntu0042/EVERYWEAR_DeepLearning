import os
import json
import scipy.io as sio
import scipy.misc
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2

def process_one_image(image, resize_height, resize_width, if_zero_one=False):
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  if if_zero_one:
    return image
  image = tf.image.resize_images(image,
                                 size=[resize_height, resize_width],
                                 method=tf.image.ResizeMethod.BILINEAR)
  return (image - 0.5) * 2.0


def _process_image(image_name, product_image_name, sess,
                   resize_width=192, resize_height=256):
    

    coarse_image = process_one_image(coarse_image, resize_height, resize_width)
    mask_output = process_one_image(mask_output, resize_height, resize_width, True)
    tps_image = process_one_image(tps_image, resize_height, resize_width)

    [coarse_image, tps_image, mask_output] = sess.run([coarse_image, tps_image, mask_output])
    
    return coarse_image, tps_image, mask_output

def load_graph(export_model):
    image_name = "000001.png"
    product_image_name = ""
    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        with tf.gfile.FastGFile('./pbtest/000001_0.jpg', 'rb') as f:
            image = f.read()
        with tf.gfile.FastGFile('./pbtest/102001_1.png', 'rb') as f:
            prod_image = f.read()
        with tf.gfile.FastGFile('./pbtest/000001_0_102001_1_mask.png', 'rb') as f:
            mask_output = f.read()
        with tf.gfile.FastGFile('./pbtest/000001_0_102001_1.png', 'rb') as f:
            coarse_image = f.read()
        with tf.gfile.FastGFile('./pbtest/000001_0_102001_1_gmm.png', 'rb') as f:
            tps_image = f.read()
        
        tf.saved_model.loader.load(
            sess, [tf.saved_model.tag_constants.SERVING], export_model)

        image_holder = graph.get_tensor_by_name("image_holder_bytes:0")
        prod_image_holder = graph.get_tensor_by_name("prod_image_holder_bytes:0")
        prod_mask_holder = graph.get_tensor_by_name("prod_mask_holder_bytes:0")
        coarse_image_holder = graph.get_tensor_by_name("coarse_image_holder_bytes:0")
        tps_image_holder = graph.get_tensor_by_name("tps_image_holder_bytes:0")

        model_image_outputs_bytes = graph.get_tensor_by_name("model_image_outputs_bytes:0")
        select_mask_bytes = graph.get_tensor_by_name("select_mask_bytes:0")

        """
        with open('./input.json') as f:
            data = json.load(f)
        image = data['image']
        prod_image = data['prod_image']
        coarse_image = data['coarse_image']
        tps_image = data['tps_image']
        mask_output = data['mask_output']
        """
        model = [
            model_image_outputs_bytes,
            select_mask_bytes
        ]

        feed_dict = {
            image_holder : image,
            prod_image_holder : prod_image,
            prod_mask_holder : mask_output,
            coarse_image_holder : coarse_image,
            tps_image_holder : tps_image
        }
        [output, sel_mask] = sess.run(model, feed_dict=feed_dict)
        output = (output / 2.0 + 0.5)
        print(output.shape)
        print(type(output))
        scipy.misc.imsave('./pbtest/1.png', np.squeeze(output))
        scipy.misc.imsave('./pbtest/2.png', np.squeeze(sel_mask))
        """
        op = graph.get_operations()
        for node in op:
            print(node.values())
        """

def main(unused_argv):
    export_model = 'savedmodel'
    load_graph(export_model)

if __name__ == '__main__':
    tf.app.run()