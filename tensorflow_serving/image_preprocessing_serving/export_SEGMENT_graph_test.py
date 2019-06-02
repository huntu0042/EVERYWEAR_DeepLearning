import os

import numpy as np
import scipy.io as sio
import scipy.misc
import tensorflow as tf


from PIL import Image

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
def load_SEGMENT_graph(export_model):
    resize_height = 640
    resize_width = 480
    N_CLASSES = 20
    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        with tf.gfile.FastGFile('./pbtest3/000001_0.jpg', 'rb') as f:
            image = f.read()

        tf.saved_model.loader.load(
            sess, [tf.saved_model.tag_constants.SERVING], export_model)

        image_holder_bytes = graph.get_tensor_by_name("image_holder_bytes:0")
        segment_holder_bytes = graph.get_tensor_by_name("segment_output:0")

        model = segment_holder_bytes

        feed_dict = {
            image_holder_bytes : image
        }
        mask = sess.run(model, feed_dict=feed_dict)
    print(len(mask))
    print(mask)
    print(mask.shape)
    print(type(mask))

    load_segment = np.zeros((resize_height, resize_width), dtype=np.uint8)
    img = Image.new('RGB', (len(mask[0, 0]), len(mask[0])))
    pixels = img.load()
    for j_, j in enumerate(mask[0, :, :, 0]):
        for k_, k in enumerate(j):
            if k < N_CLASSES:
                load_segment[j_, k_] = k
                pixels[k_, j_] = label_colours[k]

    sio.savemat('{}/{}.mat'.format('./pbtest3/', '000001'), {'segment':load_segment}, do_compression=True)
    parsing_im = Image.fromarray(np.array(img))
    parsing_im.save('{}/{}_vis.png'.format('./pbtest3/', '000001'))


        
    
def main(unused_argv):
    export_model = 'savedmodel'
    load_SEGMENT_graph(export_model)


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    tf.app.run()