# Converted to TensorFlow .caffemodel
# with the DeepLab-ResNet configuration.
# The batch normalisation layer is provided by
# the slim library (https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim).

from LIP_utils.kaffe.tensorflow import Network
import tensorflow as tf
from SEGMENTops import *

class JPPNetModel(Network):
    def setup(self, is_training, n_classes):
        '''Network definition.
        
        Args:
          is_training: whether to update the running mean and variance of the batch normalisation layer.
                       If the batch size is small, it is better to keep the running mean and variance of 
                       the-pretrained model frozen.
        '''
        (self.feed('data')
             .conv(7, 7, 64, 2, 2, biased=False, relu=False, name='conv1')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv1')
             .max_pool(3, 3, 2, 2, name='pool1')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2a_branch1')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn2a_branch1'))

        (self.feed('pool1')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2a_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2a_branch2a')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2a_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2a_branch2b')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2a_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn2a_branch2c'))

        (self.feed('bn2a_branch1', 
                   'bn2a_branch2c')
             .add(name='res2a')
             .relu(name='res2a_relu')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2b_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2b_branch2a')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2b_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2b_branch2b')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2b_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn2b_branch2c'))

        (self.feed('res2a_relu', 
                   'bn2b_branch2c')
             .add(name='res2b')
             .relu(name='res2b_relu')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2c_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2c_branch2a')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2c_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2c_branch2b')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2c_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn2c_branch2c'))

        (self.feed('res2b_relu', 
                   'bn2c_branch2c')
             .add(name='res2c')
             .relu(name='res2c_relu')
             .conv(1, 1, 512, 2, 2, biased=False, relu=False, name='res3a_branch1')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn3a_branch1'))

        (self.feed('res2c_relu')
             .conv(1, 1, 128, 2, 2, biased=False, relu=False, name='res3a_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3a_branch2a')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3a_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3a_branch2b')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3a_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn3a_branch2c'))

        (self.feed('bn3a_branch1', 
                   'bn3a_branch2c')
             .add(name='res3a')
             .relu(name='res3a_relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3b1_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3b1_branch2a')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3b1_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3b1_branch2b')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3b1_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn3b1_branch2c'))

        (self.feed('res3a_relu', 
                   'bn3b1_branch2c')
             .add(name='res3b1')
             .relu(name='res3b1_relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3b2_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3b2_branch2a')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3b2_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3b2_branch2b')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3b2_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn3b2_branch2c'))

        (self.feed('res3b1_relu', 
                   'bn3b2_branch2c')
             .add(name='res3b2')
             .relu(name='res3b2_relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3b3_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3b3_branch2a')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3b3_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3b3_branch2b')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3b3_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn3b3_branch2c'))

        (self.feed('res3b2_relu', 
                   'bn3b3_branch2c')
             .add(name='res3b3')
             .relu(name='res3b3_relu')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4a_branch1')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4a_branch1'))

        (self.feed('res3b3_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4a_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4a_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4a_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4a_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4a_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4a_branch2c'))

        (self.feed('bn4a_branch1', 
                   'bn4a_branch2c')
             .add(name='res4a')
             .relu(name='res4a_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b1_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b1_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b1_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b1_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b1_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b1_branch2c'))

        (self.feed('res4a_relu', 
                   'bn4b1_branch2c')
             .add(name='res4b1')
             .relu(name='res4b1_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b2_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b2_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b2_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b2_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b2_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b2_branch2c'))

        (self.feed('res4b1_relu', 
                   'bn4b2_branch2c')
             .add(name='res4b2')
             .relu(name='res4b2_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b3_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b3_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b3_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b3_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b3_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b3_branch2c'))

        (self.feed('res4b2_relu', 
                   'bn4b3_branch2c')
             .add(name='res4b3')
             .relu(name='res4b3_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b4_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b4_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b4_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b4_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b4_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b4_branch2c'))

        (self.feed('res4b3_relu', 
                   'bn4b4_branch2c')
             .add(name='res4b4')
             .relu(name='res4b4_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b5_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b5_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b5_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b5_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b5_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b5_branch2c'))

        (self.feed('res4b4_relu', 
                   'bn4b5_branch2c')
             .add(name='res4b5')
             .relu(name='res4b5_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b6_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b6_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b6_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b6_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b6_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b6_branch2c'))

        (self.feed('res4b5_relu', 
                   'bn4b6_branch2c')
             .add(name='res4b6')
             .relu(name='res4b6_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b7_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b7_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b7_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b7_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b7_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b7_branch2c'))

        (self.feed('res4b6_relu', 
                   'bn4b7_branch2c')
             .add(name='res4b7')
             .relu(name='res4b7_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b8_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b8_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b8_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b8_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b8_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b8_branch2c'))

        (self.feed('res4b7_relu', 
                   'bn4b8_branch2c')
             .add(name='res4b8')
             .relu(name='res4b8_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b9_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b9_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b9_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b9_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b9_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b9_branch2c'))

        (self.feed('res4b8_relu', 
                   'bn4b9_branch2c')
             .add(name='res4b9')
             .relu(name='res4b9_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b10_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b10_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b10_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b10_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b10_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b10_branch2c'))

        (self.feed('res4b9_relu', 
                   'bn4b10_branch2c')
             .add(name='res4b10')
             .relu(name='res4b10_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b11_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b11_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b11_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b11_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b11_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b11_branch2c'))

        (self.feed('res4b10_relu', 
                   'bn4b11_branch2c')
             .add(name='res4b11')
             .relu(name='res4b11_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b12_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b12_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b12_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b12_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b12_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b12_branch2c'))

        (self.feed('res4b11_relu', 
                   'bn4b12_branch2c')
             .add(name='res4b12')
             .relu(name='res4b12_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b13_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b13_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b13_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b13_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b13_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b13_branch2c'))

        (self.feed('res4b12_relu', 
                   'bn4b13_branch2c')
             .add(name='res4b13')
             .relu(name='res4b13_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b14_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b14_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b14_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b14_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b14_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b14_branch2c'))

        (self.feed('res4b13_relu', 
                   'bn4b14_branch2c')
             .add(name='res4b14')
             .relu(name='res4b14_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b15_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b15_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b15_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b15_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b15_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b15_branch2c'))

        (self.feed('res4b14_relu', 
                   'bn4b15_branch2c')
             .add(name='res4b15')
             .relu(name='res4b15_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b16_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b16_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b16_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b16_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b16_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b16_branch2c'))

        (self.feed('res4b15_relu', 
                   'bn4b16_branch2c')
             .add(name='res4b16')
             .relu(name='res4b16_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b17_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b17_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b17_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b17_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b17_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b17_branch2c'))

        (self.feed('res4b16_relu', 
                   'bn4b17_branch2c')
             .add(name='res4b17')
             .relu(name='res4b17_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b18_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b18_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b18_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b18_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b18_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b18_branch2c'))

        (self.feed('res4b17_relu', 
                   'bn4b18_branch2c')
             .add(name='res4b18')
             .relu(name='res4b18_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b19_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b19_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b19_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b19_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b19_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b19_branch2c'))

        (self.feed('res4b18_relu', 
                   'bn4b19_branch2c')
             .add(name='res4b19')
             .relu(name='res4b19_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b20_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b20_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b20_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b20_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b20_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b20_branch2c'))

        (self.feed('res4b19_relu', 
                   'bn4b20_branch2c')
             .add(name='res4b20')
             .relu(name='res4b20_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b21_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b21_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b21_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b21_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b21_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b21_branch2c'))

        (self.feed('res4b20_relu', 
                   'bn4b21_branch2c')
             .add(name='res4b21')
             .relu(name='res4b21_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b22_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b22_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b22_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b22_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b22_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b22_branch2c'))

        (self.feed('res4b21_relu', 
                   'bn4b22_branch2c')
             .add(name='res4b22')
             .relu(name='res4b22_relu'))

######################################parsing networks################################################################
        (self.feed('res4b22_relu')
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5a_branch1')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn5a_branch1'))

        (self.feed('res4b22_relu')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5a_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn5a_branch2a')
             .atrous_conv(3, 3, 512, 4, padding='SAME', biased=False, relu=False, name='res5a_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn5a_branch2b')
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5a_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn5a_branch2c'))

        (self.feed('bn5a_branch1', 
                   'bn5a_branch2c')
             .add(name='res5a')
             .relu(name='res5a_relu')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5b_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn5b_branch2a')
             .atrous_conv(3, 3, 512, 4, padding='SAME', biased=False, relu=False, name='res5b_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn5b_branch2b')
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5b_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn5b_branch2c'))

        (self.feed('res5a_relu', 
                   'bn5b_branch2c')
             .add(name='res5b')
             .relu(name='res5b_relu')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5c_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn5c_branch2a')
             .atrous_conv(3, 3, 512, 4, padding='SAME', biased=False, relu=False, name='res5c_branch2b')
             .batch_normalization(activation_fn=tf.nn.relu, name='bn5c_branch2b', is_training=is_training)
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5c_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn5c_branch2c'))

        (self.feed('res5b_relu', 
                   'bn5c_branch2c')
             .add(name='res5c')
             .relu(name='res5c_relu')
             .atrous_conv(3, 3, n_classes, 6, padding='SAME', relu=False, name='fc1_human_c0'))

        (self.feed('res5c_relu')
             .atrous_conv(3, 3, n_classes, 12, padding='SAME', relu=False, name='fc1_human_c1'))

        (self.feed('res5c_relu')
             .atrous_conv(3, 3, n_classes, 18, padding='SAME', relu=False, name='fc1_human_c2'))

        (self.feed('res5c_relu')
             .atrous_conv(3, 3, n_classes, 24, padding='SAME', relu=False, name='fc1_human_c3'))

        (self.feed('fc1_human_c0', 
                   'fc1_human_c1', 
                   'fc1_human_c2', 
                   'fc1_human_c3')
             .add(name='fc1_human'))

        (self.feed('res5c_relu')
             .conv(3, 3, 512, 1, 1, biased=True, relu=True, name='res5d_branch2a_parsing')
             .conv(3, 3, 256, 1, 1, biased=True, relu=True, name='res5d_branch2b_parsing'))


def pose_net(image, name):
    with tf.variable_scope(name) as scope:
        is_BN = False
        pose_conv1 = conv2d(image, 512, 3, 1, relu=True, bn=is_BN, name='pose_conv1')
        pose_conv2 = conv2d(pose_conv1, 512, 3, 1, relu=True, bn=is_BN, name='pose_conv2')
        pose_conv3 = conv2d(pose_conv2, 256, 3, 1, relu=True, bn=is_BN, name='pose_conv3')
        pose_conv4 = conv2d(pose_conv3, 256, 3, 1, relu=True, bn=is_BN, name='pose_conv4')
        pose_conv5 = conv2d(pose_conv4, 256, 3, 1, relu=True, bn=is_BN, name='pose_conv5')
        pose_conv6 = conv2d(pose_conv5, 256, 3, 1, relu=True, bn=is_BN, name='pose_conv6')

        pose_conv7 = conv2d(pose_conv6, 512, 1, 1, relu=True, bn=is_BN, name='pose_conv7')
        pose_conv8 = conv2d(pose_conv7, 16, 1, 1, relu=False, bn=is_BN, name='pose_conv8')

        return pose_conv8, pose_conv6


def pose_refine(pose, parsing, pose_fea, name):
    with tf.variable_scope(name) as scope:
        is_BN = False
        # 1*1 convolution remaps the heatmaps to match the number of channels of the intermediate features.
        pose = conv2d(pose, 128, 1, 1, relu=True, bn=is_BN, name='pose_remap')
        parsing = conv2d(parsing, 128, 1, 1, relu=True, bn=is_BN, name='parsing_remap')
        # concat 
        pos_par = tf.concat([pose, parsing, pose_fea], 3)
        conv1 = conv2d(pos_par, 512, 3, 1, relu=True, bn=is_BN, name='conv1')
        conv2 = conv2d(conv1, 256, 5, 1, relu=True, bn=is_BN, name='conv2')
        conv3 = conv2d(conv2, 256, 7, 1, relu=True, bn=is_BN, name='conv3')
        conv4 = conv2d(conv3, 256, 9, 1, relu=True, bn=is_BN, name='conv4')

        conv5 = conv2d(conv4, 256, 1, 1, relu=True, bn=is_BN, name='conv5')
        conv6 = conv2d(conv5, 16, 1, 1, relu=False, bn=is_BN, name='conv6')
        
        return conv6, conv4


def parsing_refine(parsing, pose, parsing_fea, name):
    with tf.variable_scope(name) as scope:
        is_BN = False
        pose = conv2d(pose, 128, 1, 1, relu=True, bn=is_BN, name='pose_remap')
        parsing = conv2d(parsing, 128, 1, 1, relu=True, bn=is_BN, name='parsing_remap')

        par_pos = tf.concat([parsing, pose, parsing_fea], 3)
        parsing_conv1 = conv2d(par_pos, 512, 3, 1, relu=True, bn=is_BN, name='parsing_conv1')
        parsing_conv2 = conv2d(parsing_conv1, 256, 5, 1, relu=True, bn=is_BN, name='parsing_conv2')
        parsing_conv3 = conv2d(parsing_conv2, 256, 7, 1, relu=True, bn=is_BN, name='parsing_conv3')
        parsing_conv4 = conv2d(parsing_conv3, 256, 9, 1, relu=True, bn=is_BN, name='parsing_conv4')

        parsing_conv5 = conv2d(parsing_conv4, 256, 1, 1, relu=True, bn=is_BN, name='parsing_conv5')
        parsing_human1 = atrous_conv2d(parsing_conv5, 20, 3, rate=6, relu=False, name='parsing_human1')
        parsing_human2 = atrous_conv2d(parsing_conv5, 20, 3, rate=12, relu=False, name='parsing_human2')
        parsing_human3 = atrous_conv2d(parsing_conv5, 20, 3, rate=18, relu=False, name='parsing_human3')
        parsing_human4 = atrous_conv2d(parsing_conv5, 20, 3, rate=24, relu=False, name='parsing_human4')
        parsing_human = tf.add_n([parsing_human1, parsing_human2, parsing_human3, parsing_human4], name='parsing_human')
        
        return parsing_human, parsing_conv4
# ###################################End################################################################

