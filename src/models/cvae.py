import tensorflow as tf

from tensorpack import Conv2D, Conv2DTranspose, FullyConnected, argscope

from src.models.base import BaseVAE


@BaseVAE.register('cvae')
class CVAE(BaseVAE):
    def encode(self, x):
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            with argscope(Conv2D, activation=tf.nn.relu):
                h = Conv2D('conv3x3_1', x, 32, 3, strides=(2, 2),
                           padding='valid')
                h = Conv2D('conv3x3_2', h, 64, 3, strides=(2, 2),
                           padding='valid')
            h = tf.layers.Flatten()(h)
            h = FullyConnected('fc', h, 2 * self._latent_dim)
            mean, logvar = tf.split(h, num_or_size_splits=2, axis=1)
        return mean, logvar

    def decode(self, z, apply_sigmoid=False):
        pre_convT_shape = [-1, int(self._image_shape[0] / 4),
                           int(self._image_shape[1] / 4), 32]
        pre_convT_unit = pre_convT_shape[1] * \
            pre_convT_shape[2] * pre_convT_shape[3]

        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            with argscope([Conv2D, FullyConnected], activation=tf.nn.relu):
                h = FullyConnected('fc', z, pre_convT_unit)
                h = tf.reshape(h, pre_convT_shape)
                h = Conv2DTranspose('convT3x3_1', h, 64, 3, strides=(2, 2))
                h = Conv2DTranspose('convT3x3_2', h, 32, 3, strides=(2, 2))
            h = Conv2DTranspose('convT1x1_1', h, self._image_shape[2],
                                3, strides=(1, 1))
            if apply_sigmoid:
                h = tf.sigmoid(h)

        return h
