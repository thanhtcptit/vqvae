import sonnet as snt
import tensorflow as tf

from src.models.base import BaseVQVAE


def residual_stack(h, num_hiddens, num_residual_layers, num_residual_hiddens):
    for i in range(num_residual_layers):
        h_i = tf.nn.relu(h)

        h_i = tf.layers.conv2d(
            h_i, num_residual_hiddens,
            3, strides=(1, 1), padding='same',
            name="res3x3_%d" % i)
        h_i = tf.nn.relu(h_i)

        h_i = tf.layers.conv2d(
            h_i, num_hiddens,
            1, strides=(1, 1), padding='same',
            name="res1x1_%d" % i)
        h += h_i
    return tf.nn.relu(h)


@BaseVQVAE.register('vqvae_v1')
class VQVAE(BaseVQVAE):
    def __init__(self, image_shape, num_hiddens, embedding_dim, num_embeddings,
                 commitment_cost, learning_rate, decay=0.99, use_ema=False):
        super(VQVAE, self).__init__(
            image_shape, num_hiddens, embedding_dim, num_embeddings,
            commitment_cost, learning_rate, decay, use_ema)

    def encode(self, x):
        with tf.variable_scope('vqvae', reuse=tf.AUTO_REUSE):
            h = tf.layers.conv2d(x, self._num_hiddens / 2, 4, strides=(2, 2),
                                 padding='same', name='enc_1')
            h = tf.nn.relu(h)

            h = tf.layers.conv2d(h, self._num_hiddens, 4, strides=(2, 2),
                                 padding='same', name="enc_2")
            h = tf.nn.relu(h)

            h = tf.layers.conv2d(
                h, self._num_hiddens, 3,
                strides=(1, 1), padding='same', name="enc_3")

            h = residual_stack(h, self._num_hiddens, 2, 32)

            ze = tf.layers.conv2d(
                h, self._embedding_dim, 1,
                strides=(1, 1), padding='same', name="to_vq")

        return ze

    def decode(self, z):
        with tf.variable_scope('vqvae', reuse=tf.AUTO_REUSE):
            h = tf.layers.conv2d(
                z, self._num_hiddens, 3,
                strides=(1, 1), padding='same', name="dec_1")

            h = residual_stack(h, self._num_hiddens, 2, 32)

            h = tf.layers.conv2d_transpose(
                h, int(self._num_hiddens / 2), 4,
                strides=(2, 2), padding='same', name="dec_2")
            h = tf.nn.relu(h)

            x_recon = tf.layers.conv2d_transpose(
                h, 3, 4,
                strides=(2, 2), padding='same', name="dec_3")

        return x_recon
