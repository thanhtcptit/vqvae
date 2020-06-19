import tensorflow as tf

from src.models.base import BaseImageEmbedding


def residual_stack(h, num_hiddens, num_residual_layers, num_residual_hiddens):
    for i in range(num_residual_layers):
        h_i = tf.nn.relu(h)

        h_i = tf.layers.conv2d(
            h_i, num_residual_hiddens, 3, strides=(1, 1), padding='same',
            name="res3x3_%d" % i)
        h_i = tf.nn.relu(h_i)

        h_i = tf.layers.conv2d(
            h_i, num_hiddens, 1, strides=(1, 1), padding='same',
            name="res1x1_%d" % i)
        h += h_i
    return tf.nn.relu(h)


@BaseImageEmbedding.register('ff')
class SimpleEmbed(BaseImageEmbedding):
    def embed(self, x, is_training):
        with tf.variable_scope('embed_layer'):
            h = tf.layers.Flatten()(x)
            h = tf.layers.dense(h, self._embeddings_dim * 4,
                                activation=tf.nn.sigmoid, name='ff1')
            h = tf.layers.dense(h, self._embeddings_dim * 2,
                                activation=tf.nn.sigmoid, name='ff2')
            h = tf.layers.dropout(h, self._drop_out, training=is_training)
            h = tf.layers.dense(h, self._embeddings_dim,
                                activation=tf.nn.sigmoid, name='ff3')
        return h


@BaseImageEmbedding.register('conv')
class ConvEmbed(BaseImageEmbedding):
    def embed(self, x, is_training):
        with tf.variable_scope('embed_layer'):
            h = tf.layers.conv2d(x, self._num_hiddens, 3, padding='valid',
                                 name='conv1')
            h = tf.nn.relu(h)
            h = tf.layers.conv2d(h, self._num_hiddens, 3, padding='valid',
                                 name='conv2')
            h = residual_stack(h, self._num_hiddens, 2, 32)
            h = tf.layers.Flatten()(h)
            h = tf.layers.dropout(h, self._drop_out, training=is_training)
            h = tf.layers.dense(h, self._embeddings_dim, name='fc')
        return h
