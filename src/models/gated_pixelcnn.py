import numpy as np
import tensorflow as tf

from nsds.common import Params
from nsds.common.registrable import Registrable
from tensorflow.python.framework.tensor_spec import TensorSpec
from tensorpack import ModelDesc, get_current_tower_context
from tensorpack.tfutils.summary import add_moving_summary

from src.models.base import BasePixelCNNPrior


def gated_func(x):
    x1, x2 = tf.split(x, num_or_size_splits=2, axis=-1)
    return tf.tanh(x1) * tf.sigmoid(x2)


def mask_kernel(mask_shape, kernel, mask_dim):
    mask = np.ones(shape=mask_shape, dtype=np.float32)
    if mask_dim == 0:
        mask[-1, :, :, :] = 0
    elif mask_dim == 1:
        mask[:, -1, :, :] = 0
    else:
        raise ValueError()

    mask = tf.get_variable(
        name=f'mask_{mask_dim}', initializer=tf.constant(mask),
        dtype=tf.float32, trainable=False)
    return kernel * mask


@BasePixelCNNPrior.register('unconditional')
class UnconditionalGatedPixelCNN(BasePixelCNNPrior):
    def __init__(self, image_shape, latent_shape, num_layers, num_embeddings,
                 num_hiddens, num_labels, learning_rate, vqvae_model_params):
        super(UnconditionalGatedPixelCNN, self).__init__(
            image_shape, latent_shape, num_layers, num_embeddings,
            num_hiddens, num_labels, learning_rate, vqvae_model_params)

    def gated_layer(self, x_v, x_h, filter_size, scope_name, mask_type='B'):
        assert filter_size % 2 == 1, 'Filter size must be odd'
        filter_size_ver = [filter_size // 2 + 1, filter_size,
                           self._num_hiddens, 2 * self._num_hiddens]
        padding_ver = [filter_size // 2, filter_size // 2]
        filter_size_hor = [1, filter_size // 2 + 1,
                           self._num_hiddens, 2 * self._num_hiddens]
        padding_hor = [0, filter_size // 2]
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            kernel_ver = tf.get_variable(name='kernel_ver',
                                         shape=filter_size_ver)
            kernel_hor = tf.get_variable(name='kernel_hor',
                                         shape=filter_size_hor)
            if mask_type == 'A':
                kernel_ver = mask_kernel(filter_size_ver, kernel_ver, 0)
                kernel_hor = mask_kernel(filter_size_hor, kernel_hor, 1)

            x_v_pad = tf.pad(x_v, [[0, 0], [padding_ver[0], padding_ver[0]],
                                   [padding_ver[1], padding_ver[1]], [0, 0]])
            x_h_pad = tf.pad(x_h, [[0, 0], [padding_hor[0], padding_hor[0]],
                                   [padding_hor[1], padding_hor[1]], [0, 0]])

            h_ver = tf.nn.conv2d(x_v_pad, kernel_ver, [1, 1, 1, 1],
                                 padding='VALID')
            h_hor = tf.nn.conv2d(x_h_pad, kernel_hor, [1, 1, 1, 1],
                                 padding='VALID')
            h_ver = h_ver[:, :self._latent_shape[0], :, :]
            h_hor = h_hor[:, :, :self._latent_shape[1], :]
            out_v = gated_func(h_ver)

            h_ver_to_hor = tf.layers.conv2d(h_ver, 2 * self._num_hiddens, 1)
            out_h = gated_func(h_hor + h_ver_to_hor)
            out_h = tf.layers.conv2d(out_h, self._num_hiddens, 1)

            if mask_type == 'B':
                out_h = out_h + x_h
        return out_v, out_h

    def generate(self, z, _):
        with tf.variable_scope('embs', reuse=tf.AUTO_REUSE):
            input_embs = tf.get_variable(
                name='input_embs', shape=[self._num_embeddings,
                                          self._num_hiddens])
        z = tf.nn.embedding_lookup(input_embs, z)

        h_v, h_h = self.gated_layer(z, z, 7, 'maskA_layer', mask_type='A')
        for i in range(1, self._num_layers):
            h_v, h_h = self.gated_layer(h_v, h_h, 3, f'maskB_layer_{i}')

        with tf.variable_scope('output', reuse=tf.AUTO_REUSE):
            h_h = tf.layers.conv2d(h_h, self._num_embeddings, 1)
            h_h = tf.nn.relu(h_h)
            h_h = tf.layers.conv2d(h_h, self._num_embeddings, 1)
        return h_h


@BasePixelCNNPrior.register('conditional')
class ConditionalGatedPixelCNN(BasePixelCNNPrior):
    def __init__(self, image_shape, latent_shape, num_layers, num_embeddings,
                 num_hiddens, num_labels, learning_rate, vqvae_model_params):
        super(ConditionalGatedPixelCNN, self).__init__(
            image_shape, latent_shape, num_layers, num_embeddings,
            num_hiddens, num_labels, learning_rate, vqvae_model_params)

    def gated_layer(self, x_v, x_h, y, filter_size, scope_name, mask_type='B'):
        assert filter_size % 2 == 1, 'Filter size must be odd'
        filter_size_ver = [filter_size // 2 + 1, filter_size,
                           self._num_hiddens, 2 * self._num_hiddens]
        padding_ver = [filter_size // 2, filter_size // 2]
        filter_size_hor = [1, filter_size // 2 + 1,
                           self._num_hiddens, 2 * self._num_hiddens]
        padding_hor = [0, filter_size // 2]
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            kernel_ver = tf.get_variable(name='kernel_ver',
                                         shape=filter_size_ver)
            kernel_hor = tf.get_variable(name='kernel_hor',
                                         shape=filter_size_hor)
            if mask_type == 'A':
                kernel_ver = mask_kernel(filter_size_ver, kernel_ver, 0)
                kernel_hor = mask_kernel(filter_size_hor, kernel_hor, 1)

            x_v_pad = tf.pad(x_v, [[0, 0], [padding_ver[0], padding_ver[0]],
                                   [padding_ver[1], padding_ver[1]], [0, 0]])
            x_h_pad = tf.pad(x_h, [[0, 0], [padding_hor[0], padding_hor[0]],
                                   [padding_hor[1], padding_hor[1]], [0, 0]])

            h_ver = tf.nn.conv2d(x_v_pad, kernel_ver, [1, 1, 1, 1],
                                 padding='VALID')
            h_hor = tf.nn.conv2d(x_h_pad, kernel_hor, [1, 1, 1, 1],
                                 padding='VALID')
            h_ver = h_ver[:, :self._latent_shape[0], :, :]
            h_hor = h_hor[:, :, :self._latent_shape[1], :]
            out_v = gated_func(h_ver + y)

            h_ver_to_hor = tf.layers.conv2d(h_ver, 2 * self._num_hiddens, 1)
            out_h = gated_func(h_hor + h_ver_to_hor + y)
            out_h = tf.layers.conv2d(out_h, self._num_hiddens, 1)

            if mask_type == 'B':
                out_h = out_h + x_h
        return out_v, out_h

    def generate(self, z, y):
        with tf.variable_scope('embs', reuse=tf.AUTO_REUSE):
            input_embs = tf.get_variable(
                name='input_embs', shape=[self._num_embeddings,
                                          self._num_hiddens])
            class_embs = tf.get_variable(
                name='class_embs', shape=[self._num_labels,
                                          2 * self._num_hiddens])
        z = tf.nn.embedding_lookup(input_embs, z)
        y = tf.nn.embedding_lookup(class_embs, y)
        y = tf.reshape(y, shape=[-1, 1, 1, 2 * self._num_hiddens])

        h_v, h_h = self.gated_layer(z, z, y, 7, 'maskA_layer', mask_type='A')
        for i in range(1, self._num_layers):
            h_v, h_h = self.gated_layer(h_v, h_h, y, 3, f'maskB_layer_{i}')

        with tf.variable_scope('output', reuse=tf.AUTO_REUSE):
            h_h = tf.layers.conv2d(h_h, self._num_embeddings, 1)
            h_h = tf.nn.relu(h_h)
            h_h = tf.layers.conv2d(h_h, self._num_embeddings, 1)
        return h_h
