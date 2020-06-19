from __future__ import print_function

import os
import subprocess
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tarfile

from six.moves import cPickle, urllib, xrange

from models import VQVAE


def unpickle(filename):
    with open(filename, 'rb') as fo:
        return cPickle.load(fo, encoding='latin1')


def reshape_flattened_image_batch(flat_image_batch):
    # convert from NCHW to NHWC
    return flat_image_batch.reshape(-1, 3, 32, 32).transpose([0, 2, 3, 1])


def combine_batches(batch_list):
    images = np.vstack([reshape_flattened_image_batch(batch['data'])
                        for batch in batch_list])
    labels = np.vstack([np.array(batch['labels'])
                        for batch in batch_list]).reshape(-1, 1)
    return {'images': images, 'labels': labels}


def cast_and_normalise_images(data_dict):
    """Convert images to floating point with the range [0.5, 0.5]"""
    images = data_dict['images']
    data_dict['images'] = (tf.cast(images, tf.float32) / 255.0) - 0.5
    return data_dict


def get_images(sess, subset='train'):
    if subset == 'train':
        return sess.run(train_dataset_batch)['images']
    elif subset == 'valid':
        return sess.run(valid_dataset_batch)['images']


local_data_dir = 'data/cifar10'
train_data_dict = combine_batches([unpickle(
    os.path.join(local_data_dir, 'cifar-10-batches-py/data_batch_%d' % i))
    for i in range(1, 5)])
valid_data_dict = combine_batches([unpickle(
    os.path.join(local_data_dir, 'cifar-10-batches-py/data_batch_5'))])
test_data_dict = combine_batches([unpickle(
    os.path.join(local_data_dir, 'cifar-10-batches-py/test_batch'))])

data_variance = np.var(train_data_dict['images'] / 255.0)


""" Build graph & Training """
tf.reset_default_graph()

# Set hyper-parameters.
batch_size = 32
image_size = 32
num_training_updates = 50000

num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2

embedding_dim = 64
num_embeddings = 512

commitment_cost = 0.25

vq_use_ema = False
decay = 0.99

learning_rate = 3e-4

model = VQVAE(num_hiddens, num_residual_hiddens,
              num_residual_hiddens, embedding_dim, num_embeddings,
              commitment_cost, decay, use_ema=False)


# Data Loading.
train_dataset_iterator = (
    tf.data.Dataset.from_tensor_slices(train_data_dict)
    .map(cast_and_normalise_images)
    .shuffle(10000)
    .repeat(-1)  # repeat indefinitely
    .batch(batch_size)).make_one_shot_iterator()
valid_dataset_iterator = (
    tf.data.Dataset.from_tensor_slices(valid_data_dict)
    .map(cast_and_normalise_images)
    .shuffle(10000)
    .repeat(1)  # 1 epoch
    .batch(batch_size)).make_initializable_iterator()
train_dataset_batch = train_dataset_iterator.get_next()
valid_dataset_batch = valid_dataset_iterator.get_next()


# Process inputs with conv stack, finishing with 1x1 to get to correct size.
x = tf.placeholder(tf.float32, shape=(None, image_size, image_size, 3))
x_recon, _, _ = model(x, is_training=False)

checkpoint_path = os.path.join('train_log/draft/model-{}.ckpt')
saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, 'train_log/draft/model-50000.ckpt')

# Reconstructions
sess.run(valid_dataset_iterator.initializer)
train_originals = get_images(sess, subset='train')
train_reconstructions = sess.run(x_recon,
                                 feed_dict={x: train_originals})
valid_originals = get_images(sess, subset='valid')
valid_reconstructions = sess.run(x_recon,
                                 feed_dict={x: valid_originals})


def convert_batch_to_image_grid(image_batch):
    reshaped = (image_batch.reshape(4, 8, 32, 32, 3)
                .transpose(0, 2, 1, 3, 4)
                .reshape(4 * 32, 8 * 32, 3))
    return reshaped + 0.5


# View reconstructions
f = plt.figure(figsize=(16, 8))
ax = f.add_subplot(2, 2, 1)
ax.imshow(convert_batch_to_image_grid(train_originals),
          interpolation='nearest')
ax.set_title('training data originals')
plt.axis('off')

ax = f.add_subplot(2, 2, 2)
ax.imshow(convert_batch_to_image_grid(train_reconstructions),
          interpolation='nearest')
ax.set_title('training data reconstructions')
plt.axis('off')

ax = f.add_subplot(2, 2, 3)
ax.imshow(convert_batch_to_image_grid(valid_originals),
          interpolation='nearest')
ax.set_title('validation data originals')
plt.axis('off')

ax = f.add_subplot(2, 2, 4)
ax.imshow(convert_batch_to_image_grid(valid_reconstructions),
          interpolation='nearest')
ax.set_title('validation data reconstructions')
plt.axis('off')
plt.savefig('train_log/draft/viz.png')
