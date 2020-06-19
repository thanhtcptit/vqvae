from __future__ import print_function

import os
import zipfile
import shutil
import tarfile
import tempfile
import subprocess
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from six.moves import cPickle, urllib, xrange

from models import VQVAE


train_log = 'train_log/vqvae-snt/'
if os.path.exists(train_log):
    shutil.rmtree(train_log)
os.makedirs(train_log + 'images')


def zip_file(file_path, zip_path):
    if os.path.exists(zip_path):
        os.remove(zip_path)

    zipw = zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED)
    for root, dirs, files in os.walk(file_path):
        for f in files:
            zipw.write(os.path.join(root, f))

    zipw.close()


def download_cifar10(download_dir):
    data_path = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    url = urllib.request.urlopen(data_path)
    archive = tarfile.open(fileobj=url, mode='r|gz')  # read a .tar.gz stream
    archive.extractall(download_dir)
    url.close()
    archive.close()
    print('extracted data files to %s' % local_data_dir)


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


def viz_reconstruction(train_originals, train_reconstructions,
                       valid_originals, valid_reconstructions, epoch):
    def convert_batch_to_image_grid(image_batch):
        reshaped = (image_batch.reshape(4, 4, 32, 32, 3)
                    .transpose(0, 2, 1, 3, 4)
                    .reshape(4 * 32, 4 * 32, 3))
        return reshaped + 0.5

    f = plt.figure(figsize=(8, 8))
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
    plt.savefig(train_log + 'images/viz_{}.png'.format(epoch))
    plt.close()


local_data_dir = 'data/cifar10'
if not os.path.exists(local_data_dir):
    tf.gfile.MakeDirs(local_data_dir)
    download_cifar10(local_data_dir)

train_data_dict = combine_batches([unpickle(
    os.path.join(local_data_dir, 'cifar-10-batches-py/data_batch_%d' % i))
    for i in range(1, 5)])
valid_data_dict = combine_batches([unpickle(
    os.path.join(local_data_dir, 'cifar-10-batches-py/data_batch_5'))])
test_data_dict = combine_batches([unpickle(
    os.path.join(local_data_dir, 'cifar-10-batches-py/test_batch'))])

data_variance = np.var(train_data_dict['images'] / 255.0)

num_sample = 16
train_sample = train_data_dict['images'][:num_sample] / 255. - 0.5
val_sample = valid_data_dict['images'][:num_sample] / 255. - 0.5

""" Build graph & Training """
tf.reset_default_graph()

# The hyper-parameters in the paper were (For ImageNet):
# batch_size = 128
# image_size = 128
batch_size = 32
image_size = 32

num_training_updates = 50000

num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2

embedding_dim = 64
num_embeddings = 512

# commitment_cost should be set appropriately. It's often useful to try
# a couple of values. It mostly depends on the scale of the reconstruction cost
# (log p(x|z)). So if the reconstruction cost is 100x higher, the
# commitment_cost should also be multiplied with the same amount.
commitment_cost = 0.25

# Use EMA updates for the codebook (instead of the Adam optimizer).
# This typically converges faster, and makes the model less dependent on choice
# of the optimizer. In the VQ-VAE paper EMA updates were not used (but was
# developed afterwards). See Appendix of the paper for more details.
vq_use_ema = True

# This is only used for EMA updates.
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
    .repeat(1)  # 1 epoch
    .batch(batch_size)).make_initializable_iterator()
train_dataset_batch = train_dataset_iterator.get_next()
valid_dataset_batch = valid_dataset_iterator.get_next()

# Process inputs with conv stack, finishing with 1x1 to get to correct size.
x = tf.placeholder(tf.float32, shape=(None, image_size, image_size, 3))
# perplexity indicates how many codes are 'active' on average.
x_recon, vq_loss, perplexity = model(x)

# Normalized MSE
# recon_error = tf.reduce_mean((x_recon - x) ** 2) / data_variance
recon_error = tf.reduce_mean((x_recon - x) ** 2)
loss = recon_error + vq_loss

# Create optimizer and TF session.
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss)

checkpoint_path = os.path.join(train_log + 'model-{}.ckpt')
saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

# Train.
train_res_recon_error = []
train_res_perplexity = []
for i in xrange(1, num_training_updates + 1):
    feed_dict = {x: get_images(sess)}
    results = sess.run([train_op, recon_error, perplexity],
                       feed_dict=feed_dict)
    train_res_recon_error.append(results[1])
    train_res_perplexity.append(results[2])

    if i % 100 == 0:
        print('%d iterations' % (i + 1))
        print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
        print('perplexity: %.3f\n' % np.mean(train_res_perplexity[-100:]))
        train_sample_recon = sess.run(x_recon,
                                      feed_dict={x: train_sample})
        val_sample_recon = sess.run(x_recon,
                                    feed_dict={x: val_sample})
        viz_reconstruction(train_sample, train_sample_recon, val_sample,
                           val_sample_recon, int(i / 100))

    if i % 1000 == 0:
        saver.save(sess, checkpoint_path.format(i))
        print('Save model')


zip_file(os.path.join(train_log, 'images'),
         os.path.join(train_log, 'images.zip'))
