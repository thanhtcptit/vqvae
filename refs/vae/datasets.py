import os
import tarfile
import numpy as np
import tensorflow as tf

from six.moves import cPickle, urllib


def get_dataset(dataset):
    if dataset == 'mnist':
        return get_mnist()
    elif dataset == 'cifar10':
        return get_cifar10('data/cifar10')
    raise ValueError()


def get_mnist():
    (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(
        train_images.shape[0], 28, 28, 1).astype('float32')
    test_images = test_images.reshape(
        test_images.shape[0], 28, 28, 1).astype('float32')

    train_images /= 255.
    test_images /= 255.

    # Binarization
    train_images[train_images >= .5] = 1.
    train_images[train_images < .5] = 0.
    test_images[test_images >= .5] = 1.
    test_images[test_images < .5] = 0.

    return train_images, test_images


def get_cifar10(data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        download_cifar10(data_dir)

    train_data_dict = combine_batches([unpickle(
        os.path.join(data_dir, 'cifar-10-batches-py/data_batch_%d' % i))
        for i in range(1, 5)])
    valid_data_dict = combine_batches([unpickle(
        os.path.join(data_dir, 'cifar-10-batches-py/data_batch_5'))])
    test_data_dict = combine_batches([unpickle(
        os.path.join(data_dir, 'cifar-10-batches-py/test_batch'))])

    train_data_dict['images'] = train_data_dict['images'] / 255
    valid_data_dict['images'] = valid_data_dict['images'] / 255

    # train_images -= 0.5
    # test_images -= 0.5

    return train_data_dict['images'], valid_data_dict['images']


def download_cifar10(data_dir):
    data_path = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    url = urllib.request.urlopen(data_path)
    archive = tarfile.open(fileobj=url, mode='r|gz')
    archive.extractall(data_dir)
    url.close()
    archive.close()
    print('extracted data files to %s' % data_dir)


def unpickle(filename):
    with open(filename, 'rb') as fo:
        return cPickle.load(fo, encoding='latin1')


def reshape_flattened_image_batch(flat_image_batch):
    # convert from NCHW to NHWC
    return flat_image_batch.reshape(-1, 3, 32, 32).transpose(
        [0, 2, 3, 1]).astype(np.float32)


def combine_batches(batch_list):
    images = np.vstack([reshape_flattened_image_batch(batch['data'])
                        for batch in batch_list])
    labels = np.vstack([np.array(batch['labels'])
                        for batch in batch_list]).reshape(-1, 1)
    return {'images': images, 'labels': labels}