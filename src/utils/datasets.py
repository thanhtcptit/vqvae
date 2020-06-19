import os
import imageio
import numpy as np

from collections import defaultdict
from six.moves import cPickle
from tensorpack.dataflow import RNGDataFlow, PrefetchDataZMQ, BatchData


class ImagePathDataflow(RNGDataFlow):
    def __init__(self, image_path_list, label_list, shuffle=False):
        self._image_path_list = image_path_list
        self._label_list = label_list
        self._shuffle = shuffle

        assert len(image_path_list) == len(label_list)

    def __len__(self):
        return len(self._label_list)

    def __iter__(self):
        idxs = np.arange(len(self._label_list))
        if self._shuffle:
            self.rng.shuffle(idxs)
        for i in idxs:
            image_path, label = self._image_path_list[i], self._label_list[i]
            yield [image_path, label]


class ImageDataflow(ImagePathDataflow):
    def __init__(self, image_path_list, label_list,
                 binarizer, shuffle=False):
        super(ImageDataflow, self).__init__(
            image_path_list, label_list, shuffle)
        self._binarizer = binarizer

    def __iter__(self):
        for image_path, label in super(ImageDataflow, self).__iter__():
            image = imageio.imread(image_path)
            yield [preprocess(image, self._binarizer), label]


class TripletDataflow(ImagePathDataflow):
    def __init__(self, image_path_list, label_list, items_per_batch,
                 images_per_item, shuffle=False):
        super(TripletDataflow, self).__init__(
            image_path_list, label_list, shuffle)
        self._image_path_dict = defaultdict(lambda: [])
        for path, label in zip(image_path_list, label_list):
            self._image_path_dict[label].append(path)

        self._classes = list(self._image_path_dict.keys())

        self._items_per_batch = items_per_batch
        self._images_per_item = images_per_item
        self._nrof_images_per_batch = items_per_batch * images_per_item

    def __iter__(self):
        for _ in range(1000):
            classes_idx = np.arange(len(self._classes))
            np.random.shuffle(classes_idx)
            classes_idx = classes_idx[:self._items_per_batch]
            images_path = []
            labels = []
            for i in classes_idx:
                num_images_from_class = len(self._image_path_dict[i])
                images_idx = np.arange(num_images_from_class)
                np.random.shuffle(images_idx)
                num_training_images = min(
                    [num_images_from_class, self._images_per_item,
                     self._nrof_images_per_batch - len(images_path)])
                images_path += [self._image_path_dict[i][p]
                                for p in images_idx[:num_training_images]]
                labels += [i] * num_training_images
            for image_path, label in zip(images_path, labels):
                image = imageio.imread(image_path)
                yield [preprocess(image, False), label]


class ToyDataflow(RNGDataFlow):
    def __init__(self, images, shuffle=False):
        super(ToyDataflow, self).__init__()
        self._images = images
        self._shuffle = shuffle

    def __len__(self):
        return len(self._images)

    def __iter__(self):
        idxs = np.arange(len(self._images))
        if self._shuffle:
            self.rng.shuffle(idxs)
        for i in idxs:
            image_data = self._images[i]
            yield [image_data]


def load_dataset(dataset_path, train_val_split):
    train_images, train_labels, val_images, val_labels = [], [], [], []
    list_class = sorted(os.listdir(dataset_path))
    for i, folder in enumerate(list_class):
        folder_path = os.path.join(dataset_path, folder)
        image_list = [os.path.join(folder_path, image)
                      for image in os.listdir(folder_path)]
        val_idx = int(len(image_list) * (1 - train_val_split))
        train_images += image_list[:val_idx]
        train_labels += [i] * val_idx
        val_images += image_list[val_idx:]
        val_labels += [i] * (len(image_list) - val_idx)

    return train_images, train_labels, val_images, val_labels


def load_images(image_paths, do_preprocess=True):
    images_array = []
    for path in image_paths:
        images_array.append(imageio.imread(path))
    images_array = np.array(images_array, dtype=np.float32)
    if do_preprocess:
        images_array = preprocess(images_array)
    return images_array


def preprocess(images, binarizer=False):
    image_data = images / 255. - 0.5
    if binarizer:
        image_data[image_data < 0.] = 0.
        image_data[image_data >= 0.] = 1.
    return image_data


def load_toy_dataset(name, batch_size=32, num_parallel=1, num_sample=16):
    if name == 'mnist':
        (train_images, _), (test_images, _) = \
            tf.keras.datasets.mnist.load_data()

        train_images = train_images.reshape(
            [train_images.shape[0], 28, 28, 1]).astype('float32')
        test_images = test_images.reshape(
            [test_images.shape[0], 28, 28, 1]).astype('float32')

        train_images /= 255.
        test_images /= 255.

        train_images[train_images >= .5] = 1.
        train_images[train_images < .5] = 0.
        test_images[test_images >= .5] = 1.
        test_images[test_images < .5] = 0.
    elif name == 'cifar10':
        def unpickle(filename):
            with open(filename, 'rb') as fo:
                return cPickle.load(fo, encoding='latin1')

        def reshape_flattened_image_batch(flat_image_batch):
            return flat_image_batch.reshape(-1, 3, 32, 32).transpose(
                [0, 2, 3, 1])

        def combine_batches(batch_list):
            images = np.vstack([reshape_flattened_image_batch(batch['data'])
                                for batch in batch_list])
            labels = np.vstack([np.array(batch['labels'])
                                for batch in batch_list]).reshape(-1, 1)
            return {'images': images, 'labels': labels}

        local_data_dir = 'data/cifar10'
        train_data_dict = combine_batches([unpickle(os.path.join(
            local_data_dir, 'cifar-10-batches-py/data_batch_%d' % i))
            for i in range(1, 5)])
        test_data_dict = combine_batches([unpickle(
            os.path.join(local_data_dir, 'cifar-10-batches-py/data_batch_5'))])

        train_images = (train_data_dict['images'] / 255.) - 0.5
        test_images = (test_data_dict['images'] / 255.) - 0.5
    else:
        raise ValueError('Cant recognize dataset')

    train_ds = ToyDataflow(train_images, True)
    # train_ds = PrefetchDataZMQ(train_ds, nr_proc=num_parallel)
    train_ds = BatchData(train_ds, batch_size, remainder=False)

    val_ds = ToyDataflow(test_images, False)
    val_ds = BatchData(val_ds, batch_size, remainder=True)
    # val_ds = PrefetchDataZMQ(val_ds, 1)

    sample_train = train_images[:num_sample]
    sample_test = test_images[:num_sample]

    return train_ds, val_ds, sample_train, sample_test
