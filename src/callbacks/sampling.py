import os
import numpy as np
import tensorflow as tf

from tensorpack import Callback

from src.utils.visualize import create_gif, plot_reconstruction_v1, save_images


def load_or_create_vectors(file_path, latent_dim, num_examples_to_generate=16):
    if not os.path.exists(file_path):
        sample_vectors = np.random.normal(
            size=(num_examples_to_generate, latent_dim))
        np.save(file_path, sample_vectors)
    else:
        sample_vectors = np.load(file_path)
    return sample_vectors


class Sampling(Callback):
    def __init__(self, model, num_images, latent_dim, save_path):
        super(Sampling, self).__init__()
        self._model = model
        self._sample = None
        self._latent_dim = latent_dim
        self._num_images = num_images

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self._save_path = save_path
        self._img_save_path = os.path.join(save_path, 'sampling_epoch_{}.png')

    def _setup_graph(self):
        random_vector = tf.constant(
            load_or_create_vectors(os.path.join(self._save_path, 'sample.npy'),
                                   self._latent_dim), dtype=tf.float32)
        self._sample = self._model.sample(random_vector)

    def _before_train(self):
        self._trigger()

    def _trigger_epoch(self):
        if self.epoch_num % 2 == 0:
            self._trigger()

    def _trigger(self):
        sample_images = self.trainer.sess.run(self._sample)
        save_images(sample_images, self._img_save_path.format(self.epoch_num),
                    self._num_images)

    def _after_train(self):
        create_gif('sampling_epoch*.png', 'sampling.gif',
                   self._save_path)


class Reconstruct(Callback):
    def __init__(self, model, train_images, test_images, save_path,
                 train_labels=None, test_labels=None):
        super(Reconstruct, self).__init__()
        self._model = model

        self._train_images = train_images
        self._test_images = test_images
        self._train_labels = train_labels
        self._test_labels = test_labels

        self._x_train_recon = None
        self._x_test_recon = None

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self._save_path = save_path
        self._img_save_path = os.path.join(save_path, 'recon_epoch_{}.png')

    def _setup_graph(self):
        train_images = tf.constant(self._train_images, dtype=tf.float32)
        test_images = tf.constant(self._test_images, dtype=tf.float32)
        if self._model.name() == 'vae':
            _, self._x_train_recon = self._model.reconstruct(train_images)
            _, self._x_test_recon = self._model.reconstruct(test_images)
        elif self._model.name() == 'vqvae':
            _, _, self._x_train_recon = self._model.reconstruct(train_images)
            _, _, self._x_test_recon = self._model.reconstruct(test_images)
        else:
            _, z_train_logits = self._model.reconstruct(
                train_images, self._train_labels)
            self._x_train_recon = self._model.decode(
                self._model.sample_from_dist(z_train_logits))

            _, z_test_logits = self._model.reconstruct(
                test_images, self._test_labels)
            self._x_test_recon = self._model.decode(
                self._model.sample_from_dist(z_test_logits))

    def _before_train(self):
        self._trigger()

    def _trigger_epoch(self):
        if self.epoch_num % 2 == 0:
            self._trigger()

    def _trigger(self):
        x_train_recon_np = self.trainer.sess.run(self._x_train_recon)
        x_test_recon_np = self.trainer.sess.run(self._x_test_recon)
        if self._train_images.shape[-1] == 1:
            x_train_recon_np[x_train_recon_np < 0.5] = 0.
            x_train_recon_np[x_train_recon_np >= 0.5] = 1.
            x_test_recon_np[x_test_recon_np < 0.5] = 0.
            x_test_recon_np[x_test_recon_np >= 0.5] = 1.

        plot_reconstruction_v1(self._train_images, x_train_recon_np,
                               self._test_images, x_test_recon_np,
                               self._img_save_path.format(self.epoch_num))

    def _after_train(self):
        create_gif('recon_epoch*.png', 'recon.gif',
                   self._save_path, False)


class SequentialSampling(Callback):
    def __init__(self, num_images, latent_shape, num_labels, model, save_path):
        super(SequentialSampling, self).__init__()
        self._model = model

        self._num_labels = num_labels
        self._latent_shape = latent_shape
        self._num_images = num_images

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self._save_path = save_path
        self._img_save_path = os.path.join(save_path, 'sample_epoch_{}.png')

        idxs = np.arange(self._num_labels)
        np.random.shuffle(idxs)
        if len(idxs) < num_images:
            idxs = list(idxs) * int(np.ceil(num_images / len(idxs)))
            idxs = np.array(idxs, dtype=np.int32)
        self._labels_sample = idxs[:num_images]

    def _setup_graph(self):
        self._z = tf.placeholder(
            shape=[self._num_images] + self._latent_shape,
            dtype=tf.int32, name='z_pre')
        self._z_gen = self._model.sample(self._z, self._labels_sample)
        self._images_gen = self._model.decode(self._z)

    def _trigger_epoch(self):
        if self.epoch_num % 2 == 0:
            self._trigger()

    def _trigger(self):
        z = np.zeros(shape=[self._num_images] + self._latent_shape)
        for i in range(self._latent_shape[0]):
            for j in range(self._latent_shape[1]):
                z_gen = self.trainer.sess.run(self._z_gen, {self._z: z})
                z[:, i, j] = z_gen[:, i, j]
        images_gen = self.trainer.sess.run(self._images_gen, {self._z: z})
        save_images(images_gen, self._img_save_path.format(self.epoch_num),
                    self._num_images)
