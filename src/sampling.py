import os
import imageio
import numpy as np
import tensorflow as tf

from tqdm import tqdm

from src.utils.datasets import load_dataset
from src.utils.tools import zip_file, get_lastest_index
from src.utils.visualize import plot_reconstruction_v2
from src.models.base import BasePixelCNNPrior


def sample(params, model_path, num_samples):
    model_dir = os.path.split(model_path)[0]
    save_dir = os.path.join(model_dir, 'viz')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dataset_params = params['dataset']
    model_params = params['model']
    latent_shape = model_params['latent_shape']
    num_labels = model_params['num_labels']

    model = BasePixelCNNPrior.from_params(model_params)

    data_dir = dataset_params['path']
    labels = os.listdir(data_dir)
    labels = sorted(labels)
    idxs = np.arange(num_labels)
    if len(idxs) < num_samples:
        idxs = list(idxs) * int(np.ceil(num_samples / len(idxs)))
        idxs = np.array(idxs, dtype=np.int32)

    z = tf.placeholder(shape=[num_samples] + latent_shape,
                       dtype=tf.int32, name='z_pre')
    y = tf.placeholder(shape=[None], dtype=tf.int32, name='y')
    z_gen = model.sample(z, y)
    images_gen = model.decode(z)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    saver = tf.train.Saver()
    saver.restore(sess, model_path)
    lastest_index = get_lastest_index(save_dir)
    for i in tqdm(range(5)):
        np.random.shuffle(idxs)
        labels_sample = idxs[:num_samples]

        lastest_index += 1
        org_images = []
        for i in labels_sample:
            label_dir = os.path.join(data_dir, labels[i])
            image_path = os.path.join(label_dir, os.listdir(label_dir)[0])
            org_images.append(imageio.imread(image_path))

        z_np = np.zeros(shape=[num_samples] + latent_shape)
        for i in range(latent_shape[0]):
            for j in range(latent_shape[1]):
                _z_gen = sess.run(z_gen, {z: z_np, y: labels_sample})
                z_np[:, i, j] = _z_gen[:, i, j]
        images_gen_np = sess.run(images_gen, {z: z_np})
        plot_reconstruction_v2(org_images, images_gen_np, os.path.join(
            save_dir, f'samples_{lastest_index}.png'))

    zip_file(save_dir, os.path.join(model_dir, 'viz.zip'))
