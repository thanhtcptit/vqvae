import os
import glob
import shutil
import imageio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image

from nsds.common.util import import_submodules

from src.utils.datasets import load_toy_dataset, preprocess
from src.utils.tools import zip_file


def convert_image_to_valid_range(images):
    if not isinstance(images, np.ndarray):
        images = np.array(images, dtype=np.float32)
    if np.max(images) > 1:
        return images
    images = ((images + 0.5) * 255).astype(int)
    images[images > 255] = 255
    images[images < 0] = 0
    return images


def save_images(images, save_path, num_images, num_channel=3):
    num_rows = num_cols = int(num_images ** 0.5)
    fig = plt.figure(figsize=(num_rows, num_cols))
    num_channel = images.shape[-1]

    images = convert_image_to_valid_range(images)
    for i in range(images.shape[0]):
        plt.subplot(num_rows, num_cols, i + 1)
        if num_channel == 1:
            plt.imshow(images[i, :, :, 0], cmap='gray')
        else:
            plt.imshow(images[i, :, :, :])
        plt.axis('off')

    plt.savefig(save_path)
    plt.close()


def create_gif(regex, gif_name, save_path, del_remain=True):
    anim_file = os.path.join(save_path, gif_name)

    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob(os.path.join(save_path, regex))
        filenames = sorted(filenames)
        last = -1
        for i, filename in enumerate(filenames):
            frame = 2 * (i ** 0.5)
            if round(frame) > round(last):
                last = frame
                image = imageio.imread(filename)
                writer.append_data(image)
            if del_remain and filename != filenames[-1]:
                os.remove(filename)

        image = imageio.imread(filename)
        writer.append_data(image)


def convert_batch_to_image_grid(image_batch):
    image_shape = image_batch.shape[1:]
    num_rows = num_cols = int(image_batch.shape[0] ** 0.5)
    if image_shape[-1] == 1:
        reshaped = (image_batch.reshape(num_rows, num_cols, *image_shape[:2])
                    .transpose(0, 2, 1, 3).reshape(num_rows * image_shape[0],
                                                   num_cols * image_shape[1]))
    else:
        reshaped = (image_batch.reshape(num_rows, num_cols, *image_shape)
                    .transpose(0, 2, 1, 3, 4).reshape(
            num_rows * image_shape[0], num_cols * image_shape[1], 3))
    return convert_image_to_valid_range(reshaped)


def plot_reconstruction_v1(x_train, x_train_recon, x_test, x_test_recon,
                           save_path):
    image_shape = x_train.shape[1:]
    num_rows = num_cols = int(x_train.shape[0] ** 0.5)
    cmap = 'gray' if image_shape[-1] == 1 else None

    figsize = (num_rows * 2, num_cols * 2)
    f = plt.figure(figsize=figsize)
    for i, batch, title in zip(
            [1, 2, 3, 4], [x_train, x_train_recon, x_test, x_test_recon],
            ['train org', 'train recon', 'val org', 'val recon']):
        ax = f.add_subplot(2, 2, i)
        ax.imshow(convert_batch_to_image_grid(batch),
                  interpolation='nearest', cmap=cmap)
        ax.set_title(title)
        plt.axis('off')

    plt.savefig(save_path)
    plt.close()


def plot_reconstruction_v2(x, x_recon, save_path):
    if not isinstance(x, np.ndarray):
        x = np.array(x)

    image_shape = x.shape[1:]
    num_rows = num_cols = int(x.shape[0] ** 0.5)
    cmap = 'gray' if image_shape[-1] == 1 else None

    figsize = (num_rows * 2, num_cols * 2)
    f = plt.figure(figsize=figsize)
    for i, batch, title in zip([1, 2], [x, x_recon], ['org', 'recon']):
        ax = f.add_subplot(1, 2, i)
        ax.imshow(convert_batch_to_image_grid(batch),
                  interpolation='nearest', cmap=cmap)
        ax.set_title(title)
        plt.axis('off')

    plt.savefig(save_path)
    plt.close()


def get_samples(data_dir, image_size, num_samples=64, seed=42):
    # np.random.seed(seed)
    samples = []
    list_images = glob.glob(os.path.join(data_dir, '*/*'))
    np.random.shuffle(list_images)
    for i in range(num_samples):
        image = Image.open(list_images[i]).resize((image_size, image_size))
        samples.append(preprocess(np.array(image)))
    return np.array(samples, dtype=np.float32)


def visualize_v1(model_path):
    model_dir = os.path.split(model_path)[0]
    _, _, sample_train, sample_test = load_toy_dataset('cifar10')

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    graph_def = tf.GraphDef()
    with tf.gfile.GFile(model_path, "rb") as graphFile:
        graph_def.ParseFromString(graphFile.read())
    tf.import_graph_def(graph_def)

    input_images = sess.graph.get_tensor_by_name('import/input:0')
    recon_images = sess.graph.get_tensor_by_name('import/x_recon:0')

    sample_train_recon = sess.run(recon_images, {input_images: sample_train})
    sample_test_recon = sess.run(recon_images, {input_images: sample_test})
    samples_recon = sess.run(recon_images, {input_images: samples})
    sess.close()

    plot_reconstruction_v1(
        sample_train, sample_train_recon, sample_test, sample_test_recon,
        os.path.join(model_dir, 'recon.png'))


def visualize_v2(model_path, image_size, original_dir, external_dir,
                 num_item_per_img=16, num_images=8):
    model_dir = os.path.split(model_path)[0]
    viz_dir = os.path.join(model_dir, 'viz')
    if os.path.exists(viz_dir):
        shutil.rmtree(viz_dir)
    os.makedirs(viz_dir)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    graph_def = tf.GraphDef()
    with tf.gfile.GFile(model_path, "rb") as graphFile:
        graph_def.ParseFromString(graphFile.read())
    tf.import_graph_def(graph_def)

    input_images = sess.graph.get_tensor_by_name('import/input:0')
    recon_images = sess.graph.get_tensor_by_name('import/x_recon:0')

    samples = get_samples(original_dir, image_size,
                          num_samples=num_item_per_img * num_images)
    for i in range(num_images):
        sample_batch = samples[i * num_item_per_img:
                               (i + 1) * num_item_per_img]
        samples_recon = sess.run(recon_images, {
            input_images: sample_batch})

        plot_reconstruction_v2(
            sample_batch, samples_recon, os.path.join(
                viz_dir, 'original_recon_{}.png'.format(i)))

    samples = get_samples(external_dir, image_size,
                          num_samples=num_item_per_img * num_images)
    for i in range(num_images):
        sample_batch = samples[i * num_item_per_img:
                               (i + 1) * num_item_per_img]
        samples_recon = sess.run(recon_images, {
            input_images: sample_batch})

        plot_reconstruction_v2(
            sample_batch, samples_recon, os.path.join(
                viz_dir, 'external_recon_{}.png'.format(i)))
    sess.close()
    zip_file(viz_dir, os.path.join(model_dir, 'viz.zip'))
