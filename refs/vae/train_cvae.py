import os
import PIL
import time
import glob
import shutil
import argparse
import IPython
import imageio
import zipfile
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from six import iteritems

from models import CVAE
from datasets import get_dataset


tf.enable_eager_execution()

learning_rate = 1e-4
latent_dim = 100
num_examples_to_generate = 16

batch_size = 100
epochs = 100
train_buf = 60000
test_buf = 10000
log_every = 1


def write_arguments_to_file(args, filename):
    with open(filename, 'w') as f:
        for key, value in iteritems(vars(args)):
            f.write('%s: %s\n' % (key, str(value)))


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint_dir', default='cvae-keras')
    parser.add_argument('-d', '--dataset', choices=['mnist', 'cifar10'],
                        default='mnist')
    parser.add_argument('-l', '--loss', choices=['sigmoid_ce', 'mse'],
                        default='sigmoid_ce')

    return parser.parse_args()


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


def compute_loss(model, x, loss_func):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)

    if loss_func == 'sigmoid_ce':
        logpx_z = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=x_logit, labels=x)
    else:
        logpx_z = (x_logit - x) ** 2
    logpx_z = -tf.reduce_sum(logpx_z, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


def reconstruct(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    return x_logit


def compute_gradients(model, x, loss_func):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x, loss_func)
    return tape.gradient(loss, model.trainable_variables), loss


def apply_gradients(optimizer, gradients, variables):
    optimizer.apply_gradients(zip(gradients, variables))


def display_image(epoch_no):
    return PIL.Image.open('epoch_{:04d}.png'.format(epoch_no))


def save_images_fig(images, name, num_rows=4, num_cols=4, num_channel=3):
    fig = plt.figure(figsize=(num_rows, num_cols))
    for i in range(images.shape[0]):
        plt.subplot(num_rows, num_cols, i + 1)
        if num_channel == 1:
            plt.imshow(images[i, :, :, 0], cmap='gray')
        else:
            plt.imshow(images[i, :, :, :])
        plt.axis('off')

    plt.savefig(os.path.join(train_log, 'images', name))
    plt.close()


def generate_and_save_images(model, test_input, name, num_channel=3):
    predictions = model.sample(test_input)
    save_images_fig(predictions, name, num_channel=num_channel)


def view_reconstruction(x_train, x_train_recon, x_test, x_test_recon,
                        save_path, image_shape, num_rows=4, num_cols=4,
                        cmap=None):
    def convert_batch_to_image_grid(image_batch):
        if image_shape[-1] == 1:
            reshaped = \
                (image_batch.reshape(num_rows, num_cols, *image_shape[:2])
                    .transpose(0, 2, 1, 3)
                    .reshape(num_rows * image_shape[0],
                             num_cols * image_shape[1]))
        else:
            reshaped = \
                (image_batch.reshape(num_rows, num_cols, *image_shape)
                    .transpose(0, 2, 1, 3, 4)
                    .reshape(num_rows * image_shape[0],
                             num_cols * image_shape[1], 3))
        return reshaped

    figsize = (num_rows * 2, num_cols * 2)
    f = plt.figure(figsize=figsize)
    ax = f.add_subplot(2, 2, 1)
    ax.imshow(convert_batch_to_image_grid(x_train),
              interpolation='nearest', cmap=cmap)
    ax.set_title('training data originals')
    plt.axis('off')

    ax = f.add_subplot(2, 2, 2)
    ax.imshow(convert_batch_to_image_grid(x_train_recon),
              interpolation='nearest', cmap=cmap)
    ax.set_title('training data reconstructions')
    plt.axis('off')

    ax = f.add_subplot(2, 2, 3)
    ax.imshow(convert_batch_to_image_grid(x_test),
              interpolation='nearest', cmap=cmap)
    ax.set_title('validation data originals')
    plt.axis('off')

    ax = f.add_subplot(2, 2, 4)
    ax.imshow(convert_batch_to_image_grid(x_test_recon),
              interpolation='nearest', cmap=cmap)
    ax.set_title('validation data reconstructions')
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()


def create_gif(del_remain=True):
    anim_file = os.path.join(train_log, 'images', 'cvae.gif')

    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob(os.path.join(train_log, 'images',
                              'sampling_epoch*.png'))
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


def zip_file(file_path, zip_path):
    if os.path.exists(zip_path):
        os.remove(zip_path)

    zipw = zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED)
    for root, dirs, files in os.walk(file_path):
        for f in files:
            zipw.write(os.path.join(root, f))

    zipw.close()


def load_or_create_vectors(file_path):
    if not os.path.exists(file_path):
        sample_vectors = np.random.normal(
            size=(num_examples_to_generate, latent_dim))
        np.save(file_path, sample_vectors)
    else:
        sample_vectors = np.load(file_path)
    return sample_vectors


def create_data_iter(dataset, train_buf, test_buf, batch_size):
    train_images, test_images = get_dataset(dataset)
    input_shape = train_images.shape[1:]

    sample_train = train_images[:num_examples_to_generate]
    sample_test = test_images[:num_examples_to_generate]

    train_dataset = tf.data.Dataset.from_tensor_slices(
        train_images).shuffle(train_buf).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices(
        test_images).batch(batch_size)
    return train_dataset, test_dataset, sample_train, sample_test, input_shape


def training(dataset, train_log, loss_func):
    train_images, test_images, sample_train, sample_test, input_shape = \
        create_data_iter(dataset, train_buf, test_buf, batch_size)
    num_channel = input_shape[-1]

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    sample_vectors = load_or_create_vectors('sample.npy')
    random_vector_for_generation = tf.constant(sample_vectors,
                                               dtype=tf.float32)

    model = CVAE(input_shape, latent_dim)
    model.build_graph()

    generate_and_save_images(model, random_vector_for_generation,
                             'sampling_epoch_0.png', num_channel=num_channel)

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        train_loss = tf.keras.metrics.Mean()
        for train_x in train_images:
            gradients, loss = compute_gradients(model, train_x, loss_func)
            train_loss(loss)
            apply_gradients(optimizer, gradients, model.trainable_variables)
        end_time = time.time()

        train_elbo = -train_loss.result()
        if epoch % log_every == 0:
            loss = tf.keras.metrics.Mean()
            for test_x in test_images:
                loss(compute_loss(model, test_x, loss_func))
            elbo = -loss.result()
            print('Epoch: {}, Train ELBO: {}, Test ELBO: {}, '
                  'Time {}'.format(
                      epoch, train_elbo, elbo, end_time - start_time))
            generate_and_save_images(model, random_vector_for_generation,
                                     'sampling_epoch_{}.png'.format(epoch),
                                     num_channel=num_channel)
            visualize_reconstruction(model, sample_train, sample_test, epoch)

    model.save_weights(os.path.join(train_log, 'model'))

    create_gif(del_remain=False)
    zip_file(os.path.join(train_log, 'images'),
             os.path.join(train_log, 'images.zip'))


def visualize_reconstruction(model, sample_train, sample_test, epoch):
    input_shape = sample_train.shape[1:]

    sample_train_recon = reconstruct(model, sample_train).numpy()
    sample_test_recon = reconstruct(model, sample_test).numpy()

    if input_shape[-1] == 1:
        sample_train_recon[sample_train_recon >= .5] = 1.
        sample_train_recon[sample_train_recon < .5] = 0.
        sample_test_recon[sample_test_recon >= .5] = 1.
        sample_test_recon[sample_test_recon < .5] = 0.

    cmap = None
    if input_shape[-1] == 1:
        cmap = 'gray'
    view_reconstruction(
        sample_train, sample_train_recon, sample_test,
        sample_test_recon,
        os.path.join(train_log, 'images', 'recon_{}.png'.format(epoch)),
        input_shape, cmap=cmap)


if __name__ == '__main__':
    args = _parse_args()
    train_log = os.path.join('train_log', args.checkpoint_dir)
    if os.path.exists(train_log):
        shutil.rmtree(train_log)
    os.makedirs(os.path.join(train_log, 'images'))
    write_arguments_to_file(args, os.path.join(train_log, 'arguments.txt'))

    training(args.dataset, train_log, args.loss)
