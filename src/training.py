import os
import shutil
import imageio
import numpy as np
import tensorflow as tf

from tensorpack import PrefetchDataZMQ, MultiThreadMapData, \
    BatchData, AutoResumeTrainConfig, QueueInput
from tensorpack.train import SyncMultiGPUTrainerParameterServer, \
    SimpleTrainer, QueueInputTrainer, launch_train_with_config
from tensorpack.utils import logger
from tensorpack.callbacks import ModelSaver, InferenceRunner, \
    MinSaver, ScalarStats, ClassificationError

from src.utils.datasets import *
from src.models.base import *
from src.callbacks.sampling import *
from src.callbacks.utils import *


def get_dataflow(dataset_path, binarizer, train_val_split,
                 batch_size, num_parallel, num_sample=16, seed=42):
    np.random.seed(seed)

    train_images, train_labels, val_images, val_labels = \
        load_dataset(dataset_path, train_val_split)

    train_ds = ImageDataflow(train_images, train_labels, binarizer, True)
    train_ds = PrefetchDataZMQ(train_ds, nr_proc=num_parallel)
    train_ds = BatchData(train_ds, batch_size, remainder=False)

    def load_img(dp):
        image_path, label = dp
        return [preprocess(imageio.imread(image_path), binarizer), label]

    val_ds = ImagePathDataflow(val_images, val_labels)
    val_ds = MultiThreadMapData(val_ds, num_parallel, load_img,
                                buffer_size=2000, strict=True)
    val_ds = BatchData(val_ds, batch_size, remainder=True)
    val_ds = PrefetchDataZMQ(val_ds, 1)

    sample_train = []
    sample_train_labels = []
    sample_val = []
    sample_val_labels = []
    train_idxs = np.arange(len(train_images))
    val_idxs = np.arange(len(val_images))
    np.random.shuffle(train_idxs)
    np.random.shuffle(val_idxs)

    for i in range(num_sample):
        sample_train.append(preprocess(imageio.imread(
            train_images[train_idxs[i]]), binarizer))
        sample_train_labels.append(train_labels[train_idxs[i]])

        sample_val.append(preprocess(imageio.imread(
            val_images[val_idxs[i]]), binarizer))
        sample_val_labels.append(val_labels[val_idxs[i]])

    return train_ds, val_ds, np.array(sample_train), np.array(sample_val), \
        sample_train_labels, sample_val_labels


def get_triplet_dataflow(dataset_path, items_per_batch,
                         images_per_item, num_parallel, seed=42):
    np.random.seed(seed)
    batch_size = items_per_batch * images_per_item

    train_images, train_labels, val_images, val_labels = \
        load_dataset(dataset_path, 0.)

    train_ds = TripletDataflow(train_images, train_labels, items_per_batch,
                               images_per_item, True)
    train_ds = PrefetchDataZMQ(train_ds, nr_proc=num_parallel)
    train_ds = BatchData(train_ds, batch_size, remainder=False)

    return train_ds


def train_vae(params, checkpoint_dir, recover=True, force=False):
    if force and os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    logger.set_logger_dir(checkpoint_dir)

    dataset_params = params['dataset']
    model_params = params['model']
    trainer_params = params['trainer']

    train_ds, val_ds, sample_train, sample_val, _, _ = \
        get_dataflow(dataset_params['path'],
                     dataset_params['binarizer'],
                     dataset_params['train_val_split'],
                     trainer_params['batch_size'],
                     trainer_params['num_parallel'])

    params.to_file(os.path.join(logger.get_logger_dir(), 'config.json'))

    latent_dim = model_params['latent_dim']
    model = BaseVAE.from_params(model_params)

    trainer = SyncMultiGPUTrainerParameterServer(
        gpus=trainer_params['num_gpus'], ps_device=None)
    trainer_config = AutoResumeTrainConfig(
        always_resume=recover,
        model=model,
        dataflow=train_ds,
        callbacks=[
            Sampling(model, trainer_params['num_examples_to_generate'],
                     latent_dim, os.path.join(checkpoint_dir, 'images')),
            Reconstruct(model, sample_train, sample_val,
                        os.path.join(checkpoint_dir, 'images')),
            ModelSaver(max_to_keep=5, checkpoint_dir=checkpoint_dir),
            InferenceRunner(input=val_ds,
                            infs=ScalarStats(['avg_logpx_z', 'neg_elbo'])),
            MinSaver(monitor_stat='validation_neg_elbo'),
            CompressResource(os.path.join(checkpoint_dir, 'images'),
                             os.path.join(checkpoint_dir, 'images.zip')),
            Notification('Training status', 'Complete')
        ],
        steps_per_epoch=trainer_params['steps_per_epoch'],
        max_epoch=trainer_params['max_epochs']
    )
    launch_train_with_config(trainer_config, trainer)


def train_vqvae(params, checkpoint_dir, recover=True, force=False):
    if force and os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    logger.set_logger_dir(checkpoint_dir)

    dataset_params = params['dataset']
    model_params = params['model']
    trainer_params = params['trainer']

    train_ds, val_ds, sample_train, sample_val, _, _ = \
        get_dataflow(dataset_params['path'], False,
                     dataset_params['train_val_split'],
                     trainer_params['batch_size'],
                     trainer_params['num_parallel'])

    params.to_file(os.path.join(logger.get_logger_dir(), 'config.json'))

    model = BaseVQVAE.from_params(model_params)

    trainer = SyncMultiGPUTrainerParameterServer(
        gpus=trainer_params['num_gpus'], ps_device=None)
    trainer_config = AutoResumeTrainConfig(
        always_resume=recover,
        model=model,
        dataflow=train_ds,
        callbacks=[
            Reconstruct(model, sample_train, sample_val,
                        os.path.join(checkpoint_dir, 'images')),
            ModelSaver(max_to_keep=5, checkpoint_dir=checkpoint_dir),
            InferenceRunner(input=val_ds,
                            infs=ScalarStats(['loss', 'perplexity'])),
            MinSaver(monitor_stat='validation_loss'),
            CompressResource(os.path.join(checkpoint_dir, 'images'),
                             os.path.join(checkpoint_dir, 'images.zip')),
            Notification('Training status', 'Complete')
        ],
        steps_per_epoch=trainer_params['steps_per_epoch'],
        max_epoch=trainer_params['max_epochs']
    )
    launch_train_with_config(trainer_config, trainer)


def train_pixelcnn_prior(params, checkpoint_dir, recover=True, force=False):
    if force and os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    logger.set_logger_dir(checkpoint_dir)

    dataset_params = params['dataset']
    model_params = params['model']
    trainer_params = params['trainer']

    train_ds, val_ds, sample_train, sample_val, sample_train_label, \
        sample_val_label = get_dataflow(
            dataset_params['path'], False,
            dataset_params['train_val_split'], trainer_params['batch_size'],
            trainer_params['num_parallel'])

    vqvae_checkpoint_path = trainer_params['vqvae_checkpoint_path']
    vqvae_config_path = os.path.join(os.path.split(vqvae_checkpoint_path)[0],
                                     'config.json')
    model_params['vqvae_model_params'] = vqvae_config_path

    latent_shape = model_params['latent_shape']
    num_labels = model_params['num_labels']

    params.to_file(os.path.join(logger.get_logger_dir(), 'config.json'))

    model = BasePixelCNNPrior.from_params(model_params)

    trainer = SyncMultiGPUTrainerParameterServer(
        gpus=trainer_params['num_gpus'], ps_device=None)
    trainer_config = AutoResumeTrainConfig(
        always_resume=recover,
        model=model,
        dataflow=train_ds,
        callbacks=[
            SequentialSampling(trainer_params['num_examples_to_generate'],
                               latent_shape, num_labels, model,
                               os.path.join(checkpoint_dir, 'images')),
            Reconstruct(model, sample_train, sample_val,
                        os.path.join(checkpoint_dir, 'images'),
                        sample_train_label, sample_val_label),
            ModelSaver(max_to_keep=5, checkpoint_dir=checkpoint_dir),
            InferenceRunner(input=val_ds,
                            infs=ScalarStats(['loss'])),
            MinSaver(monitor_stat='validation_loss'),
            CompressResource(os.path.join(checkpoint_dir, 'images'),
                             os.path.join(checkpoint_dir, 'images.zip')),
            RestoreWeights(vqvae_checkpoint_path),
            Notification('Training status', 'Complete')
        ],
        steps_per_epoch=trainer_params['steps_per_epoch'],
        max_epoch=trainer_params['max_epochs']
    )
    launch_train_with_config(trainer_config, trainer)


def train_image_embedding_triplet(params, checkpoint_dir, recover=True,
                                  force=False):
    if force and os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    logger.set_logger_dir(checkpoint_dir)

    dataset_params = params['dataset']
    model_params = params['model']
    trainer_params = params['trainer']

    train_ds = get_triplet_dataflow(
        dataset_params['path'], trainer_params['items_per_batch'],
        trainer_params['images_per_item'], trainer_params['num_parallel'])

    vqvae_checkpoint_path = trainer_params['vqvae_checkpoint_path']
    vqvae_config_path = os.path.join(os.path.split(vqvae_checkpoint_path)[0],
                                     'config.json')
    model_params['vqvae_model_params'] = vqvae_config_path

    params.to_file(os.path.join(logger.get_logger_dir(), 'config.json'))

    model = BaseImageEmbedding.from_params(model_params)

    trainer = SyncMultiGPUTrainerParameterServer(
        gpus=trainer_params['num_gpus'], ps_device=None)
    trainer_config = AutoResumeTrainConfig(
        always_resume=recover,
        model=model,
        dataflow=train_ds,
        callbacks=[
            ModelSaver(max_to_keep=5, checkpoint_dir=checkpoint_dir),
            MinSaver(monitor_stat='loss'),
            RestoreWeights(vqvae_checkpoint_path),
            SendStat('Training status', ['loss', 'pos_triplet_frac'],
                     after_every=2),
            Notification('Training status', 'Complete')
        ],
        steps_per_epoch=trainer_params['steps_per_epoch'],
        max_epoch=trainer_params['max_epochs']
    )
    launch_train_with_config(trainer_config, trainer)


def train_image_embedding_softmax(params, checkpoint_dir, recover=True,
                                  force=False):
    if force and os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    logger.set_logger_dir(checkpoint_dir)

    dataset_params = params['dataset']
    model_params = params['model']
    trainer_params = params['trainer']

    train_ds, val_ds, _, _, _, _ = get_dataflow(
        dataset_params['path'], False, dataset_params['train_val_split'],
        trainer_params['batch_size'], trainer_params['num_parallel'])

    vqvae_checkpoint_path = trainer_params['vqvae_checkpoint_path']
    vqvae_config_path = os.path.join(os.path.split(vqvae_checkpoint_path)[0],
                                     'config.json')
    model_params['vqvae_model_params'] = vqvae_config_path

    params.to_file(os.path.join(logger.get_logger_dir(), 'config.json'))

    model = BaseImageEmbedding.from_params(model_params)

    trainer = SyncMultiGPUTrainerParameterServer(
        gpus=trainer_params['num_gpus'], ps_device=None)
    trainer_config = AutoResumeTrainConfig(
        always_resume=recover,
        model=model,
        dataflow=train_ds,
        callbacks=[
            InferenceRunner(input=val_ds, infs=[
                ScalarStats('loss'),
                ClassificationError('correct_prediction',
                                    'val-correct_prediction')]),
            ModelSaver(max_to_keep=5, checkpoint_dir=checkpoint_dir),
            MinSaver(monitor_stat='val-correct_prediction'),
            RestoreWeights(vqvae_checkpoint_path),
            SendStat('Training status', [
                'loss', 'accuracy',
                'validation_loss', 'val-correct_prediction'],
                after_every=2),
            Notification('Training status', 'Complete')
        ],
        steps_per_epoch=trainer_params['steps_per_epoch'],
        max_epoch=trainer_params['max_epochs']
    )
    launch_train_with_config(trainer_config, trainer)
