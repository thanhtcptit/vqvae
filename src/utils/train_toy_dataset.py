import os
import argparse
import numpy as np
import tensorflow as tf

from tensorpack import BatchData, AutoResumeTrainConfig, QueueInput
from tensorpack.train import SimpleTrainer, launch_train_with_config
from tensorpack.utils import logger
from tensorpack.callbacks import ModelSaver, InferenceRunner, MaxSaver, \
    ScalarStats

from nsds.common import Params
from nsds.common.util import import_submodules
import_submodules('src')

from src.utils.datasets import *
from src.models.base import BaseVAE, BaseVQVAE
from src.callbacks.sampling import SamplingAfterEpoch, Reconstruct
from src.callbacks.utils import CompressResource


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['vae', 'vqvae'],
                        default='vae')
    parser.add_argument('--checkpoint_dir', default='train_log/vae-cifar10')
    parser.add_argument('--dataset', choices=['mnist', 'cifar10'],
                        default='cifar10')
    parser.add_argument('--config_path',
                        default='config/vae_training_cifar.json')

    return parser.parse_args()


def train_vae(params, dataset, checkpoint_dir):
    logger.set_logger_dir(checkpoint_dir)

    dataset_params = params['dataset']
    model_params = params['model']
    trainer_params = params['trainer']

    train_ds, val_ds, sample_train, sample_test = load_toy_dataset(
        dataset, trainer_params['batch_size'],
        trainer_params['num_parallel'])
    params.to_file(os.path.join(logger.get_logger_dir(), 'config.json'))

    latent_dim = model_params['latent_dim']
    model = BaseVAE.from_params(model_params)

    trainer_config = AutoResumeTrainConfig(
        always_resume=recover,
        model=model,
        dataflow=train_ds,
        callbacks=[
            SamplingAfterEpoch(model,
                               trainer_params['num_examples_to_generate'],
                               latent_dim,
                               os.path.join(checkpoint_dir, 'images')),
            Reconstruct(model, sample_train, sample_test,
                        os.path.join(checkpoint_dir, 'images')),
            ModelSaver(max_to_keep=5, checkpoint_dir=checkpoint_dir),
            InferenceRunner(input=val_ds,
                            infs=ScalarStats(['avg_logpx_z', 'cost'])),
            MaxSaver(monitor_stat='validation_cost'),
            CompressResource(os.path.join(checkpoint_dir, 'images'),
                             os.path.join(checkpoint_dir, 'images.zip'))
        ],
        steps_per_epoch=trainer_params['steps_per_epoch'],
        max_epoch=trainer_params['max_epochs']
    )
    launch_train_with_config(trainer_config, SimpleTrainer())


def train_vqvae(params, dataset, checkpoint_dir):
    logger.set_logger_dir(checkpoint_dir)

    dataset_params = params['dataset']
    model_params = params['model']
    trainer_params = params['trainer']
    image_shape = model_params['image_shape']

    train_ds, val_ds, sample_train, sample_test = load_toy_dataset(
        dataset, trainer_params['batch_size'],
        trainer_params['num_parallel'])

    params.to_file(os.path.join(logger.get_logger_dir(), 'config.json'))

    model = BaseVQVAE.from_params(model_params)

    trainer_config = AutoResumeTrainConfig(
        always_resume=recover,
        model=model,
        dataflow=train_ds,
        callbacks=[
            Reconstruct(model, sample_train, sample_test,
                        os.path.join(checkpoint_dir, 'images')),
            ModelSaver(max_to_keep=5, checkpoint_dir=checkpoint_dir),
            InferenceRunner(input=val_ds,
                            infs=ScalarStats(['loss', 'perplexity'])),
            MaxSaver(monitor_stat='validation_loss'),
            CompressResource(os.path.join(checkpoint_dir, 'images'),
                             os.path.join(checkpoint_dir, 'images.zip'))
        ],
        steps_per_epoch=trainer_params['steps_per_epoch'],
        max_epoch=trainer_params['max_epochs']
    )
    launch_train_with_config(trainer_config, SimpleTrainer())


if __name__ == '__main__':
    args = _parse_args()
    params = Params.from_file(args.config_path, '')
    if args.model == 'vae':
        train_vae(params, args.dataset, args.checkpoint_dir)
    else:
        train_vqvae(params, args.dataset, args.checkpoint_dir)
