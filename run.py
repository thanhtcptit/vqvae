import os
import argparse
import logging

from overrides import overrides

from nsds.common import Params
from nsds.common.util import import_submodules
from nsds.commands.subcommand import ArgumentParserWithDefaults, Subcommand

logger = logging.getLogger(__name__)

import_submodules('src')


def main(subcommand_overrides={}):
    parser = ArgumentParserWithDefaults()

    subparsers = parser.add_subparsers(title='Commands', metavar='')

    subcommands = {
        'train': Train(),
        'export': ExportModel(),
        'indexing': ImageIndexing(),
        'viz': VisualizeRecon(),
        'sample': Sample(),
        **subcommand_overrides
    }

    for name, subcommand in subcommands.items():
        subparser = subcommand.add_subparser(name, subparsers)
        if name != "configure":
            subparser.add_argument('--include-package',
                                   type=str,
                                   action='append',
                                   default=[],
                                   help='additional packages to include')

    args = parser.parse_args()

    if 'func' in dir(args):
        for package_name in getattr(args, 'include_package', ()):
            import_submodules(package_name)
        args.func(args)
    else:
        parser.print_help()


class Train(Subcommand):
    def add_subparser(self, name, parser):
        description = '''Train the specified model on the specified dataset.'''
        subparser = parser.add_parser(name, description=description,
                                      help='Train a model.')

        subparser.add_argument(
            'param_path', type=str,
            help='path to parameter file describing the model to be trained')

        subparser.add_argument(
            '-t', '--ttype', choices=[
                'vae', 'vqvae', 'prior', 'emb-triplet', 'emb-softmax'],
            default='vqvae')

        subparser.add_argument(
            '-s', '--serialization-dir', required=True, type=str,
            help='directory in which to save the model and its logs')

        subparser.add_argument(
            '-r', '--recover', action='store_true', default=False,
            help='recover training from the state in serialization_dir')

        subparser.add_argument(
            '-o', '--overrides', type=str, default="",
            help='a JSON structure used to override the experiment config')

        subparser.add_argument(
            '-f', '--force', action='store_true',
            help='force override serialization dir')

        subparser.set_defaults(func=train_model_from_args)

        return subparser


def train_model_from_args(args):
    train_model_from_file(args.param_path, args.ttype, args.serialization_dir,
                          args.overrides, args.recover, args.force)


def train_model_from_file(parameter_filename, ttype, serialization_dir,
                          param_overrides="", recover=False, force=False):
    from src import training
    if ttype == 'vae':
        train_model = training.train_vae
    elif ttype == 'vqvae':
        train_model = training.train_vqvae
    elif ttype == 'prior':
        train_model = training.train_pixelcnn_prior
    elif ttype == 'emb-triplet':
        train_model = training.train_image_embedding_triplet
    else:
        train_model = training.train_image_embedding_softmax
    params = Params.from_file(parameter_filename, param_overrides)
    return train_model(params, serialization_dir, recover, force)


class ExportModel(Subcommand):
    def add_subparser(self, name, parser):
        description = '''Export model for serving or service.'''
        subparser = parser.add_parser(name, description=description,
                                      help='Export model.')

        subparser.add_argument(
            'model_path', type=str, help='path to model checkpoint')

        subparser.add_argument(
            '-e', '--export_type', choices=['compact', 'serving'],
            default='compact', help='choose export mode')

        subparser.add_argument(
            '-o', '--output_dir', type=str, required=True,
            help='path to output dir')

        subparser.set_defaults(func=export_model)

        return subparser


def export_model(args):
    from src.export_model import export
    params_path = os.path.join(os.path.split(args.model_path)[0],
                               'config.json')
    params = Params.from_file(params_path)
    return export(params, args.model_path, args.export_type, args.output_dir)


class ImageIndexing(Subcommand):
    def add_subparser(self, name, parser):
        description = '''Run image indexing.'''
        subparser = parser.add_parser(name, description=description,
                                      help='Run image indexing.')

        subparser.add_argument(
            'model_path', type=str, help='path to model protobuf file')

        subparser.add_argument(
            '-d', '--image_dir', type=str, required=True,
            help='path to image directory')

        subparser.set_defaults(func=index_image)

        return subparser


def index_image(args):
    from src.image_indexing import run_image2vec
    return run_image2vec(args.model_path, args.image_dir)


class Sample(Subcommand):
    def add_subparser(self, name, parser):
        description = '''Run sampling.'''
        subparser = parser.add_parser(name, description=description,
                                      help='Run sampling.')
        subparser.add_argument(
            'model_path', type=str, help='path to model protobuf file')

        subparser.add_argument(
            '-n', '--num_samples', type=int, default=16,
            help='num sample to generate')

        subparser.set_defaults(func=sample_image)

        return subparser


def sample_image(args):
    from src.sampling import sample
    params_path = os.path.join(os.path.split(args.model_path)[0],
                               'config.json')
    params = Params.from_file(params_path)
    return sample(params, args.model_path, args.num_samples)


class VisualizeRecon(Subcommand):
    def add_subparser(self, name, parser):
        description = '''Run reconstruction visualization.'''
        subparser = parser.add_parser(name, description=description,
                                      help='Run reconstruction visualization.')

        subparser.add_argument(
            'model_path', type=str, help='path to model protobuf file')

        subparser.add_argument(
            'image_size', type=int, help='Size of the images')

        subparser.add_argument(
            '-g', '--original_dir', type=str, default='data/original/128x128',
            help='path to image directory')

        subparser.add_argument(
            '-e', '--external_dir', type=str, default='data/test_image',
            help='path to external image directory')

        subparser.set_defaults(func=visualize_recon)

        return subparser


def visualize_recon(args):
    from src.utils.visualize import visualize_v2
    return visualize_v2(args.model_path, args.image_size,
                        args.original_dir, args.external_dir)


def run():
    main()


if __name__ == "__main__":
    run()
