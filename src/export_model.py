import os
import argparse
import numpy as np

from tensorpack.predict import PredictConfig
from tensorpack.tfutils import get_model_loader
from tensorpack.tfutils.export import ModelExporter

from src.models.base import BaseVQVAE


def export(params, model_path, export_type, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_params = params['model']
    model = BaseVQVAE.from_params(model_params)

    pred_config = PredictConfig(
        session_init=get_model_loader(model_path),
        model=model,
        input_names=['input'],
        output_names=['x_recon', 'embeddings', 'latent_zq'])
    if export_type == 'compact':
        checkpoint_name = os.path.split(model_path)[1]
        ModelExporter(pred_config).export_compact(
            os.path.join(output_dir, 'model.pb'))
    else:
        ModelExporter(pred_config).export_serving(
            os.path.join(output_dir, 'exported'))
