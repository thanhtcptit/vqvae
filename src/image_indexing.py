import os
import math
import json
import argparse
import itertools
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from collections import defaultdict

from nsds.common import Params

from src.models.base import BaseVQVAE
from src.utils.datasets import preprocess, load_images
from src.utils.tools import append_json


def load_images_data(path):
    user_image_dict = defaultdict(list)
    for user in tqdm(os.listdir(path), desc='User'):
        user_dir = os.path.join(path, user)
        if not os.path.isdir(user_dir):
            continue
        for image in os.listdir(user_dir):
            image_path = os.path.join(user_dir, image)
            user_image_dict[user].append(image_path)

    return user_image_dict


def run_image2vec(model_path, image_dir, batch_size=50):
    data_folder = os.path.split(os.path.abspath(image_dir))[1]
    dataset = load_images_data(image_dir)
    image_paths = list(itertools.chain(*dataset.values()))
    nrof_images = len(image_paths)
    nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / batch_size))

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    graph_def = tf.GraphDef()
    with tf.gfile.GFile(model_path, "rb") as graphFile:
        graph_def.ParseFromString(graphFile.read())
    tf.import_graph_def(graph_def)

    input_images = sess.graph.get_tensor_by_name('import/input:0')
    latent_ze = sess.graph.get_tensor_by_name('import/embeddings:0')

    model_dir = os.path.split(args.model_path)[0]
    json_path = os.path.join(model_dir, 'vector.json')
    for i in tqdm(range(nrof_batches_per_epoch), desc='Batch'):
        image_vector_dict = {}
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, nrof_images)
        paths_batch = image_paths[start_index: end_index]
        images_batch = load_images(paths_batch)
        images_latent_ze = sess.run(latent_ze, {input_images: images_batch})
        for image_path, vector in zip(image_paths, images_latent_ze):
            image_id = os.path.splitext(os.path.split(image_path)[1])[0]
            image_vector_dict[image_id] = vector.tolist()
        append_json(json_path, image_vector_dict)

    sess.close()
