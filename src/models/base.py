import numpy as np
import sonnet as snt
import tensorflow as tf

from nsds.common import Params
from nsds.common.registrable import Registrable
from tensorflow.python.framework.tensor_spec import TensorSpec
from tensorpack import ModelDesc, get_current_tower_context
from tensorpack.tfutils.summary import add_moving_summary


class BaseVAE(ModelDesc, Registrable):
    def __init__(self, image_shape, latent_dim, loss_func, learning_rate):
        self._image_shape = image_shape
        self._latent_dim = latent_dim
        self._loss_func = loss_func
        self._learning_rate = learning_rate

    def inputs(self):
        return [TensorSpec([None] + list(self._image_shape),
                           tf.float32, name='input'),
                TensorSpec([None], tf.int32, name='class')]

    def encode(self, *args, **kwargs):
        raise NotImplementedError

    def decode(self, *args, **kwargs):
        raise NotImplementedError

    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def reconstruct(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z)
        return z, x_logit

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(tf.shape(mean))
        return eps * tf.exp(logvar * .5) + mean

    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)

    def compute_loss(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z)

        if self._loss_func == 'sigmoid_ce':
            logpx_z = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=x_logit, labels=x)
        elif self._loss_func == 'mse':
            logpx_z = (x_logit - x) ** 2
        else:
            raise ValueError('Undefined loss function')
        logpx_z = -tf.reduce_sum(logpx_z, axis=[1, 2, 3])
        logpz = self.log_normal_pdf(z, 0., 0.)
        logqz_x = self.log_normal_pdf(z, mean, logvar)
        return z, x_logit, logpx_z, -tf.reduce_mean(logpx_z + logpz - logqz_x)

    def build_graph(self, x, _):
        z, x_recon, logpx_z, cost = self.compute_loss(x)
        z = tf.identity(z, name='embeddings')
        x_recon = tf.identity(x_recon, name='x_recon')
        avg_logpx_z = tf.reduce_sum(logpx_z) / tf.cast(
            tf.shape(logpx_z)[0], dtype=tf.float32)
        cost = tf.identity(cost, name='neg_elbo')
        avg_logpx_z = tf.identity(avg_logpx_z, name='avg_logpx_z')
        add_moving_summary(avg_logpx_z, cost)

        return cost

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=self._learning_rate,
                             trainable=False)
        return tf.train.AdamOptimizer(learning_rate=lr, epsilon=0.1)

    def name(self):
        return 'vae'


class BaseVQVAE(ModelDesc, Registrable):
    def __init__(self, image_shape, num_hiddens, embedding_dim,
                 num_embeddings, commitment_cost, learning_rate, decay=0.99,
                 use_ema=False):
        self._image_shape = image_shape
        self._num_hiddens = num_hiddens
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        self._decay = decay
        self._use_ema = use_ema
        self._learning_rate = learning_rate

        self._vq = None

    def inputs(self):
        return [TensorSpec([None] + list(self._image_shape),
                           tf.float32, name='input'),
                TensorSpec([None], tf.int32, name='class')]

    def encode(self, *args, **kwargs):
        raise NotImplementedError

    def decode(self, *args, **kwargs):
        raise NotImplementedError

    def get_vq(self):
        with tf.variable_scope('vqvae', reuse=tf.AUTO_REUSE):
            if self._use_ema:
                vq = snt.nets.VectorQuantizerEMA(
                    embedding_dim=self._embedding_dim,
                    num_embeddings=self._num_embeddings,
                    commitment_cost=self._commitment_cost,
                    decay=self._decay)
            else:
                vq = snt.nets.VectorQuantizer(
                    embedding_dim=self._embedding_dim,
                    num_embeddings=self._num_embeddings,
                    commitment_cost=self._commitment_cost)
        return vq

    def lookup(self, z_indicies):
        if self._vq is None:
            self._vq = self.get_vq()
        return self._vq.quantize(z_indicies)

    def quantize(self, ze, is_training=False):
        if self._vq is None:
            self._vq = self.get_vq()
        return self._vq(ze, is_training)

    def reconstruct(self, x, is_training=False):
        ze = self.encode(x)
        zq = self.quantize(ze, is_training)
        x_logit = self.decode(zq['quantize'])
        return ze, zq, x_logit

    def build_graph(self, x, _):
        is_training = get_current_tower_context().is_training
        ze, zq, x_recon = self.reconstruct(x, is_training)

        tf.identity(tf.layers.Flatten()(ze), name='embeddings')
        tf.identity(tf.layers.Flatten()(zq['quantize']), name='latent_zq')
        tf.identity(zq['encoding_indices'], name='pz_x')
        perplexity = tf.identity(zq['perplexity'], name='perplexity')

        x_recon = tf.identity(x_recon, name='x_recon')
        recon_loss = tf.reduce_mean((x_recon - x) ** 2)
        loss = recon_loss + zq['loss']
        loss = tf.identity(loss, name='loss')

        add_moving_summary(loss, perplexity)
        return loss

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=self._learning_rate,
                             trainable=False)
        return tf.train.AdamOptimizer(learning_rate=lr, epsilon=0.1)

    def name(self):
        return 'vqvae'


class BasePixelCNNPrior(ModelDesc, Registrable):
    def __init__(self, image_shape, latent_shape, num_layers, num_embeddings,
                 num_hiddens, num_labels, learning_rate, vqvae_model_params):
        self._image_shape = image_shape
        self._latent_shape = latent_shape
        self._num_layers = num_layers
        self._num_embeddings = num_embeddings
        self._num_hiddens = num_hiddens
        self._num_labels = num_labels
        self._learning_rate = learning_rate
        self._vqvae_model_params = Params.from_file(vqvae_model_params)
        self._vqvae_model = BaseVQVAE.from_params(
            self._vqvae_model_params['model'])

    def inputs(self):
        return [TensorSpec([None] + list(self._image_shape), tf.float32,
                           name='input'),
                TensorSpec([None], tf.int32, name='class')]

    def generate(self, *args, **kwargs):
        raise NotImplementedError

    def sample_from_dist(self, z):
        z_logits = tf.nn.softmax(z)
        z_dist = tf.distributions.Categorical(probs=z_logits)
        z_sample = z_dist.sample()
        return z_sample

    def sample(self, z_pre, y):
        z = self.generate(z_pre, y)
        return self.sample_from_dist(z)

    def decode(self, z):
        z_embs = self._vqvae_model.lookup(z)
        return self._vqvae_model.decode(z_embs)

    def reconstruct(self, x, y):
        z = self._vqvae_model.encode(x)
        z = self._vqvae_model.quantize(z)['encoding_indices']
        z_hat = self.generate(z, y)
        return z, z_hat

    def build_graph(self, x, y):
        z, z_hat = self.reconstruct(x, y)

        z_hat = tf.reshape(z_hat, shape=[-1, self._num_embeddings])
        z_org = tf.reshape(z, shape=[-1])

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=z_hat, labels=z_org)
        loss = tf.reduce_mean(cross_entropy, name='loss')
        add_moving_summary(loss)
        return loss

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=self._learning_rate,
                             trainable=False)
        return tf.train.AdamOptimizer(learning_rate=lr, epsilon=0.1)

    def name(self):
        return 'pixelcnn'


class BaseImageEmbedding(ModelDesc, Registrable):
    def __init__(self, image_shape, num_labels, latent_shape, embeddings_dim,
                 num_hiddens, loss_stragegy, margin, learning_rate, drop_out,
                 vqvae_model_params):
        self._image_shape = image_shape
        self._num_labels = num_labels
        self._latent_shape = latent_shape
        self._embeddings_dim = embeddings_dim
        self._num_hiddens = num_hiddens

        assert loss_stragegy in ['softmax', 'triplet-all', 'triplet-hard']
        self._loss_stragegy = loss_stragegy
        self._margin = margin
        self._learning_rate = learning_rate
        self._drop_out = drop_out
        self._vqvae_model_params = Params.from_file(vqvae_model_params)
        self._vqvae_model = BaseVQVAE.from_params(
            self._vqvae_model_params['model'])

    def inputs(self):
        return [TensorSpec([None] + list(self._image_shape), tf.float32,
                           name='input'),
                TensorSpec([None], tf.int32, name='class')]

    def embed(self, *args, **kwargs):
        raise NotImplementedError

    def pairwise_distance(self, embeddings):
        emb_products = tf.matmul(embeddings, tf.transpose(embeddings))
        diagonal = tf.diag_part(emb_products)
        distance = tf.expand_dims(diagonal, 0) - 2 * emb_products + \
            tf.expand_dims(diagonal, 1)
        distance = tf.maximum(distance, 0.)
        return distance

    def mask_triplet(self, triplet, labels):
        indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
        indices_not_equal = tf.logical_not(indices_equal)
        i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
        i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
        j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

        distinct_indices = tf.logical_and(tf.logical_and(
            i_not_equal_j, i_not_equal_k), j_not_equal_k)

        label_equal = tf.equal(tf.expand_dims(labels, 0),
                               tf.expand_dims(labels, 1))
        i_equal_j = tf.expand_dims(label_equal, 2)
        i_equal_k = tf.expand_dims(label_equal, 1)

        valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

        mask = tf.logical_and(distinct_indices, valid_labels)
        mask = tf.to_float(mask)

        return triplet * mask, tf.reduce_sum(mask)

    def get_valid_mask(self, labels, positive_mask=True):
        label_equal = tf.equal(tf.expand_dims(labels, 0),
                               tf.expand_dims(labels, 1))
        if positive_mask:
            indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
            indices_not_equal = tf.logical_not(indices_equal)
            mask = tf.logical_and(indices_not_equal, label_equal)
        else:
            mask = tf.logical_not(label_equal)
        mask = tf.to_float(mask)
        return mask

    def build_graph(self, x, y):
        is_training = get_current_tower_context().is_training
        z = self._vqvae_model.encode(x)
        z = self._vqvae_model.quantize(z)['quantize']

        embeddings = self.embed(z, is_training)
        embeddings = tf.nn.l2_normalize(embeddings, 1, 1e-10,
                                        name='embeddings')

        if self._loss_stragegy == 'triplet-all':
            distance = self.pairwise_distance(embeddings)
            triplet_distance = tf.expand_dims(distance, 2) - \
                tf.expand_dims(distance, 1) + self._margin
            triplet_distance, num_valid_triplet = \
                self.mask_triplet(triplet_distance, y)
            triplet_distance = tf.maximum(triplet_distance, 0.)
            num_pos_triplet = tf.reduce_sum(tf.to_float(
                tf.greater(triplet_distance, 1e-16)))
            loss = tf.reduce_sum(triplet_distance) / (num_pos_triplet + 1e-16)
            pos_triplet_frac = num_pos_triplet / (num_valid_triplet + 1e-16)
            add_moving_summary(tf.identity(loss, 'loss'))
            add_moving_summary(tf.identity(
                pos_triplet_frac, 'pos_triplet_frac'))
        elif self._loss_stragegy == 'triplet-hard':
            distance = self.pairwise_distance(embeddings)
            valid_pos_mask = self.get_valid_mask(y)
            valid_pos_anchor = distance * valid_pos_mask
            hardest_pos_anchor = tf.reduce_max(distance,
                                               axis=1, keepdims=True)

            valid_neg_mask = self.get_valid_mask(y, positive_mask=False)
            max_dist = tf.reduce_max(distance, axis=1, keepdims=True)
            dummy_distance = distance + max_dist * (1. - valid_neg_mask)
            hardest_neg_anchor = tf.reduce_min(dummy_distance,
                                               axis=1, keepdims=True)
            triplet_loss = tf.maximum(
                hardest_pos_anchor - hardest_neg_anchor + self._margin, 0.)
            loss = tf.reduce_mean(triplet_loss)
            add_moving_summary(tf.identity(loss, 'loss'))
        else:
            logits = tf.layers.dense(embeddings, self._num_labels)
            predictions = tf.argmax(logits, axis=1)
            correct_prediction = tf.to_float(
                tf.equal(predictions, tf.cast(y, tf.int64)),
                name='correct_prediction')
            accuracy = tf.reduce_mean(correct_prediction, name='accuracy')
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=y)
            loss = tf.reduce_mean(cross_entropy, name='loss')

            add_moving_summary(loss)
            add_moving_summary(accuracy)

        return loss

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=self._learning_rate,
                             trainable=False)
        return tf.train.AdamOptimizer(learning_rate=lr, epsilon=0.1)

    def name(self):
        return 'image_embedding'
