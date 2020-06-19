import sonnet as snt
import tensorflow as tf


def residual_stack(h, num_hiddens, num_residual_layers, num_residual_hiddens):
    for i in range(num_residual_layers):
        h_i = tf.nn.relu(h)

        h_i = snt.Conv2D(
            output_channels=num_residual_hiddens,
            kernel_shape=(3, 3),
            stride=(1, 1),
            name="res3x3_%d" % i)(h_i)
        h_i = tf.nn.relu(h_i)

        h_i = snt.Conv2D(
            output_channels=num_hiddens,
            kernel_shape=(1, 1),
            stride=(1, 1),
            name="res1x1_%d" % i)(h_i)
        h += h_i
    return tf.nn.relu(h)


class Encoder(snt.AbstractModule):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
                 name='encoder'):
        super(Encoder, self).__init__(name=name)
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

    def _build(self, x):
        h = snt.Conv2D(
            output_channels=self._num_hiddens / 2,
            kernel_shape=(4, 4),
            stride=(2, 2),
            name="enc_1")(x)
        h = tf.nn.relu(h)

        h = snt.Conv2D(
            output_channels=self._num_hiddens,
            kernel_shape=(4, 4),
            stride=(2, 2),
            name="enc_2")(h)
        h = tf.nn.relu(h)

        h = snt.Conv2D(
            output_channels=self._num_hiddens,
            kernel_shape=(3, 3),
            stride=(1, 1),
            name="enc_3")(h)

        h = residual_stack(
            h,
            self._num_hiddens,
            self._num_residual_layers,
            self._num_residual_hiddens)

        return h


class Decoder(snt.AbstractModule):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
                 name='decoder'):
        super(Decoder, self).__init__(name=name)
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

    def _build(self, x):
        h = snt.Conv2D(
            output_channels=self._num_hiddens,
            kernel_shape=(3, 3),
            stride=(1, 1),
            name="dec_1")(x)

        h = residual_stack(
            h,
            self._num_hiddens,
            self._num_residual_layers,
            self._num_residual_hiddens)

        h = snt.Conv2DTranspose(
            output_channels=int(self._num_hiddens / 2),
            output_shape=None,
            kernel_shape=(4, 4),
            stride=(2, 2),
            name="dec_2")(h)
        h = tf.nn.relu(h)

        x_recon = snt.Conv2DTranspose(
            output_channels=3,
            output_shape=None,
            kernel_shape=(4, 4),
            stride=(2, 2),
            name="dec_3")(h)

        return x_recon


class VQVAE(snt.AbstractModule):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
                 embedding_dim, num_embeddings, commitment_cost, decay=0.99,
                 use_ema=False, name='vqvae'):
        super(VQVAE, self).__init__(name=name)
        self._encoder = Encoder(num_hiddens, num_residual_layers,
                                num_residual_hiddens)
        self._decoder = Decoder(num_hiddens, num_residual_layers,
                                num_residual_hiddens)

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        if use_ema:
            self._vq_vae = snt.nets.VectorQuantizerEMA(
                embedding_dim=embedding_dim,
                num_embeddings=num_embeddings,
                commitment_cost=commitment_cost,
                decay=decay)
        else:
            self._vq_vae = snt.nets.VectorQuantizer(
                embedding_dim=embedding_dim,
                num_embeddings=num_embeddings,
                commitment_cost=commitment_cost)

        self._pre_vq = snt.Conv2D(
            output_channels=self._embedding_dim, kernel_shape=(1, 1),
            stride=(1, 1), name="to_vq")

    def _build(self, x, is_training=True):
        z = self._pre_vq(self._encoder(x))  # 8 x 8 x 64
        vq_output = self._vq_vae(z, is_training)
        x_recon = self._decoder(vq_output["quantize"])

        return x_recon, vq_output['loss'], vq_output["perplexity"]

    def generate_image(self, num_images=8):
        # sample_ze = tf.random.normal(
        #     shape=(num_images, 8, 8, self._embedding_dim))
        # vq_output = self._vq_vae(sample_ze, False)
        sample_p_z_x = tf.random.uniform(
            shape=[num_images * 8 * 8], maxval=self._num_embeddings,
            dtype=tf.dtypes.int32)
        # sample_p_z_x = tf.one_hot(sample_p_z_x, self._num_embeddings)
        sample_p_z_x = tf.reshape(sample_p_z_x, [num_images, 8, 8])
        sample_p_z_x = tf.cast(sample_p_z_x, dtype=tf.dtypes.int32)
        quntized = self._vq_vae.quantize(sample_p_z_x)

        return self._decoder(quntized)
