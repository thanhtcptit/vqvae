import tensorflow as tf


class CVAE(tf.keras.Model):
    def __init__(self, input_shape, latent_dim):
        super(CVAE, self).__init__()
        self._input_shape = input_shape
        self._latent_dim = latent_dim

    def build_graph(self):
        self.inference_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=self._input_shape),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Flatten(),
            # No activation
            tf.keras.layers.Dense(2 * self._latent_dim),
        ])

        pre_convT_shape = (int(self._input_shape[0] / 4),
                           int(self._input_shape[1] / 4), 32)
        pre_convT_unit = pre_convT_shape[0] * \
            pre_convT_shape[1] * pre_convT_shape[2]

        self.generative_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self._latent_dim,)),
            tf.keras.layers.Dense(units=pre_convT_unit, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=pre_convT_shape),
            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=3, strides=(2, 2), padding="SAME",
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=32, kernel_size=3, strides=(2, 2), padding="SAME",
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=self._input_shape[2], kernel_size=3, strides=(1, 1),
                padding="SAME"),
        ])

    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.inference_net(x),
                                num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits
