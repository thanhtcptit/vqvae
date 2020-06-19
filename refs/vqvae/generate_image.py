import os
import matplotlib.pyplot as plt
import tensorflow as tf

from models import VQVAE


def convert_batch_to_image_grid(image_batch):
    reshaped = (image_batch.reshape(4, 8, 32, 32, 3)
                .transpose(0, 2, 1, 3, 4)
                .reshape(4 * 32, 8 * 32, 3))
    return reshaped + 0.5


num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2

embedding_dim = 64
num_embeddings = 512

commitment_cost = 0.25

vq_use_ema = False
decay = 0.99

model = VQVAE(num_hiddens, num_residual_hiddens,
              num_residual_hiddens, embedding_dim, num_embeddings,
              commitment_cost, decay, use_ema=False)
sample = model.generate_image(num_images=32)

checkpoint_path = os.path.join('train_log/draft/model-{}.ckpt')
saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, 'train_log/draft/model-50000.ckpt')

generate_sample = sess.run(sample)

plt.imshow(convert_batch_to_image_grid(generate_sample))
plt.savefig('train_log/draft/sample.png')
