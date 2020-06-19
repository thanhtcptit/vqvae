import os
import zipfile
import tensorflow as tf

from numpy import loadtxt, round
from tensorpack.utils import logger
from tensorpack.callbacks import Callback

from src.utils.tools import zip_file


class CompressResource(Callback):
    def __init__(self, resource_path, output_path):
        super(CompressResource, self).__init__()
        self._resource_path = resource_path
        self._output_path = output_path

    def _after_train(self):
        zip_file(self._resource_path, self._output_path)


class RestoreWeights(Callback):
    def __init__(self, checkpoint_path):
        super(RestoreWeights, self).__init__()
        self._checkpoint_path = checkpoint_path
        self._saver = None

    def _setup_graph(self):
        train_vars = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
        vqvae_model_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                            scope='vqvae')
        for var in vqvae_model_var:
            train_vars.remove(var)
        self._saver = tf.train.Saver(vqvae_model_var)

    def _before_train(self):
        self._saver.restore(self.trainer.sess, self._checkpoint_path)


class Notification(Callback):
    def __init__(self, title, body, after_every=0, after_train=True):
        if not os.path.exists('config/pushbullet.id'):
            self._activate = False
        else:
            self._activate = True
            self._user_id = loadtxt('config/pushbullet.id', dtype='str')
            self._trigger_after_every = after_every
            self._trigger_after_train = after_train
            self._command = f'''curl -u {self._user_id}: \\
                https://api.pushbullet.com/v2/pushes \\
                -d type=note -d title="{title}" \\
                -d body="{body}" > /dev/null 2>&1'''

    def run(self):
        self._run_command()

    def _run_command(self):
        ret = os.system(self._command)
        if ret != 0:
            logger.error("Command {} failed with ret={}!".format(
                self._command, ret))

    def _trigger(self):
        if self._activate and self._trigger_after_every:
            if self.epoch_num % self._trigger_after_every == 0:
                self._run_command()

    def _after_train(self):
        if self._activate and self._trigger_after_train:
            self._run_command()


class SendStat(Notification):
    def __init__(self, title, stats, after_every=0, after_train=True):
        super(SendStat, self).__init__(title, '', after_every, after_train)
        if not isinstance(stats, list):
            stats = [stats]
        self._stats = stats

        self._command = f'''
                curl -u {self._user_id}: \\
                https://api.pushbullet.com/v2/pushes \\
                -d type=note -d title="{title}" '''
        body = '-d body="{}" > /dev/null 2>&1'
        stats_string = ['Epoch: {{epoch}}']
        for stat in self._stats:
            stats_string.append(stat + ': {' + '{}'.format(stat) + '}')
        body = body.format(' - '.join(stats_string))
        self._command += body

    def _run_command(self):
        m = self.trainer.monitors
        v = {k: round(m.get_latest(k), 4) for k in self._stats}
        v['epoch'] = self.epoch_num
        cmd = self._command.format(**v)
        ret = os.system(cmd)
        if ret != 0:
            logger.error("Command {} failed with ret={}!".format(
                cmd, ret))
