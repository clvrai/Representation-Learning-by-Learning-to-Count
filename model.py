from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from ops import conv2d, fc, max_pool
from util import log


class Model(object):

    def __init__(self, config,
                 debug_information=False,
                 is_train=True):
        self.debug = debug_information

        self.config = config
        self.batch_size = self.config.batch_size
        self.input_height = self.config.data_info[0]
        self.input_width = self.config.data_info[1]
        self.c_dim = self.config.data_info[2]
        self.num_class = self.config.data_info[3]

        # create placeholders for the input
        self.image_x = tf.placeholder(
            name='image_x', dtype=tf.float32,
            shape=[self.batch_size, self.input_height, self.input_width, self.c_dim],
        )
        self.label_x = tf.placeholder(
            name='label_x', dtype=tf.float32, shape=[self.batch_size, self.num_class],
        )
        self.image_y = tf.placeholder(
            name='image_y', dtype=tf.float32,
            shape=[self.batch_size, self.input_height, self.input_width, self.c_dim],
        )
        self.label_y = tf.placeholder(
            name='label_y', dtype=tf.float32, shape=[self.batch_size, self.num_class],
        )

        self.is_training = tf.placeholder_with_default(bool(is_train), [], name='is_training')

        self.build(is_train=is_train)

    def get_feed_dict(self, batch_chunk, step=None, is_training=None):
        fd = {
            self.image_x: batch_chunk['image_x'],  # [B, h, w, c]
            self.label_x: batch_chunk['label_x'],  # [B, n]
            self.image_y: batch_chunk['image_y'],  # [B, h, w, c]
            self.label_y: batch_chunk['label_y'],  # [B, n]
        }
        if is_training is not None:
            fd[self.is_training] = is_training

        return fd

    def build(self, is_train=True):

        h = self.input_height
        w = self.input_width
        M = 10

        # build loss and accuracy {{{
        def build_loss(output):
            [phi_D_x, phi_D_y, phi_T_x_1, phi_T_x_2, phi_T_x_3, phi_T_x_4] = output
            pair = tf.reduce_mean((phi_D_x - (phi_T_x_1 + phi_T_x_2 + phi_T_x_3 + phi_T_x_4)) ** 2)
            unpair_raw = M - (phi_D_y - (phi_T_x_1 + phi_T_x_2 + phi_T_x_3 + phi_T_x_4)) ** 2
            condition = tf.less(unpair_raw, 0.)
            unpair = tf.reduce_mean(tf.where(condition, tf.zeros_like(unpair_raw), unpair_raw))
            loss = pair + unpair
            return loss, pair, unpair
        # }}}

        # Classifier: takes images as input and tries to output class label [B, n]
        def Counter(img, reuse=True, scope='Counter'):
            with tf.variable_scope(scope, reuse=reuse) as scope:
                if not reuse: log.warn(scope.name)

                _ = conv2d(img, 64, is_train, info=not reuse, name='conv1_1')
                _ = conv2d(_, 64, is_train, info=not reuse, name='conv1_2')
                _ = max_pool(_)

                _ = conv2d(_, 128, is_train, info=not reuse, name='conv2_1')
                _ = conv2d(_, 128, is_train, info=not reuse, name='conv2_2')
                _ = max_pool(_)

                _ = conv2d(_, 256, is_train, info=not reuse, name='conv3_1')
                _ = conv2d(_, 256, is_train, info=not reuse, name='conv3_2')
                _ = conv2d(_, 256, is_train, info=not reuse, name='conv3_3')
                _ = max_pool(_)

                _ = conv2d(_, 512, is_train, info=not reuse, name='conv4_1')
                _ = conv2d(_, 512, is_train, info=not reuse, name='conv4_2')
                _ = conv2d(_, 512, is_train, info=not reuse, name='conv4_3')
                _ = max_pool(_)

                _ = conv2d(_, 512, is_train, info=not reuse, name='conv5_1')
                _ = conv2d(_, 512, is_train, info=not reuse, name='conv5_2')
                _ = conv2d(_, 512, is_train, info=not reuse, name='conv5_3')
                _ = max_pool(_)

                _ = fc(tf.reshape(_, [self.batch_size, -1]), 4096,
                       is_train, info=not reuse, name='fc_1')
                log.info('{} {}'.format(scope.name, _))
                _ = fc(_, 4096, is_train, info=not reuse, name='fc_2')
                log.info('{} {}'.format(scope.name, _))
                _ = fc(_, 1000, is_train, info=not reuse, name='fc_3')
                log.info('{} {}'.format(scope.name, _))
                _ = fc(_, 1000, is_train, info=not reuse, batch_norm=False, name='fc_4')
                log.info('{} {}'.format(scope.name, _))
                return _

        dh = int(h/2)
        dw = int(w/2)

        D_x = tf.image.resize_images(self.image_x, [dh, dw])
        D_y = tf.image.resize_images(self.image_y, [dh, dw])
        T_x_1 = self.image_x[:, :dh, :dw, :]
        T_x_2 = self.image_x[:, dh:, :dw, :]
        T_x_3 = self.image_x[:, :dh, dw:, :]
        T_x_4 = self.image_x[:, dh:, dw:, :]

        input = [D_x, D_y, T_x_1, T_x_2, T_x_3, T_x_4]
        dict = ['D_x', 'D_y', 'T_x_1', 'T_x_2', 'T_x_3', 'T_x_4']
        output = []
        for t in range(len(input)):
            output.append(Counter(input[t], reuse=t > 0))
            tf.summary.image("img/{}".format(dict[t]), input[t])

        self.loss, self.loss_pair, self.loss_unpair = build_loss(output)

        tf.summary.scalar("loss/loss", self.loss)
        tf.summary.scalar("loss/pair", self.loss_pair)
        tf.summary.scalar("loss/unpair", self.loss_unpair)
        tf.summary.scalar("count/D_x", tf.reduce_mean(output[0]))
        tf.summary.scalar("count/D_y", tf.reduce_mean(output[1]))
        log.warn('Successfully loaded the model.')