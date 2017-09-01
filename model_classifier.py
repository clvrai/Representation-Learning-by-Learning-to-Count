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

        # build loss and accuracy {{{
        def build_loss(logits, labels):
            # Cross-entropy loss
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)

            # Classification accuracy
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            return tf.reduce_mean(loss), accuracy
        # }}}

        # Counter: takes an image as input and outputs a counting vector
        def Counter(img, reuse=True, scope='Counter'):
            with tf.variable_scope(scope, reuse=reuse) as scope:
                if not reuse: log.warn(scope.name)

                _ = conv2d(img, 64, is_train, info=not reuse, name='conv1_1')
                _ = conv2d(_, 64, is_train, info=not reuse, name='conv1_2')
                conv1 = max_pool(_, name='conv1')

                _ = conv2d(conv1, 128, is_train, info=not reuse, name='conv2_1')
                _ = conv2d(_, 128, is_train, info=not reuse, name='conv2_2')
                conv2 = max_pool(_, name='conv2')

                _ = conv2d(conv2, 256, is_train, info=not reuse, name='conv3_1')
                _ = conv2d(_, 256, is_train, info=not reuse, name='conv3_2')
                _ = conv2d(_, 256, is_train, info=not reuse, name='conv3_3')
                conv3 = max_pool(_, name='conv3')

                _ = conv2d(conv3, 512, is_train, info=not reuse, name='conv4_1')
                _ = conv2d(_, 512, is_train, info=not reuse, name='conv4_2')
                _ = conv2d(_, 512, is_train, info=not reuse, name='conv4_3')
                conv4 = max_pool(_, name='conv4')

                _ = conv2d(conv4, 512, is_train, info=not reuse, name='conv5_1')
                _ = conv2d(_, 512, is_train, info=not reuse, name='conv5_2')
                _ = conv2d(_, 512, is_train, info=not reuse, name='conv5_3')
                conv5 = max_pool(_, name='conv5')

                fc1 = fc(tf.reshape(conv5, [self.batch_size, -1]),
                         4096, is_train, info=not reuse, name='fc_1')
                fc2 = fc(fc1, 4096, is_train, info=not reuse, name='fc_2')
                fc3 = fc(fc2, 1000, is_train, info=not reuse, name='fc_3')
                fc4 = fc(fc3, 1000, is_train, info=not reuse, batch_norm=False, name='fc_4')
                return [conv1, conv2, conv3, conv4, conv5, fc1, fc2, fc3, fc4]

        def Classifier(f, reuse=False, scope='Classifier'):
            return fc(tf.reshape(f, [self.batch_size, -1]), self.num_class, is_train,
                      info=not reuse, batch_norm=False, activation_fn=None, name=scope)

        dh = int(h/2)
        dw = int(w/2)

        D_x = tf.image.resize_images(self.image_x, [dh, dw])

        feature = Counter(D_x, reuse=False)
        log.warn('Classifier')
        output = []
        name = []
        for f in feature:
            name_step = f.name.rsplit('/')[-1].rsplit(':')[0]
            name.append(name_step)
            output.append(Classifier(f, scope=name_step))

        loss = []
        accuracy = []
        for logits in output:
            loss_step, accuray_step = build_loss(logits, self.label_x)
            loss.append(loss_step)
            accuracy.append(accuray_step)

        log.warn('Add summaries')
        for i in range(len(name)):
            log.info(name[i])
            tf.summary.scalar("loss/"+name[i], loss[i])
            tf.summary.scalar("accuracy/"+name[i], accuracy[i])

        self.loss = tf.reduce_mean(tf.stack(loss))
        self.accuracy = tf.reduce_mean(tf.stack(accuracy))

        tf.summary.scalar("loss/loss", self.loss)
        log.warn('Successfully loaded the model.')
