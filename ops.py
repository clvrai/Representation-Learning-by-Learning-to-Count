import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.slim as slim
from util import log


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * tf.where(x > 0.0, x, alpha * tf.exp(x) - alpha)


def huber_loss(labels, predictions, delta=1.0):
    residual = tf.abs(predictions - labels)
    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)
    return tf.where(condition, small_res, large_res)


def bn_act(input, is_train, batch_norm=True, activation_fn=None):
    _ = input
    if activation_fn is not None:
        _ = activation_fn(_)
    if batch_norm is True:
        _ = tf.contrib.layers.batch_norm(
            _, center=True, scale=True, decay=0.9,
            is_training=is_train, updates_collections=None
        )
    return _


def conv2d(input, output_shape, is_train, info=False, k_h=3, k_w=3, s=1,
           stddev=0.1, name="conv2d", activation_fn=tf.nn.relu, batch_norm=True):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input.get_shape()[-1], output_shape],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        _ = tf.nn.conv2d(input, w, strides=[1, s, s, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_shape],
                                 initializer=tf.constant_initializer(0.0))

        _ = tf.reshape(tf.nn.bias_add(_, biases), _.get_shape())
        if info: log.info('{} {}'.format(name, _))
    return bn_act(_, is_train, batch_norm=batch_norm, activation_fn=activation_fn)


def max_pool(input, info=False, k=2, s=2, padding='SAME', name='pool'):
    _ = tf.nn.max_pool(input, ksize=[1, k, k, 1],
                          strides=[1, s, s, 1], padding=padding, name=name)
    if info: log.info('{} {}'.format(name, _))
    return _


def deconv2d(input, deconv_info, is_train, name="deconv2d",
             stddev=0.02, activation_fn=tf.nn.relu, batch_norm=True):
    with tf.variable_scope(name):
        output_shape = deconv_info[0]
        k = deconv_info[1]
        s = deconv_info[2]
        _ = layers.conv2d_transpose(
            input,
            num_outputs=output_shape,
            weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
            biases_initializer=tf.zeros_initializer(),
            kernel_size=[k, k], stride=[s, s], padding='SAME'
        )

    return bn_act(_, is_train, batch_norm=batch_norm, activation_fn=activation_fn)


def bilinear_deconv2d(input, deconv_info, is_train, name="bilinear_deconv2d",
                      stddev=0.02, activation_fn=tf.nn.relu, batch_norm=True):
    with tf.variable_scope(name):
        output_shape = deconv_info[0]
        k = deconv_info[1]
        s = deconv_info[2]
        h = int(input.get_shape()[1]) * s
        w = int(input.get_shape()[2]) * s
        _ = tf.image.resize_bilinear(input, [h, w])
        _ = conv2d(_, output_shape, is_train, k_h=k, s=1, k_w=k,
                   batch_norm=False, activation_fn=None)

    return bn_act(_, is_train, batch_norm=batch_norm, activation_fn=activation_fn)


def residual_conv(input, num_filters, filter_size, stride, reuse=False,
                  pad='SAME', dtype=tf.float32, bias=False):
    stride_shape = [1, stride, stride, 1]
    filter_shape = [filter_size, filter_size, input.get_shape()[3], num_filters]
    w = tf.get_variable('w', filter_shape, dtype, tf.random_normal_initializer(0.0, 0.02))
    p = (filter_size - 1) // 2
    x = tf.pad(input, [[0, 0], [p, p], [p, p], [0, 0]], 'REFLECT')
    conv = tf.nn.conv2d(x, w, stride_shape, padding='VALID')
    return conv


def residual(input, num_filters, name, is_train, reuse=False, pad='REFLECT'):
    with tf.variable_scope(name, reuse=reuse):
        with tf.variable_scope('res1', reuse=reuse):
            out = residual_conv(input, num_filters, 3, 1, reuse, pad)
            out = tf.contrib.layers.batch_norm(
                out, center=True, scale=True, decay=0.9,
                is_training=is_train, updates_collections=None
            )
            out = tf.nn.relu(out)

        with tf.variable_scope('res2', reuse=reuse):
            out = residual_conv(out, num_filters, 3, 1, reuse, pad)
            out = tf.contrib.layers.batch_norm(
                out, center=True, scale=True, decay=0.9,
                is_training=is_train, updates_collections=None
            )

        return tf.nn.relu(input + out)


def fc(input, output_shape, is_train, info=False, batch_norm=True,
       activation_fn=tf.nn.relu, name="fc"):
    _ = slim.fully_connected(input, output_shape, activation_fn=None)
    if info: log.info('{} {}'.format(name, _))
    return bn_act(_, is_train, batch_norm=batch_norm, activation_fn=activation_fn)
