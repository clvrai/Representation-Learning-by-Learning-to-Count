import numpy as np
import tensorflow as tf

from util import log


def check_data_id(dataset, data_id):
    if not data_id:
        return

    wrong = []
    for id in data_id:
        if id in dataset.data:
            pass
        else:
            wrong.append(id)

    if len(wrong) > 0:
        raise RuntimeError("There are %d invalid ids, including %s" % (
            len(wrong), wrong[:5]
        ))


def create_input_ops(dataset,
                     batch_size,
                     num_threads=16,           # for creating batches
                     is_training=False,
                     data_id=None,
                     scope='inputs',
                     shuffle=True,
                     ):
    '''
    Return a batched tensor for the inputs from the dataset.
    '''
    input_ops = {}

    if data_id is None:
        data_id = dataset.ids
        log.info("input_ops [%s]: Using %d IDs from dataset", scope, len(data_id))
    else:
        log.info("input_ops [%s]: Using specified %d IDs", scope, len(data_id))

    # single operations
    with tf.device("/cpu:0"), tf.name_scope(scope):
        match = True
        while match:
            input_ops['id_x'] = tf.train.string_input_producer(
                tf.convert_to_tensor(data_id),
                capacity=128
            ).dequeue(name='input_ids_dequeue')
            input_ops['id_y'] = tf.train.string_input_producer(
                tf.convert_to_tensor(data_id),
                capacity=128
            ).dequeue(name='input_ids_dequeue')
            match = input_ops['id_x'] == input_ops['id_y']

        img_x, img_y, l_x, l_y = dataset.get_data(data_id[0], data_id[1])

        def load_fn(id_x, id_y):
            # img [h, w, c], l: [n]
            img_x, img_y, l_x, l_y = dataset.get_data(id_x, id_y)
            return (id_x, id_y,
                    img_x.astype(np.float32), l_x.astype(np.float32),
                    img_y.astype(np.float32), l_y.astype(np.float32))

        input_ops['id_x'], input_ops['id_y'], input_ops['image_x'], \
            input_ops['label_x'], input_ops['image_y'], input_ops['label_y'] = tf.py_func(
            load_fn, inp=[input_ops['id_x'], input_ops['id_y']],
            Tout=[tf.string, tf.string, tf.float32,
                  tf.float32, tf.float32, tf.float32], name='func'
        )

        input_ops['id_x'].set_shape([])
        input_ops['id_y'].set_shape([])
        input_ops['image_x'].set_shape(list(img_x.shape))
        input_ops['label_x'].set_shape(list(l_x.shape))
        input_ops['image_y'].set_shape(list(img_y.shape))
        input_ops['label_y'].set_shape(list(l_y.shape))

    # batchify
    capacity = 2 * batch_size * num_threads
    min_capacity = min(int(capacity * 0.75), 1024)

    if shuffle:
        batch_ops = tf.train.shuffle_batch(
            input_ops,
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=capacity,
            min_after_dequeue=min_capacity,
        )
    else:
        batch_ops = tf.train.batch(
            input_ops,
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=capacity,
        )

    return input_ops, batch_ops
