"""
Models
- Customized LaNet5
- SimpleCNN
- ComplexCNN
-

"""

import tensorflow as tf

from .utils import weight_variable, bias_variable, conv2d, POOLS


class LaNet5:
    MODEL_NAME = 'LaNet5'
    DEFAULT_CNN_KEEP_PROB = 1.0
    DEFAULT_FC_KEEP_PROB = 1.0
    _POOLS = POOLS

    @classmethod
    def interence(cls, src, cnn_keep_prob, fc_keep_prob, pool_mode='MAX'):
        if pool_mode not in cls._POOLS:
            raise Exception('`pool_mode` must be either {0} or {1}.'.format(*cls._POOLS.keys()))
        else:
            pool_layer = cls._POOLS[pool_mode]

        with tf.name_scope(cls.MODEL_NAME):
            with tf.name_scope('reshape'):
                x = tf.reshape(src, [-1, 28, 28, 1])

            # 畳み込み層第1層(1*28*28 -> 6*28*28)
            with tf.name_scope('conv1'):
                W = weight_variable((3, 3, 1, 6))
                h = conv2d(x, W)
            # プーリング層第1層(6*28*28 -> 6*14*14)
            with tf.name_scope('max_pool1'):
                h = pool_layer(h)
            # 畳み込み層第2層(6*14*14 -> 16*14*14)
            with tf.name_scope('conv2'):
                W = weight_variable((5, 5, 6, 16))
                h = conv2d(h, W)
            # プーリング層第2層(16*14*14 -> 16*7*7)
            with tf.name_scope('max_pool2'):
                h = pool_layer(h)
            # 畳み込み層第3層(16*7*7 -> 120*7*7)
            with tf.name_scope('conv3'):
                W = weight_variable((5, 5, 16, 120))
                h = conv2d(h, W)
            # プーリング層第3層(120*7*7 -> 120*4*4)
            with tf.name_scope('max_pool3'):
                h = pool_layer(h)
            # flatten(120*4*4 -> 1920)
            with tf.name_scope('flatten'):
                h = tf.reshape(h, (-1, 120*4*4))
            # 全結合層第1層(1920 -> 120)
            with tf.name_scope('fc1'):
                W = weight_variable((120*4*4, 120))
                b = bias_variable((120, ))
                h = tf.nn.relu(tf.matmul(h, W) + b)
            # 全結合層第2層(120 -> 120)
            with tf.name_scope('fc2'):
                W = weight_variable((120, 120))
                b = bias_variable((120, ))
                h = tf.nn.relu(tf.matmul(h, W) + b)
            # 全結合層第3層(120 -> 10)
            with tf.name_scope('output'):
                W = weight_variable((120, 10))
                b = bias_variable((10, ))
                y = tf.nn.softmax(tf.matmul(h, W) + b)
            return y

    @staticmethod
    def loss(logits, labels):
        with tf.name_scope('loss'):
            # 交差エントロピーの計算
            cross_entropy = -tf.reduce_sum(labels*tf.log(tf.clip_by_value(logits, 1e-10, 1.0)))
        return cross_entropy

    @staticmethod
    def trainning(loss, learning_rate=5e-4):
        with tf.name_scope('optimaze'):
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        return train_step

    @staticmethod
    def accuracy(logits, labels):
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        return accuracy


class SimpleCNN:
    MODEL_NAME = 'SimpleCNN'
    DEFAULT_CNN_KEEP_PROB = 0.25
    DEFAULT_FC_KEEP_PROB = 0.25
    _POOLS = POOLS

    @classmethod
    def interence(cls, src, cnn_keep_prob, fc_keep_prob, pool_mode='AVG'):
        if pool_mode not in cls._POOLS:
            raise Exception('`pool_mode` must be either {0} or {1}.'.format(*cls._POOLS.keys()))
        else:
            pool_layer = cls._POOLS[pool_mode]

        with tf.name_scope(cls.MODEL_NAME):
            with tf.name_scope('reshape'):
                x = tf.reshape(src, [-1, 28, 28, 1])

            # 畳み込み層第1層(1*28*28 -> 32*28*28)
            with tf.name_scope('conv1'):
                W = weight_variable((3, 3, 1, 32))
                h = conv2d(x, W)
            # プーリング層第1層(32*28*28 -> 32*14*14)
            with tf.name_scope('avg_pool1'):
                h = pool_layer(h)
            # 畳み込み層第2層(32*14*14 -> 32*14*14)
            with tf.name_scope('conv2'):
                W = weight_variable((3, 3, 32, 32))
                h = conv2d(h, W)
            # プーリング層第2層(32*14*14 -> 32*7*7)
            with tf.name_scope('avg_pool2'):
                h = pool_layer(h)
            # 畳み込み層第3層(32*7*7 -> 64*7*7)
            with tf.name_scope('conv3'):
                W = weight_variable((3, 3, 32, 64))
                h = conv2d(h, W)
            # プーリング層第3層(64*7*7 -> 64*4*4)
            with tf.name_scope('avg_pool3'):
                h = pool_layer(h)
            # ドロップアウト層第1層
            with tf.name_scope('dropout1'):
                h = tf.nn.dropout(h, cnn_keep_prob)
            # flatten(64*4*4 -> 1024)
            with tf.name_scope('flatten'):
                h = tf.reshape(h, (-1, 64*4*4))
            # 全結合層第1層(1024 -> 256)
            with tf.name_scope('fc1'):
                W = weight_variable((64*4*4, 256))
                b = bias_variable((256, ))
                h = tf.nn.relu(tf.matmul(h, W) + b)
            # ドロップアウト層第2層
            with tf.name_scope('dropout2'):
                h = tf.nn.dropout(h, fc_keep_prob)
            # 全結合層第2層(256 -> 256)
            with tf.name_scope('fc2'):
                W = weight_variable((256, 256))
                b = bias_variable((256, ))
                h = tf.nn.relu(tf.matmul(h, W) + b)
            # 全結合層第3層(256 -> 10)
            with tf.name_scope('output'):
                W = weight_variable((256, 10))
                b = bias_variable((10, ))
                y = tf.nn.softmax(tf.matmul(h, W) + b)
            return y

    @staticmethod
    def loss(logits, labels):
        with tf.name_scope('loss'):
            # 交差エントロピーの計算
            cross_entropy = -tf.reduce_sum(labels*tf.log(tf.clip_by_value(logits, 1e-10, 1.0)))
        return cross_entropy

    @staticmethod
    def trainning(loss, learning_rate=5e-4):
        with tf.name_scope('optimaze'):
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        return train_step

    @staticmethod
    def accuracy(logits, labels):
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        return accuracy


class ComplexCNN:
    MODEL_NAME = 'ComplexCNN'
    DEFAULT_CNN_KEEP_PROB = 0.25
    DEFAULT_FC_KEEP_PROB = 0.5
    _POOLS = POOLS

    @classmethod
    def interence(cls, src, cnn_keep_prob, fc_keep_prob, pool_mode='MAX'):
        if pool_mode not in cls._POOLS:
            raise Exception('`pool_mode` must be either {0} or {1}.'.format(*cls._POOLS.keys()))
        else:
            pool_layer = cls._POOLS[pool_mode]

        with tf.name_scope(cls.MODEL_NAME):
            with tf.name_scope('reshape'):
                x = tf.reshape(src, [-1, 28, 28, 1])

            # 畳み込み層第1層(1*28*28 -> 32*28*28)
            with tf.name_scope('conv1'):
                W = weight_variable((3, 3, 1, 32))
                b = bias_variable((32, ))
                h = tf.nn.relu(conv2d(x, W) + b)
            # 畳み込み層第2層(32*28*28 -> 32*28*28)
            with tf.name_scope('conv2'):
                W = weight_variable((3, 3, 32, 32))
                b = bias_variable((32, ))
                h = tf.nn.relu(conv2d(h, W) + b)
            # プーリング層第1層(32*28*28 -> 32*14*14)
            with tf.name_scope('max_pool1'):
                h = pool_layer(h)
            # 畳み込み層第3層(32*14*14 -> 32*14*14)
            with tf.name_scope('conv3'):
                W = weight_variable((3, 3, 32, 32))
                b = bias_variable((32, ))
                h = tf.nn.relu(conv2d(h, W) + b)
            # 畳み込み層第4層(32*14*14 -> 32*14*14)
            with tf.name_scope('conv4'):
                W = weight_variable((3, 3, 32, 32))
                b = bias_variable((32, ))
                h = tf.nn.relu(conv2d(h, W) + b)
            # プーリング層第2層(32*14*14 -> 32*7*7)
            with tf.name_scope('max_pool2'):
                h = pool_layer(h)
            # 畳み込み層第5層(32*7*7 -> 64*7*7)
            with tf.name_scope('conv5'):
                W = weight_variable((3, 3, 32, 64))
                b = bias_variable((64, ))
                h = tf.nn.relu(conv2d(h, W) + b)
            # 畳み込み層第6層(32*7*7 -> 64*7*7)
            with tf.name_scope('conv6'):
                W = weight_variable((3, 3, 64, 64))
                b = bias_variable((64, ))
                h = tf.nn.relu(conv2d(h, W) + b)
            # プーリング層第3層(64*7*7 -> 64*4*4)
            with tf.name_scope('max_pool3'):
                h = pool_layer(h)
            # ドロップアウト層第1層
            with tf.name_scope('dropout1'):
                h = tf.nn.dropout(h, cnn_keep_prob)
            # flatten(64*4*4 -> 1024)
            with tf.name_scope('flatten'):
                h = tf.reshape(h, (-1, 64*4*4))
            # 全結合層第1層(1024 -> 256)
            with tf.name_scope('fc1'):
                W = weight_variable((64*4*4, 256))
                b = bias_variable((256, ))
                h = tf.nn.relu(tf.matmul(h, W) + b)
            # ドロップアウト層第2層
            with tf.name_scope('dropout2'):
                h = tf.nn.dropout(h, fc_keep_prob)
            # 全結合層第2層(256 -> 256)
            with tf.name_scope('fc2'):
                W = weight_variable((256, 256))
                b = bias_variable((256, ))
                h = tf.nn.relu(tf.matmul(h, W) + b)
            # ドロップアウト層第3層
            with tf.name_scope('dropout3'):
                h = tf.nn.dropout(h, fc_keep_prob)
            # 全結合層第3層(256 -> 10)
            with tf.name_scope('output'):
                W = weight_variable((256, 10))
                b = bias_variable((10, ))
                y = tf.nn.softmax(tf.matmul(h, W) + b)
            return y

    @staticmethod
    def loss(logits, labels):
        with tf.name_scope('loss'):
            # 交差エントロピーの計算
            cross_entropy = -tf.reduce_sum(labels*tf.log(tf.clip_by_value(logits, 1e-10, 1.0)))
        return cross_entropy

    @staticmethod
    def trainning(loss, learning_rate=5e-4):
        with tf.name_scope('optimaze'):
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        return train_step

    @staticmethod
    def accuracy(logits, labels):
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        return accuracy
