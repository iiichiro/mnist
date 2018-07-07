"""
Tensorflow　MNISTチュートリアルCNNモデル

"""
import tensorflow as tf

from .utils import weight_variable, bias_variable, conv2d, max_pool_2x2


class TensorflowTutorialModel:
    MODEL_NAME = 'TFTutorialModel'
    DEFAULT_CNN_KEEP_PROB = 0.5
    DEFAULT_FC_KEEP_PROB = 0.5

    @classmethod
    def interence(cls, src, cnn_keep_prob, fc_keep_prob):
        with tf.name_scope(cls.MODEL_NAME):
            with tf.name_scope('reshape'):
                x = tf.reshape(src, [-1, 28, 28, 1])

            # 畳み込み層第1層(1*28*28 -> 32*28*28)
            with tf.name_scope('conv1'):
                W = weight_variable((5, 5, 1, 32))
                b = bias_variable((32, ))
                h = tf.nn.relu(conv2d(x, W) + b)
            # プーリング層第1層(32*28*28 -> 32*14*14)
            with tf.name_scope('max_pool1'):
                h = max_pool_2x2(h)
            # 畳み込み層第2層(32*14*14 -> 64*14*14)
            with tf.name_scope('conv2'):
                W = weight_variable((5, 5, 32, 64))
                b = bias_variable((64, ))
                h = tf.nn.relu(conv2d(h, W) + b)
            # プーリング層第2層(64*14*14 -> 64*7*7)
            with tf.name_scope('max_pool2'):
                h = max_pool_2x2(h)
            # flatten(64*7*7 -> 3136)
            with tf.name_scope('flatten'):
                h = tf.reshape(h, (-1, 64*7*7))
            # 全結合層第1層(3136 -> 1024)
            with tf.name_scope('fc1'):
                W = weight_variable((64*7*7, 1024))
                b = bias_variable((1024, ))
                h = tf.nn.relu(tf.matmul(h, W) + b)
            # ドロップアウト層第1層
            with tf.name_scope('dropout1'):
                h = tf.nn.dropout(h, fc_keep_prob)
            # 全結合層第2層(1024 -> 10)
            with tf.name_scope('output'):
                W = weight_variable((1024, 10))
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
    def trainning(loss, learning_rate=1e-4):
        with tf.name_scope('optimaze'):
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        return train_step

    @staticmethod
    def accuracy(logits, labels):
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        return accuracy
