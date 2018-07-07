import tensorflow as tf


# 重みを標準偏差0.1の正規分布で初期化
def weight_variable(shape):
    inital = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(inital, name='weight')


# バイアスを標準偏差0.1の正規分布で初期化
def bias_variable(shape):
    inital = tf.constant(0.1, shape=shape)
    return tf.Variable(inital, name='bias')


# 畳み込み層
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


# プーリング層
def max_pool_2x2(x, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=padding)


# プーリング層
def avg_pool_2x2(x, padding='SAME'):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=padding)


POOLS = {'MAX': max_pool_2x2, 'AVG': avg_pool_2x2}
