import argparse
import datetime
import logging
import logging.handlers
import os
import pickle

import numpy as np
import tensorflow as tf

from dataset.mnist_dataset import (MnistDataset, MnistDatasetManager,
                                   load_mnist_data)
from models import DEFINED_MODELS


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Train MNIST.')
    arg_parser.add_argument('-e', '--epochs', required=True, type=int,
                            help='Learning frequency.')
    arg_parser.add_argument('-b', '--batchsize', default=128, type=int,
                            help='Size used for mini batch.')
    arg_parser.add_argument('-m', '--model', required=True, choices=DEFINED_MODELS.keys(),
                            help='A model architecture to use.')
    arg_parser.add_argument('-o', '--output', default=None, type=str,
                            help='Output location of result.')
    arg_parser.add_argument('-d', '--dataset', required=True, type=str,
                            help='Pickle file location of MnistDatasetManger.')
    arg_parser.add_argument('--keep_num', default=None, type=int,
                            help='Number of models to keep.')
    arg_parser.add_argument('--test_freq', default=10, type=int,
                            help='Test interval.')
    arg_parser.add_argument('--save_freq', default=5, type=int,
                            help='Save interval.')
    arg_parser.add_argument('--crop', default=None, type=int)
    args = arg_parser.parse_args()

    # 各変数の定義
    max_epochs = args.epochs  # エポック数
    batch_size = args.batchsize  # バッチサイズ
    # 出力結果の保存先の設定および作成
    if args.output is not None:
        output_dir = args.output
    else:
        OUTPUT_BASE_DIR = './output/'
        output_dir = os.path.join(
            OUTPUT_BASE_DIR,
            f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")}'
        )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    # チェックポイントの保存先
    output_ckpt_dir = os.path.join(output_dir, 'model')
    if not os.path.exists(output_ckpt_dir):
        os.makedirs(output_ckpt_dir, exist_ok=True)
    # ログの出力先
    output_log_dir = os.path.join(output_dir, 'log')
    if not os.path.exists(output_log_dir):
        os.makedirs(output_log_dir, exist_ok=True)
    # TensorBoard用のログの出力先
    output_summary_dir = os.path.join(output_dir, 'summary')
    if not os.path.exists(output_summary_dir):
        os.makedirs(output_summary_dir, exist_ok=True)

    # ロガーの設定
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    rotate_handler = logging.handlers.RotatingFileHandler(
        os.path.join(output_log_dir, 'mnist.log'),
        maxBytes=10**6,
        backupCount=2
    )
    rotate_handler.setLevel(logging.INFO)
    rotate_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(rotate_handler)

    # 学習するモデルのアーキテクチャ
    model = DEFINED_MODELS[args.model]

    # データセットの準備
    with open(args.dataset, 'rb') as f:
        dataset = pickle.load(f)

    # 学習
    with tf.Graph().as_default():
        # 変数
        global_step = tf.Variable(0, trainable=False, name='global_step')
        x = tf.placeholder('float', shape=(None, 28, 28), name='inputs')
        y = tf.placeholder('float', shape=(None, 10), name='labels')
        cnn_keep_prob = tf.placeholder_with_default(model.DEFAULT_CNN_KEEP_PROB, shape=(), name='cnn_keep_prob')
        fc_keep_prob = tf.placeholder_with_default(model.DEFAULT_FC_KEEP_PROB, shape=(), name='fc_keep_prob')
        # モデル
        logits = model.interence(x, cnn_keep_prob, fc_keep_prob)
        # ロスとか精度とか
        loss_value = model.loss(logits, y)
        accur = model.accuracy(logits, y)
        value_holder = tf.Variable(0.0, trainable=False, dtype=tf.float32, name='val')
        with tf.name_scope('train'):
            train_loss_summary = tf.summary.scalar('loss', value_holder)
            train_acc_summary = tf.summary.scalar('acc', value_holder)
        with tf.name_scope('test'):
            test_loss_summary = tf.summary.scalar('loss', value_holder)
            test_acc_summary = tf.summary.scalar('acc', value_holder)
        # 学習ステップ
        train_step = model.trainning(loss_value)
        saver = tf.train.Saver(max_to_keep=args.keep_num)

        with tf.Session() as sess:
            writer = tf.summary.FileWriter(output_summary_dir, sess.graph)
            ckpt = tf.train.get_checkpoint_state(output_ckpt_dir)
            if ckpt:
                # チェックポイントの読み出し
                last_model = ckpt.model_checkpoint_path
                logger.info('Load checkpoint.')
                logger.info(f'Checkpoint: {last_model}')
                saver.restore(sess, last_model)
            else:
                # 変数の初期化
                init_op = tf.global_variables_initializer()
                sess.run(init_op)

            logger.info('Start Train.')
            start = sess.run(global_step) + 1
            for epoch in range(start, max_epochs+1):
                logger.info(f'epoch: {epoch}')
                sess.run(global_step.assign(epoch))
                # 訓練
                losses = []
                accures = []
                dataset.train.reset()
                while dataset.train.has_next():
                    images, labels = dataset.train.next_batch(batch_size, crop=args.crop)
                    _, loss, acc = sess.run(
                        (train_step, loss_value, accur),
                        feed_dict={
                            x: images,
                            y: labels
                        }
                    )
                    losses.append(loss)
                    accures.append(acc)
                # TensorBoard用に書き出し
                avg_loss = sess.run(value_holder.assign(np.mean(losses)))
                writer.add_summary(sess.run(train_loss_summary), epoch)
                avg_acc = sess.run(value_holder.assign(np.mean(accures)))
                writer.add_summary(sess.run(train_acc_summary), epoch)
                logger.info(f'train loss: {avg_loss}, train acc: {avg_acc}')

                # テスト
                if args.test_freq > 0 and epoch % args.test_freq == 0:
                    losses = []
                    accures = []
                    dataset.test.reset()
                    while dataset.test.has_next():
                        images, labels = dataset.test.next_batch(batch_size, crop=args.crop)
                        loss, acc = sess.run(
                            (loss_value, accur),
                            feed_dict={x: images, y: labels, cnn_keep_prob: 1.0, fc_keep_prob: 1.0}
                        )
                        losses.append(loss)
                        accures.append(acc)
                    # TensorBoard用に書き出し
                    avg_loss = sess.run(value_holder.assign(np.mean(losses)))
                    writer.add_summary(sess.run(test_loss_summary), epoch)
                    avg_acc = sess.run(value_holder.assign(np.mean(accures)))
                    writer.add_summary(sess.run(test_acc_summary), epoch)
                    logger.info(f'test loss: {avg_loss}, test acc: {avg_acc}')

                # モデルの保存
                if (args.keep_num is None or args.keep_num > 0)\
                        and (args.save_freq > 0 and epoch % args.save_freq == 0):
                    logger.info('save model.')
                    saver.save(sess, os.path.join(output_ckpt_dir, 'mnist.ckpt'), global_step=epoch)
        logger.info('Finish Train.')
