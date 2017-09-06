# coding: utf-8
import os
import numpy as np

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from .MnistNN import MnistNN


# coding: utf-8
import os
import json
import numpy as np

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from .MnistNN import MnistNN
from config import *


class MnistRNN(MnistNN):
    """RNN预测mnist数据集"""
    def __init__(self):
        super(MnistRNN, self).__init__()
        self.model_path = RNN_MODEL_SAVE_PATH
        self.regularizer= tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
        self.test_result_path = RNN_TEST_RESULT_PATH

    def inference(self, input_tensor, reuse=False):
        # 定义变量
        # with tf.variable_scope('rnn', reuse=reuse):
        weights = self.get_weight([RNN_HIDDEN_SIZE, OUTPUT_SIZE])
        biases = tf.get_variable("biases", [OUTPUT_SIZE], initializer=tf.constant_initializer(0.0))

        # 转换输入数据，将输入数据由[ batch_size,nsteps,n_input] 变为 [ nsteps，batch_size,n_input]
        input_tensor = tf.transpose(input_tensor, [1, 0, 2])
        input_tensor = tf.reshape(input_tensor, [-1, RNN_INPUT_SIZE])
        input_tensor = tf.split(input_tensor, RNN_STEP_SIZE, 0)
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(RNN_HIDDEN_SIZE)
        outputs, states = tf.nn.static_rnn(lstm_cell, input_tensor, dtype=tf.float32)
        return tf.matmul(outputs[-1], weights) + biases

    def train(self):
        x = tf.placeholder(tf.float32, [None, RNN_STEP_SIZE, RNN_INPUT_SIZE], name='x-input')
        y = tf.placeholder(tf.float32, [None, OUTPUT_SIZE], name='y-input')
        # RNN
        y_ = self.inference(x)
        # # softmax交叉熵值损失
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=y))
        train_step = tf.train.AdamOptimizer(learning_rate=RNN_LEARNING_RATE).minimize(loss)
        #
        # # 计算准确率
        correct_pred = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        saver = tf.train.Saver()
        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            for i in range(RNN_TRAINING_STEPS):
                batch_xs, batch_ys = self.mnist.train.next_batch(RNN_BATCH_SIZE)
                batch_xs = batch_xs.reshape((RNN_BATCH_SIZE, RNN_STEP_SIZE, RNN_INPUT_SIZE))

                _, loss_value, ss = sess.run([accuracy, loss, train_step], feed_dict={x: batch_xs, y: batch_ys})
                if i % 1000 == 0:
                    print("After %d training step(s), loss on training batch is %g." % (i, loss_value))
                    saver.save(sess, os.path.join(self.model_path, MODEL_NAME), global_step=i)
            saver.save(sess, os.path.join(self.model_path, MODEL_NAME), global_step=i)

    def evaluate(self):
        img_num = self.mnist.validation.images.shape[0]
        with self.graph.as_default():
            x = tf.placeholder(tf.float32, [None, RNN_STEP_SIZE, RNN_INPUT_SIZE], name='x-input')
            y = tf.placeholder(tf.float32, [None, OUTPUT_SIZE], name='y-input')
            xs, ys = self.mnist.validation.next_batch(img_num)
            xs = xs.reshape((img_num, RNN_STEP_SIZE, RNN_INPUT_SIZE))
            validate_feed = {x: xs, y: ys}
            # RNN
            y_ = self.inference(x)

            # # 计算准确率
            correct_pred = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

            self.restore()

            accuracy_score = self.sess.run(accuracy, feed_dict=validate_feed)
            print("validation accuracy of RNN = %g" % (accuracy_score))

    def predict(self, reuse=False):
        img_num = self.mnist.test.images.shape[0]
        with self.graph.as_default():
            x = tf.placeholder(tf.float32, [None, RNN_STEP_SIZE, RNN_INPUT_SIZE], name='x-input')
            y = self.inference(x, reuse)
            pred = tf.argmax(y, 1)
            xs, _ = self.mnist.test.next_batch(img_num)
            xs = xs.reshape((img_num, RNN_STEP_SIZE, RNN_INPUT_SIZE))
            test_feed = {x: xs}
            self.restore()
            result = self.sess.run(pred, feed_dict=test_feed)
            # 预测结果输入到文本
            with open(self.test_result_path, 'wb') as f:
                np.set_printoptions(threshold='nan')
                f.write(json.dumps(str(result)))


