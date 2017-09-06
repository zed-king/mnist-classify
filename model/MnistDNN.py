# coding: utf-8
import os
import json
import numpy as np

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from .MnistNN import MnistNN
from config import *


class MnistDNN(MnistNN):
    """DNN预测mnist数据集"""
    def __init__(self):
        super(MnistDNN, self).__init__()
        self.model_path = DNN_MODEL_SAVE_PATH
        self.regularizer= tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
        self.test_result_path = DNN_TEST_RESULT_PATH

    def inference(self, input_tensor, reuse=False):
        with tf.variable_scope('layer1', reuse=reuse):
            weights = self.get_weight([DNN_INPUT_SIZE, DNN_LAYER1_SIZE])
            biases = tf.get_variable("biases", [DNN_LAYER1_SIZE], initializer=tf.constant_initializer(0.0))
            layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

        with tf.variable_scope('layer2', reuse=reuse):
            weights = self.get_weight([DNN_LAYER1_SIZE, OUTPUT_SIZE])
            biases = tf.get_variable("biases", [OUTPUT_SIZE], initializer=tf.constant_initializer(0.0))
            layer2 = tf.matmul(layer1, weights) + biases

        return layer2

    def train(self):
        x = tf.placeholder(tf.float32, [None, DNN_INPUT_SIZE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, OUTPUT_SIZE], name='y-input')
        y = self.inference(x)
        global_step = tf.Variable(0, trainable=False)
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
        learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE,
            global_step,
            self.mnist.train.num_examples / DNN_BATCH_SIZE, LEARNING_RATE_DECAY,
            staircase=True)
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
        with tf.control_dependencies([train_step, variables_averages_op]):
            train_op = tf.no_op(name='train')

        saver = tf.train.Saver()
        with self.sess.as_default():
            tf.global_variables_initializer().run()

            for i in range(DNN_TRAINING_STEPS):
                xs, ys = self.mnist.train.next_batch(DNN_BATCH_SIZE)
                _, loss_value, step = self.sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
                if i % 1000 == 0:
                    print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                    saver.save(self.sess, os.path.join(self.model_path, MODEL_NAME), global_step=global_step)
            saver.save(self.sess, os.path.join(self.model_path, MODEL_NAME), global_step=global_step)

    def evaluate(self):
        with self.graph.as_default():
            x = tf.placeholder(tf.float32, [None, DNN_INPUT_SIZE], name='x-input')
            y_ = tf.placeholder(tf.float32, [None, OUTPUT_SIZE], name='y-input')
            validate_feed = {x: self.mnist.validation.images,
                             y_: self.mnist.validation.labels}

            y = self.inference(x)
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
            variables_to_restore = variable_averages.variables_to_restore()

            self.restore(variables_to_restore)

            accuracy_score = self.sess.run(accuracy, feed_dict=validate_feed)
            print("validation accuracy of DNN = %g" % (accuracy_score))

    def predict(self, reuse=False):
        with self.graph.as_default():
            x = tf.placeholder(tf.float32, [None, DNN_INPUT_SIZE], name='x-input')
            y = self.inference(x, reuse)
            pred = tf.argmax(y, 1)
            test_feed = {x: self.mnist.test.images}
            self.restore()
            result = self.sess.run(pred, feed_dict=test_feed)
            # 预测结果输入到文本
            with open(self.test_result_path, 'wb') as f:
                np.set_printoptions(threshold='nan')
                f.write(json.dumps(str(result)))


