# coding: utf-8
from .MnistNN import MnistNN

# coding: utf-8
import os
import json
import numpy as np

import tensorflow as tf
from .MnistNN import MnistNN
from config import *


class MnistCNN(MnistNN):
    """CNN预测mnist数据集"""
    def __init__(self):
        super(MnistCNN, self).__init__()
        self.model_path = CNN_MODEL_SAVE_PATH
        self.regularizer= tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
        self.test_result_path = CNN_TEST_RESULT_PATH

    def inference(self, input_tensor, reuse=False, train=False):
        with tf.variable_scope('layer1-conv1'):
            conv1_weights = tf.get_variable(
                "weight", [CNN_CONV1_SIZE, CNN_CONV1_SIZE, IMAGE_CHANNELS, CNN_CONV1_DEEP],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv1_biases = tf.get_variable("bias", [CNN_CONV1_DEEP], initializer=tf.constant_initializer(0.0))
            conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
            relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

        with tf.name_scope("layer2-pool1"):
            pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        with tf.variable_scope("layer3-conv2"):
            conv2_weights = tf.get_variable(
                "weight", [CNN_CONV2_SIZE, CNN_CONV2_SIZE, CNN_CONV1_DEEP, CNN_CONV2_DEEP],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv2_biases = tf.get_variable("bias", [CNN_CONV2_DEEP], initializer=tf.constant_initializer(0.0))
            conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
            relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

        with tf.name_scope("layer4-pool2"):
            pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            pool_shape = pool2.get_shape().as_list()
            SIZEs = pool_shape[1] * pool_shape[2] * pool_shape[3]
            reshaped = tf.reshape(pool2, [pool_shape[0], SIZEs])

        with tf.variable_scope('layer5-fc1'):
            fc1_weights = tf.get_variable("weight", [SIZEs, CNN_FC_SIZE],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
            if self.regularizer is not None:
                tf.add_to_collection('losses', self.regularizer(fc1_weights))
            fc1_biases = tf.get_variable("bias", [CNN_FC_SIZE], initializer=tf.constant_initializer(0.1))

            fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
            if train:
                fc1 = tf.nn.dropout(fc1, 0.5)

        with tf.variable_scope('layer6-fc2'):
            fc2_weights = tf.get_variable("weight", [CNN_FC_SIZE, OUTPUT_SIZE],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
            if self.regularizer is not None:
                tf.add_to_collection('losses', self.regularizer(fc2_weights))
            fc2_biases = tf.get_variable("bias", [OUTPUT_SIZE], initializer=tf.constant_initializer(0.1))
            logit = tf.matmul(fc1, fc2_weights) + fc2_biases

        return logit

    def train(self):
        x = tf.placeholder(tf.float32, [
            CNN_BATCH_SIZE,
            IMAGE_SIZE,
            IMAGE_SIZE,
            IMAGE_CHANNELS],
                           name='x-input')
        y_ = tf.placeholder(tf.float32, [None, OUTPUT_SIZE], name='y-input')
        y = self.inference(x, train=True)
        global_step = tf.Variable(0, trainable=False)
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
        learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE,
            global_step,
            self.mnist.train.num_examples / CNN_BATCH_SIZE, LEARNING_RATE_DECAY,
            staircase=True)
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
        with tf.control_dependencies([train_step, variables_averages_op]):
            train_op = tf.no_op(name='train')

        saver = tf.train.Saver()

        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            for i in range(CNN_TRAINING_STEPS):
                xs, ys = self.mnist.train.next_batch(CNN_BATCH_SIZE)
                reshaped_xs = np.reshape(xs, (
                    CNN_BATCH_SIZE,
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                    IMAGE_CHANNELS))
                _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})
                if i % 1000 == 0:
                    print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                    saver.save(sess, os.path.join(self.model_path, MODEL_NAME), global_step=global_step)
            saver.save(sess, os.path.join(self.model_path, MODEL_NAME), global_step=global_step)

    def evaluate(self):
        img_num = len(self.mnist.validation.images)
        with self.graph.as_default():
            x = tf.placeholder(tf.float32, [
                img_num,
                IMAGE_SIZE,
                IMAGE_SIZE,
                IMAGE_CHANNELS],
                           name='x-input')
            y_ = tf.placeholder(tf.float32, [None, OUTPUT_SIZE], name='y-input')
            reshaped_xs = np.reshape(self.mnist.validation.images, (
                img_num,
                IMAGE_SIZE,
                IMAGE_SIZE,
                IMAGE_CHANNELS))
            validate_feed = {x: reshaped_xs,
                             y_: self.mnist.validation.labels}

            y = self.inference(x)
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
            variables_to_restore = variable_averages.variables_to_restore()

            self.restore(variables_to_restore)

            accuracy_score = self.sess.run(accuracy, feed_dict=validate_feed)
            print("validation accuracy of CNN = %g" % accuracy_score)

    def predict(self, reuse=False):
        img_num = self.mnist.test.images.shape[0]
        with self.graph.as_default():
            x = tf.placeholder(tf.float32, [
                img_num,
                IMAGE_SIZE,
                IMAGE_SIZE,
                IMAGE_CHANNELS],
                               name='x-input')
            reshaped_xs = np.reshape(self.mnist.test.images, (
                img_num,
                IMAGE_SIZE,
                IMAGE_SIZE,
                IMAGE_CHANNELS))
            y = self.inference(x)
            pred = tf.argmax(y, 1)
            test_feed = {x: reshaped_xs}
            self.restore()
            result = self.sess.run(pred, feed_dict=test_feed)
            # 预测结果输入到文本
            with open(self.test_result_path, 'wb') as f:
                np.set_printoptions(threshold='nan')
                f.write(json.dumps(str(result)))


