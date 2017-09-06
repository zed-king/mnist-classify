# coding: utf-8
import os
import numpy as np

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from config import NNIST_DATA_PATH


class MnistNN(object):
    """mnist数据集三种分类方法基类"""
    def __init__(self):
        """定义神经网络的参数"""
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)            # 神经网络会话
        self.model_path = ''                        # 模型保存路径
        self.regularizer = None                     # 归一化参数，用于抑制过拟合
        self.mnist = input_data.read_data_sets(NNIST_DATA_PATH, one_hot=True)

    def get_weight(self, shape):
        """
        获取权重参数
        :param shape: 参数尺寸
        :param regularizer: 归一化参数，用于抑制过拟合
        :return:
        """
        weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
        if self.regularizer is not None:
            tf.add_to_collection('losses', self.regularizer(weights))
        return weights

    def inference(self, *args, **kwargs):
        """
        定义接口
        :param input_tensor: 输入张量
        :param regularizer: 归一化参数
        :return: 输出张量
        """
        raise NotImplementedError

    def train(self, *args, **kwargs):
        """
        用于训练神经网络结构
        :return:
        """
        raise NotImplementedError

    def evaluate(self, *args, **kwargs):
        """
        用验证集评估模型准确性
        :return:
        """
        raise NotImplementedError

    def predict(self):
        """
        预测测试集结果
        :return:
        """
        raise NotImplementedError

    def restore(self, variables_to_restore=None):
        """
        读取神经网络模型
        :return:
        """

        saver = tf.train.Saver(variables_to_restore)
        with self.sess.as_default():
            ckpt = tf.train.get_checkpoint_state(self.model_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(self.sess, ckpt.model_checkpoint_path)
            else:
                raise Exception(u'No checkpoint file found in %s' % self.model_path)
        return ckpt

    def __del__(self):
        """
        释放session
        :return:
        """
        if self.sess:
            self.sess.close()
