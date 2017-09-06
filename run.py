# coding: utf-8
import os
import numpy as np
import struct
import matplotlib.pyplot as plt
from model import MnistDNN, MnistCNN, MnistRNN


def run():
    nn_map = dict(dnn=MnistDNN, cnn=MnistCNN, rnn=MnistRNN)
    input = raw_input(u'请选择模型，dnn、cnn、rnn: ')
    class_nn = nn_map.get(input)
    if class_nn:
        nn = class_nn()
        # nn.train()            # 重新训练的话，取消注释
        is_predict = True
        if not is_predict:
            nn.evaluate()
        else:
            nn.predict()     # 如与连用则reuse=True,否则为False;; rnn  evaluate,predict有bug
    else:
        print u'输入必须为：dnn cnn rnn'

if __name__ == '__main__':
    run()
