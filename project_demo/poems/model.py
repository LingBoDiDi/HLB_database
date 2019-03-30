# -*- coding: utf-8 -*-
# file: model.py
# author: JinTian
# time: 07/03/2017 3:07 PM
# Copyright 2017 JinTian. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
import tensorflow as tf
import numpy as np


def rnn_model(model, input_data, output_data, vocab_size, rnn_size=128, num_layers=2, batch_size=64,
              learning_rate=0.01):
    """
    construct rnn seq2seq model.
    :param model: model class
    :param input_data: input data placeholder
    :param output_data: output data placeholder
    :param vocab_size:
    :param rnn_size:
    :param num_layers:
    :param batch_size:
    :param learning_rate:
    :return:
    """
    end_points = {}

    if model == 'rnn':
        cell_fun = tf.contrib.rnn.BasicRNNCell
    elif model == 'gru':
        cell_fun = tf.contrib.rnn.GRUCell
    elif model == 'lstm':
        cell_fun = tf.contrib.rnn.BasicLSTMCell

    cell = cell_fun(rnn_size, state_is_tuple=True)
    cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)

    if output_data is not None:
        initial_state = cell.zero_state(batch_size, tf.float32)  #...
    else:
        initial_state = cell.zero_state(1, tf.float32)

    """
    使用with后不管with中的代码出现什么错误，都会进行对当前对象进行清理工作。
    指定运行的cpu
    """
    with tf.device("/cpu:0"):
        embedding = tf.get_variable('embedding', initializer=tf.random_uniform(
            [vocab_size + 1, rnn_size], -1.0, 1.0))  #创建一个新的变量：变量名，分布方式（均匀分布）
        inputs = tf.nn.embedding_lookup(embedding, input_data)  #选取一个张量里面索引对应的元素

    #训练前初始化
    # [batch_size, ?, rnn_size] = [64, ?, 128]
    outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state)  #outputs包含了所有时刻的输出 H，last_state包含了最后一个时刻的输出 H 和 C（隐含嵌套循环）
    output = tf.reshape(outputs, [-1, rnn_size])  #转化为n*rnn_size维度的矩阵（-1为自动计算矩阵的行数）

    weights = tf.Variable(tf.truncated_normal([rnn_size, vocab_size + 1]))  #正态分布产生隐藏层权重（初始化变量weights）
    bias = tf.Variable(tf.zeros(shape=[vocab_size + 1]))  #偏置
    logits = tf.nn.bias_add(tf.matmul(output, weights), bias=bias)  #加偏置后
    # [?, vocab_size+1]

    if output_data is not None:
        # output_data must be one-hot encode
        labels = tf.one_hot(tf.reshape(output_data, [-1]), depth=vocab_size + 1)  #将多个数值联合放在一起作为多个相同类型的向量，可用于表示各自的概率分布
        # should be [?, vocab_size+1]

        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        # loss shape should be [?, vocab_size+1]
        total_loss = tf.reduce_mean(loss)
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)  #使用梯度下降法进行训练

        end_points['initial_state'] = initial_state
        end_points['output'] = output
        end_points['train_op'] = train_op
        end_points['total_loss'] = total_loss
        end_points['loss'] = loss
        end_points['last_state'] = last_state
    else:
        prediction = tf.nn.softmax(logits)

        end_points['initial_state'] = initial_state
        end_points['last_state'] = last_state
        end_points['prediction'] = prediction

    return end_points
