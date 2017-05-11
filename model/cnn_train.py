# -*- coding: utf-8 -*-
# !/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function

from data_unit import load_task, cut2list, vectorize_data
from cnnmodel import TextCNN
from six.moves import range
from sklearn import metrics

import tensorflow as tf
import numpy as np
import ConfigParser


tf.flags.DEFINE_float("learning_rate", 0.01, "Learning rate for SGD.")
tf.flags.DEFINE_float("anneal_rate", 25, "Number of epochs between halving the learnign rate.")
tf.flags.DEFINE_float("anneal_stop_epoch", 100, "Epoch number to end annealed lr schedule.")
tf.flags.DEFINE_integer("evaluation_interval", 10, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_size", 10, "Batch size for training.")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_integer("epochs", 500, "Number of epochs to train for.")
tf.flags.DEFINE_integer("random_state", None, "Random state.")
FLAGS = tf.flags.FLAGS

print("Started Training")

conf = ConfigParser.ConfigParser()

# task data
train, test = load_task()
data = train + test

Q, C, answer_size, sentence_size = vectorize_data(data)

c_res = []
for i in C.tolist():
    c_res.append(i.index(1))

n_train = Q.shape[0]

tf.set_random_seed(FLAGS.random_state)
batch_size = FLAGS.batch_size

batches = zip(range(0, n_train-batch_size, batch_size), range(batch_size, n_train, batch_size))
batches = [(start, end) for start, end in batches]
if batches[-1][1] < n_train:
    batches.append((batches[-1][1], n_train - 1))

conf.add_section('CNN')
conf.set("CNN", "batch_size", batch_size)
conf.set("CNN", "answer_size", answer_size)
conf.set("CNN", "sentence_size", sentence_size)
conf.set("CNN", "filter_sizes", FLAGS.filter_sizes)
conf.set("CNN", "num_filters", FLAGS.num_filters)
conf.set("CNN", "l2_reg_lambda", FLAGS.l2_reg_lambda)
conf.set("CNN", "dropout_keep_prob", FLAGS.dropout_keep_prob)
conf.write(open("./data/CNN.cfg", "w"))

with tf.Session() as sess:
    model = TextCNN(sentence_size, answer_size,
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=FLAGS.num_filters,
                    l2_reg_lambda=FLAGS.l2_reg_lambda,
                    session=sess)

    for t in range(1, FLAGS.epochs+1):
        np.random.shuffle(batches)
        total_cost = 0.0
        for start, end in batches:
            q = Q[start:end]
            c = C[start:end]
            cost_t, summary = model.batch_fit(q, c, FLAGS.dropout_keep_prob)
            model.writer.add_summary(summary)
            total_cost += cost_t

        if t % FLAGS.evaluation_interval == 0:
            train_preds = []
            for start in range(0, n_train, batch_size):
                end = start + batch_size
                q = Q[start:end]
                pred = model.predict(q, FLAGS.dropout_keep_prob)
                train_preds += list(pred)

            print(c_res)
            print(train_preds)
            train_acc = metrics.accuracy_score(np.array(train_preds), c_res)

            print('-----------------------')
            print('Epoch', t)
            print('Total Cost:', total_cost)
            print('Training Accuracy:', train_acc)
            print('-----------------------')
            model.saver.save(sess, './tensorboard/logs/data.chkp')



