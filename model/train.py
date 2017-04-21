# -*- coding: utf-8 -*-
# !/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function

from data_unit import load_task, cut2list, vectorize_data
from nnmodel import MemN2N
from six.moves import range
from functools import reduce
from sklearn import metrics

import tensorflow as tf
import numpy as np
import ConfigParser


tf.flags.DEFINE_float("learning_rate", 0.01, "Learning rate for SGD.")
tf.flags.DEFINE_float("anneal_rate", 25, "Number of epochs between halving the learnign rate.")
tf.flags.DEFINE_float("anneal_stop_epoch", 100, "Epoch number to end annealed lr schedule.")
tf.flags.DEFINE_float("max_grad_norm", 40.0, "Clip gradients to this norm.")
tf.flags.DEFINE_integer("evaluation_interval", 10, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_size", 32, "Batch size for training.")
tf.flags.DEFINE_integer("hops", 3, "Number of hops in the Memory Network.")
tf.flags.DEFINE_integer("epochs", 5000, "Number of epochs to train for.")
tf.flags.DEFINE_integer("embedding_size", 100, "Embedding size for embedding matrices.")
tf.flags.DEFINE_integer("memory_size", 50, "Maximum size of memory.")
tf.flags.DEFINE_integer("random_state", None, "Random state.")
FLAGS = tf.flags.FLAGS

print("Started Training")

conf = ConfigParser.ConfigParser()

# task data
train, test = load_task()
data = train + test

vocab = sorted(reduce(lambda x, y: x | y, (set(cut2list(q)) for q, c in data)))
word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

sentence_size = max(map(len, (q for q, c in data)))
vocab_size = len(vocab) + 1  # +1 for nil word

Q, C, answer_size = vectorize_data(data, word_idx, sentence_size)

c_res = []
for i in C.tolist():
    c_res.append(i.index(1))

n_train = Q.shape[0]

tf.set_random_seed(FLAGS.random_state)
batch_size = FLAGS.batch_size

batches = zip(range(0, n_train-batch_size, batch_size), range(batch_size, n_train, batch_size))
batches = [(start, end) for start, end in batches]

conf.add_section('RNN')
conf.set("RNN", "batch_size", batch_size)
conf.set("RNN", "answer_size", answer_size)
conf.set("RNN", "sentence_size", sentence_size)
conf.set("RNN", "embedding_size", FLAGS.embedding_size)
conf.set("RNN", "vocab_size", vocab_size)
conf.set("RNN", "hops", FLAGS.hops)
conf.set("RNN", "max_grad_norm", FLAGS.max_grad_norm)
conf.write(open("./data/RNN.cfg", "w"))

with tf.Session() as sess:
    model = MemN2N(batch_size, answer_size, sentence_size, FLAGS.embedding_size, vocab_size, session=sess,
                   hops=FLAGS.hops, max_grad_norm=FLAGS.max_grad_norm)
    for t in range(1, FLAGS.epochs+1):
        # Stepped learning rate
        if t - 1 <= FLAGS.anneal_stop_epoch:
            anneal = 2.0 ** ((t - 1) // FLAGS.anneal_rate)
        else:
            anneal = 2.0 ** (FLAGS.anneal_stop_epoch // FLAGS.anneal_rate)
        lr = FLAGS.learning_rate / anneal

        np.random.shuffle(batches)
        total_cost = 0.0
        for start, end in batches:
            q = Q[start:end]
            c = C[start:end]
            cost_t, summary = model.batch_fit(q, c, lr)
            model.writer.add_summary(summary)
            total_cost += cost_t

        if t % FLAGS.evaluation_interval == 0:
            train_preds = []
            for start in range(0, n_train, batch_size):
                end = start + batch_size
                q = Q[start:end]
                pred = model.predict(q)
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



