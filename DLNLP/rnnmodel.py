# -*- coding: utf-8 -*-
# !/usr/bin/env python
"""End-To-End Memory Networks.

The implementation is based on http://arxiv.org/abs/1503.08895 [1]
"""
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
import json
import ConfigParser
from word import Word
from six.moves import range
from data_unit import cut2list


def zero_nil_slot(t, name=None):
    """
    Overwrites the nil_slot (first row) of the input Tensor with zeros.

    The nil_slot is a dummy slot and should not be trained and influence
    the training algorithm.
    """
    with tf.op_scope([t], name, "zero_nil_slot") as name:
        t = tf.convert_to_tensor(t, name="t")
        s = tf.shape(t)[1]
        z = tf.zeros(tf.stack([1, s]))
        return tf.concat(axis=0, values=[z, tf.slice(t, [1, 0], [-1, -1])], name=name)


def add_gradient_noise(t, stddev=1e-3, name=None):
    """
    Adds gradient noise as described in http://arxiv.org/abs/1511.06807 [2].

    The input Tensor `t` should be a gradient.

    The output will be `t` + gaussian noise.

    0.001 was said to be a good fixed value for memory networks [2].
    """
    with tf.op_scope([t, stddev], name, "add_gradient_noise") as name:
        t = tf.convert_to_tensor(t, name="t")
        gn = tf.random_normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn, name=name)


class MemN2N(object):
    """End-To-End Memory Network."""
    def __init__(self, batch_size, answer_size, sentence_size, embedding_size,
                 hops=3,
                 max_grad_norm=40.0,
                 nonlin=None,
                 initializer=tf.random_normal_initializer(stddev=0.1),
                 session=tf.Session(),
                 name='MemN2N'):
        """Creates an End-To-End Memory Network

        Args:
            batch_size: The size of the batch.

            vocab_size: The size of the vocabulary (should include the nil word). The nil word
            one-hot encoding should be 0.

            sentence_size: The max size of a sentence in the data. All sentences should be padded
            to this length. If padding is required it should be done with nil one-hot encoding (0).

            embedding_size: The size of the word embedding.

            hops: The number of hops. A hop consists of reading and addressing a memory slot.
            Defaults to `3`.

            max_grad_norm: Maximum L2 norm clipping value. Defaults to `40.0`.

            nonlin: Non-linearity. Defaults to `None`.

            initializer: Weight initializer. Defaults to `tf.random_normal_initializer(stddev=0.1)`.

            encoding: A function returning a 2D Tensor (sentence_size, embedding_size). Defaults to `position_encoding`.

            session: Tensorflow Session the model is run with. Defaults to `tf.Session()`.

            name: Name of the End-To-End Memory Network. Defaults to `MemN2N`.
        """

        self.writer = tf.summary.FileWriter("./tensorboard/logs", session.graph)
        self.vacob = {}
        self.answer = []
        self.word = Word()

        self._batch_size = batch_size
        self._answer_size = answer_size
        self._sentence_size = sentence_size
        self._embedding_size = embedding_size
        self._hops = hops
        self._max_grad_norm = max_grad_norm
        self._nonlin = nonlin
        self._init = initializer
        self._name = name

        self._build_inputs()
        self._build_vars()

        self._opt = tf.train.GradientDescentOptimizer(learning_rate=self._lr)

        # cross entropy
        logits = self._inference(self._queries)  # (batch_size, vocab_size)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.cast(self._answers, tf.float32), name="cross_entropy")
        cross_entropy_sum = tf.reduce_sum(cross_entropy, name="cross_entropy_sum")

        # loss op
        loss_op = cross_entropy_sum
        tf.summary.scalar("cost", loss_op)

        # gradient pipeline
        grads_and_vars = self._opt.compute_gradients(loss_op)
        grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v) for g,v in grads_and_vars]
        grads_and_vars = [(add_gradient_noise(g), v) for g,v in grads_and_vars]
        nil_grads_and_vars = []
        for g, v in grads_and_vars:
            if v.name in self._nil_vars:
                nil_grads_and_vars.append((zero_nil_slot(g), v))
            else:
                nil_grads_and_vars.append((g, v))
        train_op = self._opt.apply_gradients(nil_grads_and_vars, name="train_op")

        # predict ops
        predict_op = tf.argmax(logits, 1, name="predict_op")
        predict_proba_op = tf.nn.softmax(logits, name="predict_proba_op")
        predict_log_proba_op = tf.log(predict_proba_op, name="predict_log_proba_op")

        # assign ops
        self.loss_op = loss_op
        self.predict_op = predict_op
        self.predict_proba_op = predict_proba_op
        self.predict_log_proba_op = predict_log_proba_op
        self.train_op = train_op

        init_op = tf.global_variables_initializer()
        self._sess = session
        self._sess.run(init_op)
        self.saver = tf.train.Saver()
        self.merge = tf.summary.merge_all()


    def _build_inputs(self):
        self._queries = tf.placeholder(tf.float32, [None, self._sentence_size, self._embedding_size], name="queries")
        self._answers = tf.placeholder(tf.int32, [None, self._answer_size], name="answers")
        self._lr = tf.placeholder(tf.float32, [], name="learning_rate")

    def _build_vars(self):
        with tf.variable_scope(self._name):
            A = self._init([self._sentence_size, self._embedding_size])
            C = self._init([self._sentence_size, self._embedding_size])
            W = self._init([self._answer_size, self._embedding_size])


            self.W = tf.Variable(W, name="W")
            self.A_1 = tf.Variable(A, name="A")

            tf.summary.histogram('W', self.W)
            tf.summary.histogram('A_1', self.A_1)

            self.C = []

            for hopn in range(self._hops):
                with tf.variable_scope('hop_{}'.format(hopn)):
                    self.C.append(tf.Variable(C, name="C"))
                    tf.summary.histogram('hop_{}'.format(hopn), self.C[-1])

            # Dont use projection for layerwise weight sharing
            # self.H = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="H")

            # Use final C as replacement for W
            # self.W = tf.Variable(self._init([self._embedding_size, self._vocab_size]), name="W")

        self._nil_vars = set([self.A_1.name] + [x.name for x in self.C])

    def _inference(self, queries):
        with tf.variable_scope(self._name):
            # Use A_1 for thee question embedding as per Adjacent Weight Sharing
            q_emb = self.A_1 * queries
            u_0 = tf.reduce_sum(q_emb, 1)
            u = [u_0]

            for hopn in range(self._hops):
                if hopn == 0:
                    m_emb_A = self.A_1 * queries
                    m_A = tf.reduce_sum(m_emb_A, 1)

                else:
                    with tf.variable_scope('hop_{}'.format(hopn - 1)):
                        m_emb_A = self.C[hopn - 1] * queries
                        m_A = tf.reduce_sum(m_emb_A, 1)

                # hack to get around no reduce_dot
                u_temp = tf.transpose(u[-1], [0, 1])
                dotted = tf.reduce_sum(m_A * u_temp, 1)

                # Calculate probabilities
                probs = tf.nn.softmax(dotted)

                probs_temp = tf.transpose(tf.expand_dims(probs, -1), [1, 0])
                with tf.variable_scope('hop_{}'.format(hopn)):
                    m_emb_C = self.C[hopn] * queries
                m_C = tf.reduce_sum(m_emb_C, 1)

                c_temp = tf.transpose(m_C, [1, 0])
                o_k = tf.reduce_sum(c_temp * probs_temp, 1)

                # Dont use projection layer for adj weight sharing
                # u_k = tf.matmul(u[-1], self.H) + o_k

                u_k = u[-1] + o_k

                # nonlinearity
                if self._nonlin:
                    u_k = nonlin(u_k)

                u.append(u_k)

            # Use last C for output (transposed)
            with tf.variable_scope('hop_{}'.format(self._hops)):
                return tf.matmul(u_k, tf.transpose(self.W, [1,0]))

    def batch_fit(self, queries, answers, learning_rate):
        """Runs the training algorithm over the passed batch

        Args:
            learning_rate: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)
            answers: Tensor (None, vocab_size)

        Returns:
            loss: floating-point number, the loss computed for the batch
        """
        feed_dict = {self._queries: queries, self._answers: answers, self._lr: learning_rate}

        loss, _, summary = self._sess.run([self.loss_op, self.train_op, self.merge], feed_dict=feed_dict)

        return loss, summary

    def predict(self, queries):
        """Predicts answers as one-hot encoding.

        Args:
            queries: Tensor (None, sentence_size)

        Returns:
            answers: Tensor (None, vocab_size)
        """
        feed_dict = {self._queries: queries}
        return self._sess.run(self.predict_op, feed_dict=feed_dict)

    def predict_proba(self, queries):
        """Predicts probabilities of answers.

        Args:
            queries: Tensor (None, sentence_size)

        Returns:
            answers: Tensor (None, vocab_size)
        """
        feed_dict = {self._queries: queries}
        return self._sess.run(self.predict_proba_op, feed_dict=feed_dict)

    def predict_log_proba(self, queries):
            """Predicts log probabilities of answers.
    
            Args:
                queries: Tensor (None, sentence_size)
            Returns:
                answers: Tensor (None, vocab_size)
            """
            feed_dict = {self._queries: queries}
            return self._sess.run(self.predict_log_proba_op, feed_dict=feed_dict)

    def load(self, checkpoint_dir):
        tf.logging.warning("model start load")
        with open("./data/ans.json", 'r') as pf:
            self.answer = json.load(pf)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self._sess, ckpt.model_checkpoint_path)
            tf.logging.warning("model restore success")
        else:
            tf.logging.error("model restore wrong!")

    def find_simword(self, word):
        simword = ""
        maxscore = 0
        for index, value in enumerate(self.vacob.keys()):
            score = self.word.word_sim(value, word)
            if score > maxscore:
                simword = value
                maxscore = score
        tf.logging.warning("find_simword, word is {}, simword is {}".format(word, simword))
        return simword

    def string_to_vec(self, string):
        vector = [[0] * self._embedding_size] * self._sentence_size
        strlist = cut2list(string)
        for index, word in enumerate(strlist):
            vecjs = self.word.word_vec(word)
            vec = json.loads(vecjs)
            while isinstance(vec, unicode):
                vec = json.loads(vec)
            vector[index] = vec
        return vector

    def vec_to_answer(self, maxindex):
        return self.answer[maxindex]

    def respond(self, query):
        qvec = self.string_to_vec(query)
        feed_dict = {self._queries:[qvec]}
        maxindex = self._sess.run(self.predict_op, feed_dict=feed_dict)
        answer = self.vec_to_answer(maxindex[0])
        return answer


if __name__ == "__main__":
    # for test
    mdir = "./tensorboard/logs/"
    conf = ConfigParser.ConfigParser()
    conf.read("./data/RNN.cfg")
    batch_size = int(conf.get("RNN", "batch_size"))
    answer_size = int(conf.get("RNN", "answer_size"))
    sentence_size = int(conf.get("RNN", "sentence_size"))
    embedding_size = int(conf.get("RNN", "embedding_size"))
    hops = int(conf.get("RNN", "hops"))
    max_grad_norm = float(conf.get("RNN", "max_grad_norm"))

    with tf.Session() as sess:
        model = MemN2N(batch_size, answer_size, sentence_size, embedding_size, session=sess,
                       hops=hops, max_grad_norm=max_grad_norm)
        model.load('./model/rnn/')
        while(1):
            print(model.respond(raw_input(">")))
