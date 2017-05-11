# -*- coding: utf-8 -*-
# !/usr/bin/env python
import tensorflow as tf
import ConfigParser
import json
from word import Word
from data_unit import cut2list


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


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self, sentence_size, answer_size, filter_sizes, num_filters,
                 l2_reg_lambda=0.0,
                 embedding_size = 128,
                 max_grad_norm=40.0,
                 session=tf.Session(),
                 name = 'TextCNN'):

            self.writer = tf.summary.FileWriter("./tensorboard/logs", session.graph)
            self.answer = []
            self.word = Word()

            self._answer_size = answer_size
            self._sentence_size = sentence_size
            self._embedding_size = embedding_size
            self._max_grad_norm = max_grad_norm
            self._name = name
            self._embedding_size = embedding_size
            self._filter_sizes = filter_sizes
            self._num_filters = num_filters
            self._l2_reg_lambda = l2_reg_lambda
            self.loss = None
            self.accuracy = None
            self.predictions = None

            self._build_inputs()
            self._inference()

            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            self._optimizer = tf.train.AdamOptimizer(1e-3)

            # gradient pipeline
            grads_and_vars = self._optimizer.compute_gradients(self.loss)

            train_op = self._optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

            # Summaries for loss and accuracy
            tf.summary.scalar("loss", self.loss)
            tf.summary.scalar("accuracy", self.accuracy)

            # assign ops
            self.loss_op = self.loss
            self.predict_op = self.predictions
            self.train_op = train_op

            init_op = tf.global_variables_initializer()
            self._sess = session
            self._sess.run(init_op)
            self.saver = tf.train.Saver()
            self.merge = tf.summary.merge_all()

    def _build_inputs(self):
        self.input_x = tf.placeholder(tf.float32, [None, self._sentence_size, self._embedding_size], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self._answer_size], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

    def _inference(self):
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.embedded_chars = self.input_x
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(self._filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, self._embedding_size, 1, self._num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self._num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self._sentence_size - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = self._num_filters * len(self._filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, self._answer_size],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self._answer_size]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + self._l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def batch_fit(self, queries, answers, dropout_keep_prob):
        """Runs the training algorithm over the passed batch

        Args:
            queries: Tensor (None, sentence_size)
            answers: Tensor (None, vocab_size)

        Returns:
            loss: floating-point number, the loss computed for the batch
        """
        feed_dict = {self.input_x: queries, self.input_y: answers, self.dropout_keep_prob: dropout_keep_prob}

        loss, _, summary = self._sess.run([self.loss_op, self.train_op, self.merge], feed_dict=feed_dict)

        return loss, summary

    def predict(self, queries, dropout_keep_prob):
        """Predicts answers as one-hot encoding.

        Args:
            queries: Tensor (None, sentence_size)

        Returns:
            answers: Tensor (None, vocab_size)
        """
        feed_dict = {self.input_x: queries, self.dropout_keep_prob: dropout_keep_prob}
        return self._sess.run(self.predict_op, feed_dict=feed_dict)

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

    def respond(self, query, dropout_keep_prob):
        qvec = self.string_to_vec(query)
        feed_dict = {self.input_x:[qvec], self.dropout_keep_prob: dropout_keep_prob}
        maxindex = self._sess.run(self.predict_op, feed_dict=feed_dict)
        answer = self.vec_to_answer(maxindex[0])
        return answer



if __name__ == "__main__":
    mdir = "./tensorboard/logs/"
    conf = ConfigParser.ConfigParser()
    conf.read("./data/CNN.cfg")
    answer_size = int(conf.get("CNN", "answer_size"))
    sentence_size = int(conf.get("CNN", "sentence_size"))
    filter_sizes = conf.get("CNN", "filter_sizes")
    num_filters = int(conf.get("CNN", "num_filters"))
    l2_reg_lambda = float(conf.get("CNN", "l2_reg_lambda"))
    dropout_keep_prob = float(conf.get("CNN", "dropout_keep_prob"))

    with tf.Session() as sess:
        model = TextCNN(sentence_size, answer_size,
                    filter_sizes=list(map(int, filter_sizes.split(","))),
                    num_filters=num_filters,
                    l2_reg_lambda=l2_reg_lambda,
                    session=sess)
        model.load('./model/cnn/')
        while(1):
            print(model.respond(raw_input(">"), dropout_keep_prob))

