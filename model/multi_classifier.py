import tensorflow as tf
from tensorflow.contrib import rnn

from utils.utils import load_embeddings


class Model(object):
    def __init__(self, vocabulary_size, num_class, args, vocab):
        self.embedding_size = args.embedding_size
        self.num_layers = args.num_layers
        self.num_hidden = args.num_hidden

        self.X = tf.placeholder(tf.int32, [None, args.max_document_len])
        self.Y1 = tf.placeholder(tf.int32, [None])
        self.Y2 = tf.placeholder(tf.int32, [None])
        self.dropout = tf.placeholder(tf.float64, [])

        self.X_len = tf.reduce_sum(tf.sign(self.X), 1)

        with tf.name_scope("embedding"):
            # self.embeddings = load_embeddings(vocab)
            init_embeddings = load_embeddings(vocab)
            # init_embeddings = tf.random_uniform([vocabulary_size, self.embedding_size])
            embeddings = tf.get_variable("embeddings", initializer=init_embeddings)
            self.x_emb = tf.nn.embedding_lookup(embeddings, self.X)

        with tf.name_scope("rnn"):
            cell = rnn.MultiRNNCell([self.make_cell() for _ in range(self.num_layers)])
            rnn_outputs, _ = tf.nn.dynamic_rnn(
                cell, self.x_emb, sequence_length=self.X_len, dtype=tf.float64)

        with tf.name_scope("subtaska-output"):
            rnn_outputs_flat = tf.reshape(rnn_outputs, [-1, args.max_document_len * self.num_hidden])
            self.subtaska_logits = tf.layers.dense(rnn_outputs_flat, num_class)
            self.subtaska_predictions = tf.argmax(self.subtaska_logits, -1, output_type=tf.int32)

        with tf.name_scope("subtaskb-output"):
            rnn_outputs_flat = tf.reshape(rnn_outputs, [-1, args.max_document_len * self.num_hidden])
            self.subtaskb_logits = tf.layers.dense(rnn_outputs_flat, num_class)
            self.subtaskb_predictions = tf.argmax(self.subtaskb_logits, -1, output_type=tf.int32)

        with tf.name_scope("loss"):
            self.subtaska_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.subtaska_logits, labels=self.Y1))
            self.subtaskb_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.subtaskb_logits, labels=self.Y2))
            self.Loss = self.subtaska_loss + self.subtaskb_loss

        with tf.name_scope("subtaska-accuracy"):
            correct_predictions = tf.equal(self.subtaska_predictions, self.Y1)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))

        # with tf.name_scope("subtaskb-accuracy"):
        #     correct_predictions = tf.equal(self.subtaskb_predictions, self.Y2)
        #     self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))

    def make_cell(self):
        cell = rnn.BasicLSTMCell(self.num_hidden)
        cell = rnn.DropoutWrapper(cell, output_keep_prob=self.dropout)
        return cell
