import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

from utils.utils import load_embeddings


class Model(object):
    def __init__(self, num_class, args, vocab):
        self.embedding_size = args.embedding_size
        self.num_layers = args.num_layers
        self.num_hidden = args.num_hidden

        self.X = tf.placeholder(tf.int32, [None, args.max_document_len])
        self.Y1 = tf.placeholder(tf.int32, [None])
        self.Y2 = tf.placeholder(tf.int32, [None])
        self.dropout = tf.placeholder(tf.float64, [])

        self.X_len = tf.reduce_sum(tf.sign(self.X), 1)

        with tf.name_scope("embedding"):
            init_embeddings = load_embeddings(vocab)
            # Random Embeddingss
            # init_embeddings = tf.random_uniform([vocabulary_size, self.embedding_size])
            embeddings = tf.get_variable("embeddings", initializer=init_embeddings)
            self.x_emb = tf.nn.embedding_lookup(embeddings, self.X)

        with tf.name_scope("rnn"):
            fw_multi_cell = rnn.MultiRNNCell([self.make_cell() for _ in range(self.num_layers)])
            bw_multi_cell = rnn.MultiRNNCell([self.make_cell() for _ in range(self.num_layers)])
            rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                fw_multi_cell, bw_multi_cell, self.x_emb, sequence_length=self.X_len, dtype=tf.float64)
            rnn_outputs = tf.concat(rnn_outputs, 2)

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

            valid_idxs = tf.where(self.Y2 < 2)[:, 0]
            valid_logits = tf.gather(self.subtaskb_logits, valid_idxs)
            valid_labels = tf.gather(self.Y2, valid_idxs)
            self.subtaskb_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=valid_labels, logits=valid_logits))
            self.Loss = self.subtaska_loss + self.subtaskb_loss

        with tf.name_scope("subtaska-accuracy"):
            correct_predictions = tf.equal(self.subtaska_predictions, self.Y1)
            self.suba_accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))

        with tf.name_scope("subtaskb-accuracy"):
            correct_predictions = tf.equal(self.subtaskb_predictions, self.Y2)
            self.subb_accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))

        with tf.name_scope("subtaska-f1"):
            TP1 = tf.count_nonzero(self.subtaska_predictions * self.Y1)
            TN1 = tf.count_nonzero((self.subtaska_predictions - 1) * (self.Y1 - 1))
            FP1 = tf.count_nonzero(self.subtaska_predictions * (self.Y1 - 1))
            FN1 = tf.count_nonzero((self.subtaska_predictions - 1) * self.Y1)
            precision1 = TP1 / (TP1 + FP1)
            recall1 = TP1 / (TP1 + FN1)

            self.f1_suba = 2 * precision1 * recall1 / (precision1 + recall1)

        with tf.name_scope("subtaskb-f1"):
            TP = tf.count_nonzero(self.subtaskb_predictions * self.Y2)
            TN = tf.count_nonzero((self.subtaskb_predictions - 1) * (self.Y2 - 1))
            FP = tf.count_nonzero(self.subtaskb_predictions * (self.Y2 - 1))
            FN = tf.count_nonzero((self.subtaskb_predictions - 1) * self.Y2)
            self.precision = TP / (TP + FP)
            self.recall = TP / (TP + FN)
            self.f1_subb = 2 * self.precision * self.recall / (self.precision + self.recall)

    def make_cell(self):
        cell = rnn.BasicLSTMCell(self.num_hidden / 2)
        cell = rnn.DropoutWrapper(cell, output_keep_prob=self.dropout)
        return cell
