import tensorflow as tf
from tensorflow.contrib import rnn

from utils.utils import load_glove_embeddings


class BertClassificationModel(object):
    def __init__(self, vocabulary_size, num_class, args, vocab):
        self.embedding_size = args.embedding_size
        self.num_layers = args.num_layers
        self.num_hidden = args.num_hidden
        self.num_hidden = self.num_hidden * 2

        self.X = tf.placeholder(tf.int32, [None, args.max_document_len])
        self.Y1 = tf.placeholder(tf.int32, [None])
        self.Y2 = tf.placeholder(tf.int32, [None])
        self.dropout = tf.placeholder(tf.float64, [])

        self.X_len = tf.reduce_sum(tf.sign(self.X), 1)

        with tf.name_scope("embedding"):
            # Embeddings whether GloVe or Bert
            # self.embeddings = load_embeddings(vocab)

            # Bert Embeddings
            # init_embeddings = load_bert_embeddings(self.X, vocab)

            # Glove Embeddings
            # init_embeddings = load_bert_embeddings(vocab)

            # Random Embeddingss
            init_embeddings = tf.random_uniform([vocabulary_size, self.embedding_size])
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
            self.subtaskb_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.subtaskb_logits, labels=self.Y2))
            self.Loss = self.subtaska_loss + self.subtaskb_loss

        with tf.name_scope("subtaska-accuracy"):
            correct_predictions = tf.equal(self.subtaska_predictions, self.Y1)
            self.suba_accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))

        with tf.name_scope("subtaskb-accuracy"):
            correct_predictions = tf.equal(self.subtaskb_predictions, self.Y2)
            self.subb_accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))

    def make_cell(self):
        cell = rnn.BasicLSTMCell(self.num_hidden / 2)
        cell = rnn.DropoutWrapper(cell, output_keep_prob=self.dropout)
        return cell
