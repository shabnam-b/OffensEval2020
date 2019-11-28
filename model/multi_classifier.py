import tensorflow as tf
from tensorflow.contrib import rnn

from utils import load_embeddings


class MultiClassifier():

    def __init__(self, num_labels, FLAGS):

        # self.dropout = args.dropout
        self.y_t1 = None
        self.y_t2 = None
        self.max_len = FLAGS.max_document_len
        self.emb_size = FLAGS.embedding_size
        self.num_layers = FLAGS.num_layers
        self.num_hidden = FLAGS.num_hidden
        self.num_labels = num_labels

        self.inp = tf.placeholder(tf.int32, [None, FLAGS.max_document_len])
        self.lm_y = tf.placeholder(tf.int32, [None, FLAGS.max_document_len])
        self.clf_y = tf.placeholder(tf.int32, [None])

        self.dropout = tf.placeholder(tf.float32, [])
        self.emb_size = FLAGS.embedding_size

        self.x_len = tf.reduce_sum(tf.sign(self.inp), 1)


        # Load the embeddings
        with tf.name_scope("embedding"):
            embeddings = tf.get_variable("embeddings", initializer=load_embeddings())
            self.vocab_size = embeddings.size(0)
            self.x_emb = tf.nn.embedding_lookup(embeddings, self.input_ids)

        with tf.name_scope("birnn"):
            lstm_cell = rnn.MultiRNNCell([self.make_cell() for _ in range(self.num_layers)])
            lstm_outputs, _ = tf.nn.dynamic_rnn(lstm_cell, self.x_emb, sequence_length=self.x_len, dtype=tf.float32)

        flattened = tf.reshape(lstm_outputs, [-1, self.max_len * self.num_hidden])

        with tf.name_scope("task1-output"):
            self.task1_logits = tf.layers.dense(flattened, self.vocab_size)
            self.pred1 = tf.argmax(self.task1_logits, axis=-1)

        with tf.name_scope("task2-output"):
            self.task2_logits = tf.layers.dense(flattened, self.num_labels)
            self.pred2 = tf.argmax(self.task2_logits, axis=-1)


        with tf.name_scope("loss"):
            self.loss_t1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.task1_logits,
                labels=self.y_t1))
            self.loss_t2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.task2_logits,
                labels=self.y_t2))

            self.Loss = self.loss_t1 + self.loss_t2


    def make_cell(self):
        lstm_cell = rnn.BasicLSTMCell(self.num_hidden)
        lstm_cell_drp = rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.dropout)
        return lstm_cell_drp