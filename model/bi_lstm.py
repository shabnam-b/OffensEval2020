import logging

import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn

"""
    Bi-LSTM network built upon BERT component
    For using this component, there should be a trainer file that can connect all of the components to each other.
    
    It's important that BiLSTM component is initialized with the hidden representations produced by the BERT module, 
    along with some training/evaluation configs stored in `configs` dictionary and passed to the class constructor.
    
    For now, I'm fetching learning rate (alpha) from the configs file...
    
"""


class BILSTM(object):

    def __init__(self, bert_outputs, configs):
        self.batch_size, self.seq_len, self.hidden_size = bert_outputs.shape()
        self.num_labels = 2
        self.learning_rate = configs['lr']
        self.input = bert_outputs

    def forward(self):
        logging.info("Building the computation graph...")
        hidden_var = tf.Variable(self.input)

        lstm_outputs, _ = bi_rnn(BasicLSTMCell(self.hidden_size),
                                 BasicLSTMCell(self.hidden_size),
                                 inputs=hidden_var, dtype=tf.float32)

        # Extracting forward and backward outputs
        fw_lstm, bw_lstm = lstm_outputs

        # Parameter initialization; can be wiser //TODO
        W = tf.Variable(tf.random_normal([self.hidden_size], stddev=0.1))

        # Concatenating forward and backward lstm outputs as the final hidden
        H = fw_lstm + bw_lstm  # (batch_size, seq_len, hidden)

        # //TODO @Geoff
        # Applying tanh activation
        # M = tf.tanh(H)  # M = tanh(H)  (batch_size, seq_len, HIDDEN_SIZE)