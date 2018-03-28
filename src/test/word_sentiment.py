import tensorflow as tf


class WordModel(object):

    def __init__(self,
                 glossary_size,
                 embedding_size,
                 hidden_size):
        self.glossary_size = glossary_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size


    def buildGraph(self):
        inputs = self.define_IO()
        self.trainable_parameters()
        inputs = self.embedding_layer(inputs)
        outputs = self.bi_attn_lstm_layer(inputs)
        outputs = self.bi_sigmoid_layer(outputs)


    def define_IO(self):
        self.inputs = tf.placeholder(tf.int32, shape=[None, 1], name='inputs')
        self.pretrained_wv = tf.placeholder(tf.float32, shape=[self.glossary_size, self.embedding_size])
        return self.inputs


    def trainable_parameters(self):
        self.keep_prob = 1

        with tf.name_scope('embeddings'), tf.variable_scope('embeddings'):
            self.embeddings = tf.Variable(self.pretrained_wv, name='embeddings')
        with tf.name_scope('lstm'), tf.variable_scope('lstm'):
            self.lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, forget_bias=1.0, state_is_tuple=True)
            self.lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, forget_bias=1.0, state_is_tuple=True)
        # with tf.name_scope('attention'), tf.variable_scope('attention'):
        #     self.u1_w = tf.Variable(tf.truncated_normal([self.hidden_size*2, self.attn_lenth], stddev=0.1), name='attention_w')
        #     self.u1_b = tf.Variable(tf.constant(0.1, shape=[self.attn_lenth]), name='attention_b')
        #     self.u2_w = tf.Variable(tf.truncated_normal([self.attn_lenth, 1], stddev=0.1), name='attention_u')
        with tf.name_scope('lastlayer'), tf.variable_scope('lastlayer'):
            self.sigmoid_weights = tf.Variable(tf.random_uniform(shape=[self.hidden_size*2, 1], minval=-1.0, maxval=1.0), name='sigmoid_w')
            self.sigmoid_biases = tf.Variable(tf.zeros([1]), name='sigmoid_b')


    def embedding_layer(self, inputs):
        with tf.name_scope('embeddings'), tf.variable_scope('embeddings'):
            embeded_outputs = tf.nn.embedding_lookup(self.embeddings, inputs)
            embeded_outputs = tf.nn.dropout(embeded_outputs, keep_prob=self.keep_prob)
        return embeded_outputs


    def bi_attn_lstm_layer(self, inputs):
        # lstm
        with tf.name_scope('lstm'), tf.variable_scope('lstm'):
            drop_lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(cell=self.lstm_fw_cell, output_keep_prob=self.keep_prob)
            drop_lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(cell=self.lstm_bw_cell, output_keep_prob=self.keep_prob)
            rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(drop_lstm_fw_cell, drop_lstm_bw_cell, inputs, dtype=tf.float32)
            rnn_outputs = tf.concat(rnn_outputs, axis=2)
            rnn_outputs = tf.reduce_mean(rnn_outputs, axis=1)
        return rnn_outputs


    def bi_sigmoid_layer(self, inputs):
        logits = tf.matmul(inputs, self.sigmoid_weights) + self.sigmoid_biases
        self.raw_expection = tf.sigmoid(logits)
        self.expection = tf.round(self.raw_expection)
        return self.expection
