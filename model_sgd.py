import tensorflow as tf
import numpy as np


"""
    bi_attn_lstm model
"""

class EncoderModel(object):

    def __init__(self,
                 max_seq_size,
                 glossary_size,
                 embedding_size,
                 hidden_size,
                 attn_lenth,
                 learning_rate):
        self.max_seq_size = max_seq_size
        self.glossary_size = glossary_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.attn_lenth = attn_lenth
        self.learning_rate = learning_rate


    def defineIO(self):
        self.title_input = tf.placeholder(tf.int32, shape=[1, self.max_seq_size])
        self.title_lenth = tf.placeholder(tf.int32, shape=[1])
        self.text_inputs = tf.placeholder(tf.int32, shape=[None, self.max_seq_size])
        self.text_lenths = tf.placeholder(tf.int32, shape=[None])
        self.label = tf.placeholder(tf.float32, shape=[1])
        return self.title_input, self.text_inputs, self.title_lenth, self.text_lenths, self.label

    # def loadPreTrainedParameters(self):
    #     self.pretrained_embeddings = tf.placeholder(tf.float32, shape=[self.glossary_size, self.embedding_size])
    #     self.pretrained_attn_w = tf.placeholder(tf.float32, shape=[2*self.hidden_size, self.attn_lenth])
    #     self.pretrained_attn_b = tf.placeholder(tf.float32, shape=[self.attn_lenth])
    #     self.pretrained_attn_u = tf.placeholder(tf.float32, shape=[self.attn_lenth, 1])



    def trainableParameters(self):
        self.keep_prob = 0.5
        with tf.name_scope('embeddings'), tf.variable_scope('embeddings', reuse=True):
            # self.embeddings = tf.Variable(self.pretrained_embeddings, name='embeddings')
            self.embeddings = tf.Variable(tf.truncated_normal([self.glossary_size, self.embedding_size], stddev=0.1), name='embeddings')

        with tf.name_scope('rnn'), tf.variable_scope('rnn', reuse=True):
            self.lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, forget_bias=1.0, state_is_tuple=True)
            self.lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, forget_bias=1.0, state_is_tuple=True)

        with tf.name_scope('attention'), tf.variable_scope('attention', reuse=True):
            self.attn_w = tf.Variable(tf.truncated_normal([2*self.hidden_size, self.attn_lenth], stddev=0.1), name='attention_w')
            self.attn_b = tf.Variable(tf.constant(0.1, shape=[self.attn_lenth]), name='attention_b')
            self.attn_u = tf.Variable(tf.truncated_normal([self.attn_lenth, 1], stddev=0.1), name='attention_u')
        with tf.name_scope('integrated'), tf.variable_scope('integrated', reuse=True):
            self.inte_attn_w = tf.Variable(tf.truncated_normal([2*self.hidden_size, self.attn_lenth], stddev=0.1), name='integrated_attenedsum_w')
            self.inte_attn_b = tf.Variable(tf.constant(0.1, shape=[self.attn_lenth]), name='integrated_attenedsum_b')
            self.inte_attn_u = tf.Variable(tf.truncated_normal([self.attn_lenth, 1], stddev=0.1), name='integrated_attenedsum_u')
        with tf.name_scope('merge'), tf.variable_scope('merge', reuse=True):
            self.merge_inde_w = tf.Variable(tf.truncated_normal([2*self.hidden_size, 1], stddev=0.1), name='merge_independent_w')
            self.merge_inde_b = tf.Variable(tf.constant(0.1, shape=[1]), name='merge_independent_b')
            self.merge_inte_w = tf.Variable(tf.truncated_normal([2*self.hidden_size, 1], stddev=0.1), name='merge_integrated_w')
            self.merge_inte_b = tf.Variable(tf.constant(0.1, shape=[1]), name='merge_integrated_b')




    def embeddingLayer(self, inputs):
        with tf.name_scope('embeddings'), tf.variable_scope('embeddings', reuse=True):
            embeded_outputs = tf.nn.embedding_lookup(self.embeddings, inputs)
            embeded_outputs = tf.nn.dropout(embeded_outputs, keep_prob=self.keep_prob)
        return embeded_outputs


    def rnnComponent(self, inputs, lenths):
        drop_lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(cell=self.lstm_fw_cell, output_keep_prob=self.keep_prob)
        drop_lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(cell=self.lstm_bw_cell, output_keep_prob=self.keep_prob)

        rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(drop_lstm_fw_cell, drop_lstm_bw_cell, inputs,
                                                     sequence_length=lenths, dtype=tf.float32)
        rnn_outputs = tf.concat(rnn_outputs, axis=2)


        rnn_outputs = tf.reshape(rnn_outputs, [-1, 2*self.hidden_size])

        rnn_alpha = tf.matmul((tf.matmul(rnn_outputs, self.attn_w) + self.attn_b), self.attn_u)
        rnn_alpha = tf.reshape(rnn_alpha, [-1, self.max_seq_size, 1])
        rnn_alpha = tf.nn.softmax(rnn_alpha, dim=1)

        rnn_outputs = tf.reshape(rnn_outputs, [-1, self.max_seq_size, 2*self.hidden_size])
        rnn_outputs = tf.reduce_sum(rnn_outputs*rnn_alpha, axis=1)
        return rnn_outputs


    def integratedSumLayer(self, inputs):
        inte_sum_outputs = tf.reduce_sum(inputs, axis=0)
        inte_sum_outputs = tf.reshape(inte_sum_outputs, [1, -1])
        return inte_sum_outputs


    def integratedAttsumLayer(self, inputs):
        rnn_outputs = tf.reshape(inputs, [-1, 2*self.hidden_size])
        inte_attsum_alpha = tf.matmul((tf.matmul(rnn_outputs, self.inte_attn_w) + self.inte_attn_b), self.inte_attn_u)
        inte_attsum_alpha = tf.nn.softmax(inte_attsum_alpha, dim=0)
        inte_attsum_outputs = tf.reduce_sum(rnn_outputs * inte_attsum_alpha, axis=0)
        return inte_attsum_outputs


    def mergeLayer(self, inputs_inde, inputs_inte):
        inde_outputs = tf.matmul(inputs_inde, self.merge_inde_w) + self.merge_inde_b
        inte_outputs = tf.matmul(inputs_inte, self.merge_inte_w) + self.merge_inte_b
        merge_outputs = tf.reshape(inde_outputs+inte_outputs, [1])
        return merge_outputs


    def lossAndOptimizer(self, inputs, labels):
        self._logits = tf.round(tf.sigmoid(inputs))
        self._labels = labels
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=inputs, labels=labels)
        self.loss = loss

        train_vars = tf.trainable_variables()
        self.max_grad_norm = 1
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, train_vars), self.max_grad_norm)

        # self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).apply_gradients(zip(grads, train_vars))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

        self.train_scalar = tf.summary.scalar('train_loss', self.loss[0])
        self.train_accuracy = tf.cast(tf.equal(self._logits, labels), tf.float32)



    def buildTrainGraph(self):
        title_input, text_inputs, title_lenth, text_lenths, label = self.defineIO()
        self.trainableParameters()
        title_outputs = self.rnnComponent(self.embeddingLayer(title_input), title_lenth)
        text_outputs = self.integratedSumLayer(self.rnnComponent(self.embeddingLayer(text_inputs), text_lenths))
        merge_outputs = self.mergeLayer(title_outputs, text_outputs)
        self.lossAndOptimizer(merge_outputs, self.label)

