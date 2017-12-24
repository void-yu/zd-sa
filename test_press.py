import tensorflow as tf
import reader
import numpy as np
import time

class EncoderModel(object):

    def __init__(self,
                 batch_size,
                 glossary_size,
                 embedding_size,
                 hidden_size,
                 attn_lenth):
        self.batch_size = batch_size
        self.glossary_size = glossary_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.attn_lenth = attn_lenth


    def build_train_graph(self):
        inputs, lenth, labels = self.define_IO()
        self.trainable_parameters()
        inputs = self.embedding_layer(inputs)
        outputs = self.bi_attn_lstm_layer(inputs, lenth)
        outputs = self.bi_sigmoid_layer(outputs)
        self.loss_and_optimize(outputs, labels)


    def define_IO(self):
        self.inputs = tf.placeholder(tf.int32, shape=[self.batch_size, None], name='inputs')
        self.labels = tf.placeholder(tf.float32, shape=[self.batch_size, 1], name='targets')
        self.lenth = tf.placeholder(tf.int32, shape=[self.batch_size], name='lenth')
        self.pretrained_wv = tf.placeholder(tf.float32, shape=[self.glossary_size, self.embedding_size])
        return self.inputs, self.lenth, self.labels


    def trainable_parameters(self):
        self.keep_prob = 0.5
        with tf.name_scope('embeddings'), tf.variable_scope('embeddings'):
            # self.embeddings = tf.Variable(self.pretrained_wv, name='embeddings')
            self.embeddings = tf.Variable(tf.truncated_normal([self.glossary_size, self.embedding_size], stddev=0.1), name='embeddings')
        with tf.name_scope('lstm'), tf.variable_scope('lstm'):
            self.lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, forget_bias=1.0, state_is_tuple=True)
            self.lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, forget_bias=1.0, state_is_tuple=True)
        with tf.name_scope('attention'), tf.variable_scope('attention'):
            self.u1_w = tf.Variable(tf.truncated_normal([2*self.hidden_size, self.attn_lenth], stddev=0.1), name='attention_w')
            self.u1_b = tf.Variable(tf.constant(0.1, shape=[self.attn_lenth]), name='attention_b')
            self.u2_w = tf.Variable(tf.truncated_normal([self.attn_lenth, 1], stddev=0.1), name='attention_uw')
        with tf.name_scope('lastlayer'), tf.variable_scope('lastlayer'):
            self.sigmoid_weights = tf.Variable(tf.random_uniform(shape=[self.hidden_size * 2, 1], minval=-1.0, maxval=1.0), name='sigmoid_w')
            self.sigmoid_biases = tf.Variable(tf.zeros([1]), name='sigmoid_b')


    """
        arg:
            inputs - shape=[batch_size, seq_size]
        return:
            outputs - shape=[batch_size, seq_size, hidden_size]
    """
    def embedding_layer(self, inputs):
        with tf.name_scope('embeddings'), tf.variable_scope('embeddings'):
            embeded_outputs = tf.nn.embedding_lookup(self.embeddings, inputs)
            embeded_outputs = tf.nn.dropout(embeded_outputs, keep_prob=self.keep_prob)
        return embeded_outputs


    """
        arg:
            inputs - shape=[batch_size, seq_size, hidden_size]
        return:
            outputs - shape=[batch_size, seq_size, hidden_size*2]
    """
    def bi_attn_lstm_layer(self, inputs, lenth):
        with tf.name_scope('lstm'), tf.variable_scope('lstm'):
            drop_lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(cell=self.lstm_fw_cell, output_keep_prob=self.keep_prob)
            drop_lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(cell=self.lstm_bw_cell, output_keep_prob=self.keep_prob)

            rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(drop_lstm_fw_cell, drop_lstm_bw_cell, inputs, sequence_length=lenth, dtype=tf.float32)
            rnn_outputs = tf.concat(rnn_outputs, axis=2)

        # with tf.name_scope('attention'), tf.variable_scope('attention'):
        #     attn_outputs = []
        #     for i in range(self.batch_size):
        #         attn_output = rnn_outputs[i][:lenth[i]]
        #         attn_z = tf.matmul(tf.tanh(tf.matmul(attn_output, self.u1_w) + self.u1_b), self.u2_w)
        #         alpha = tf.nn.softmax(attn_z)
        #         attn_output = tf.reduce_sum(attn_output * alpha, axis=0)
        #         attn_outputs.append(attn_output)
        #     attn_outputs = tf.convert_to_tensor(attn_outputs)
        # return attn_outputs
        return rnn_outputs[:, -1, :]


    """
        arg:
            inputs - shape=[batch_size, hidden_size]
        return:
            outputs - shape=[batch_size, 1]
    """
    def bi_sigmoid_layer(self, inputs):
        with tf.name_scope('lastlayer'), tf.variable_scope('lastlayer'):
            logits = tf.matmul(inputs, self.sigmoid_weights) + self.sigmoid_biases
        return logits


    """
        arg:
            inputs - shape=[batch_size, seq_size, glossary_size]
            labels - shape=[batch_size, seq_size]
        return:
            outputs - shape=[batch_size, seq_size, hidden_size]
    """
    def loss_and_optimize(self, inputs, labels):
        self.learning_rate = tf.placeholder(tf.float32)

        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=inputs, labels=labels)
        self.loss = tf.reduce_mean(loss)

        train_vars = tf.trainable_variables()
        self.max_grad_norm = 1
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, train_vars), self.max_grad_norm)

        # self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).apply_gradients(zip(grads, train_vars))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

        self.train_scalar = tf.summary.scalar('train_loss', self.loss)
        self.train_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(tf.sigmoid(inputs)), labels), tf.float32))




def main():
    batch_size = 8
    with tf.Graph().as_default(), tf.Session() as sess:
        model = EncoderModel(
            batch_size=batch_size,
            glossary_size=30000,
            embedding_size=300,
            hidden_size=300,
            attn_lenth=350
        )
        model.build_train_graph()

        init = tf.global_variables_initializer()
        sess.run(init)
        train_corpus, valid_corpus, _ = reader.read_corpus(index='2', pick_test=False)
        train_inputs = []
        train_lenth = []
        train_labels = []
        train_num = 0
        for item in train_corpus:
            train_inputs.append(item[0][0])
            train_lenth.append(int(item[0][1]))
            train_labels.append(1 if item[1] is 'T' else 0)
            train_num += 1
        train_labels = np.reshape(train_labels, [-1, 1])
        feed_dict = {
            model.inputs: train_inputs[:batch_size],
            model.lenth: train_lenth[:batch_size],
            model.labels: train_labels[:batch_size],
            model.learning_rate: 0.01
        }
        for i in range(100):
            start = time.time()
            loss, _, t_scalar, t_acc = sess.run([model.loss,
                                                 model.optimizer,
                                                 model.train_scalar,
                                                 model.train_accuracy],
                                                feed_dict=feed_dict)
            end = time.time()
            print(loss, end - start)
main()