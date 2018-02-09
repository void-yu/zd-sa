import tensorflow as tf


"""
    bi_attn_lstm model
"""

class EncoderModel(object):

    def __init__(self,
                 seq_size,
                 glossary_size,
                 embedding_size,
                 hidden_size,
                 attn_lenth,
                 is_training=True):
        self.seq_size = seq_size
        self.glossary_size = glossary_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.attn_lenth = attn_lenth
        self.is_training = is_training


    def buildTrainGraph(self):
        inputs, lenths, labels = self.define_IO()
        self.trainable_parameters()
        inputs = self.embedding_layer(inputs)
        outputs = self.bi_attn_lstm_layer(inputs, lenths)
        outputs = self.bi_sigmoid_layer(outputs)
        self.loss_and_optimize(outputs, labels)


    def define_IO(self):
        self.inputs = tf.placeholder(tf.int32, shape=[None, self.seq_size], name='inputs')
        self.labels = tf.placeholder(tf.float32, shape=[None], name='labels')
        self.lenths = tf.placeholder(tf.int32, shape=[None], name='lenths')
        self.pretrained_wv = tf.placeholder(tf.float32, shape=[self.glossary_size, self.embedding_size])
        return self.inputs, self.lenths, self.labels


    def trainable_parameters(self):
        if self.is_training is True:
            self.keep_prob = 0.5
        else:
            self.keep_prob = 1

        with tf.name_scope('embeddings'), tf.variable_scope('embeddings'):
            self.embeddings = tf.Variable(self.pretrained_wv, name='embeddings')
            # self.embeddings = tf.Variable(tf.truncated_normal([self.glossary_size, self.embedding_size], stddev=0.1), name='embeddings')
        with tf.name_scope('lstm'), tf.variable_scope('lstm'):
            self.lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, forget_bias=1.0, state_is_tuple=True)
            self.lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, forget_bias=1.0, state_is_tuple=True)
        with tf.name_scope('attention'), tf.variable_scope('attention'):
            self.u1_w = tf.Variable(tf.truncated_normal([self.hidden_size*2, self.attn_lenth], stddev=0.1), name='attention_w')
            self.u1_b = tf.Variable(tf.constant(0.1, shape=[self.attn_lenth]), name='attention_b')
            self.u2_w = tf.Variable(tf.truncated_normal([self.attn_lenth, 1], stddev=0.1), name='attention_u')
        with tf.name_scope('lastlayer'), tf.variable_scope('lastlayer'):
            self.sigmoid_weights = tf.Variable(tf.random_uniform(shape=[self.hidden_size*2, 1], minval=-1.0, maxval=1.0), name='sigmoid_w')
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
            outputs - shape=[batch_size, hidden_size*2]
    """
    def bi_attn_lstm_layer(self, inputs, lenths):
        # lstm
        with tf.name_scope('lstm'), tf.variable_scope('lstm'):
            drop_lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(cell=self.lstm_fw_cell, output_keep_prob=self.keep_prob)
            drop_lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(cell=self.lstm_bw_cell, output_keep_prob=self.keep_prob)

            rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(drop_lstm_fw_cell, drop_lstm_bw_cell, inputs, sequence_length=lenths, dtype=tf.float32)
            rnn_outputs = tf.concat(rnn_outputs, axis=2)

        # attention
        with tf.name_scope('attention'), tf.variable_scope('attention'):
            alpha = tf.reshape(rnn_outputs, [-1, self.hidden_size*2])
            alpha = tf.matmul(tf.nn.tanh(tf.matmul(alpha, self.u1_w) + self.u1_b), self.u2_w)
            alpha = tf.reshape(alpha, [-1, self.seq_size])
            alpha = tf.nn.softmax(alpha)
            self.alpha = alpha
            alpha = tf.reshape(alpha, [-1, self.seq_size, 1])
            attn_outputs = tf.reduce_mean(rnn_outputs * alpha, axis=1)
        return attn_outputs


    """
        arg:
            inputs - shape=[batch_size, hidden_size]
        return:
            outputs - shape=[batch_size, relu]
    """
    def bi_sigmoid_layer(self, inputs):
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

        labels = tf.reshape(labels, [-1, 1])
        # loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=inputs, labels=labels)
        loss = tf.nn.weighted_cross_entropy_with_logits(logits=inputs, targets=labels, pos_weight=1)
        self.loss = tf.reduce_mean(loss)

        train_vars = tf.trainable_variables()
        self.max_grad_norm = 1
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, train_vars), self.max_grad_norm)

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

        self.train_scalar = tf.summary.scalar('train_loss', self.loss)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(tf.sigmoid(inputs)), labels), tf.float32))


        self.expection = tf.round(tf.sigmoid(inputs))


    # def optimize(self,
    #              loss,
    #              global_step,
    #              max_grad_norm,
    #              lr,
    #              lr_decay,
    #              sync_replicas=False,
    #              replicas_to_aggregate=relu,
    #              task_id=0):
    #     """Builds optimization graph.
    #
    #     * Creates an optimizer, and optionally wraps with SyncReplicasOptimizer
    #     * Computes, clips, and applies gradients
    #     * Maintains moving averages for all trainable variables
    #     * Summarizes variables and gradients
    #
    #     Args:
    #       loss: scalar loss to minimize.
    #       global_step: integer scalar Variable.
    #       max_grad_norm: float scalar. Grads will be clipped to this value.
    #       lr: float scalar, learning rate.
    #       lr_decay: float scalar, learning rate decay rate.
    #       sync_replicas: bool, whether to use SyncReplicasOptimizer.
    #       replicas_to_aggregate: int, number of replicas to aggregate when using
    #         SyncReplicasOptimizer.
    #       task_id: int, id of the current task; used to ensure proper initialization
    #         of SyncReplicasOptimizer.
    #
    #     Returns:
    #       train_op
    #     """
    #     with tf.name_scope('optimization'):
    #         # Compute gradients.
    #         tvars = tf.trainable_variables()
    #         grads = tf.gradients(
    #             loss,
    #             tvars,
    #             aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
    #
    #         # Clip non-embedding grads
    #         non_embedding_grads_and_vars = [(g, v) for (g, v) in zip(grads, tvars)
    #                                         if 'embedding' not in v.op.name]
    #         embedding_grads_and_vars = [(g, v) for (g, v) in zip(grads, tvars)
    #                                     if 'embedding' in v.op.name]
    #
    #         ne_grads, ne_vars = zip(*non_embedding_grads_and_vars)
    #         ne_grads, _ = tf.clip_by_global_norm(ne_grads, max_grad_norm)
    #         non_embedding_grads_and_vars = zip(ne_grads, ne_vars)
    #
    #         grads_and_vars = embedding_grads_and_vars + non_embedding_grads_and_vars
    #
    #         # Summarize
    #         # _summarize_vars_and_grads(grads_and_vars)
    #
    #         # Decaying learning rate
    #         lr = tf.train.exponential_decay(
    #             lr, global_step, relu, lr_decay, staircase=True)
    #         tf.summary.scalar('learning_rate', lr)
    #         opt = tf.train.AdamOptimizer(lr)
    #
    #         # Track the moving averages of all trainable variables.
    #         variable_averages = tf.train.ExponentialMovingAverage(0.999, global_step)
    #
    #         # Apply gradients
    #         # Non-sync optimizer
    #         variables_averages_op = variable_averages.apply(tvars)
    #         apply_gradient_op = opt.apply_gradients(grads_and_vars, global_step)
    #         with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    #             train_op = tf.no_op(name='train_op')
    #
    #         return train_op

