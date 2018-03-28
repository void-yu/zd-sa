import numpy as np
import pandas as pd
import re
import jieba
import pickle
import tensorflow as tf

import reader

from test.model_sent import EncoderModel as em_sent
from test.word_sentiment import WordModel as wm

import time

flags = tf.app.flags

flags.DEFINE_string("ckpt_dir", "../save/bd_ce_w=1_tanh_init_mean_fixed", "")
flags.DEFINE_integer("save_every_n", 10, "")

flags.DEFINE_integer("embedding_size", 400, "")
flags.DEFINE_integer("hidden_size", 300, "")
flags.DEFINE_integer("attn_lenth", 350, "")

'''
    33272(known)+(1876+8)(unknown)=35156
'''
flags.DEFINE_integer("glossary_size", 35156, "")
flags.DEFINE_integer("batch_size", 128, "")
flags.DEFINE_integer("seq_lenth", 150, "")
flags.DEFINE_integer("epoches", 100, "")

FLAGS = flags.FLAGS

SESS = tf.Session()
model = em_sent(
    seq_size=FLAGS.seq_lenth,
    glossary_size=FLAGS.glossary_size,
    embedding_size=FLAGS.embedding_size,
    hidden_size=FLAGS.hidden_size,
    attn_lenth=FLAGS.attn_lenth,
    is_training=False
)
model.buildTrainGraph()


def get_test_batch(inputs, lenths, labels, num, input_label=True):
    batch_size = FLAGS.batch_size
    count_num = 0

    if input_label is True:
        while(count_num < num):
            batch_inputs = []
            batch_lenths = []
            batch_labels = []
            for j in range(min(batch_size, num - count_num)):
                batch_inputs.append(inputs[count_num])
                batch_lenths.append(lenths[count_num])
                batch_labels.append(labels[count_num])
                count_num += 1
            yield batch_inputs, batch_lenths, batch_labels
    else:
        while(count_num < num):
            batch_inputs = []
            batch_lenths = []
            for j in range(min(batch_size, num - count_num)):
                batch_inputs.append(inputs[count_num])
                batch_lenths.append(lenths[count_num])
                count_num += 1
            yield batch_inputs, batch_lenths

def padded_ones_list_like(lenths, max_lenth):
    o = [[1.0] * i for i in lenths]
    for i in o:
        i.extend([0.0]*(max_lenth - len(i)))
    return o

"""takes a list of d strings and outputs a (d, k) numpy array with prediction probabilities, 
    where k is the number of classes."""
def test_for_lime(text):
    corpus = reader.preprocess([[i] for i in text],
                               seq_lenth=FLAGS.seq_lenth,
                               seq_num=1,
                               overlap_lenth=0,
                               input_label=False,
                               output_index=False,
                               split_func=lambda x: x.split(' '),
                               de_duplicated=False)
    # vocab, word2id = reader.read_glossary()

    test_inputs = []
    test_lenths = []
    test_num = 0
    for item in corpus:
        test_inputs.append(item[0])
        test_lenths.append(item[1])
        test_num += 1

    saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=10)

    if reader.restore_from_checkpoint(SESS, saver, FLAGS.ckpt_dir):
        total_expection = []
        for piece_inputs, piece_lenths in get_test_batch(test_inputs, test_lenths, None, test_num,
                                                         input_label=False):
            test_feed_dict = {
                model.inputs: piece_inputs,
                model.lenths: piece_lenths,
                model.lenths_weight: padded_ones_list_like(piece_lenths, FLAGS.seq_lenth),
            }
            expection = SESS.run(model.raw_expection, feed_dict=test_feed_dict)
            total_expection.extend([[round(i[0], 4), round(1-i[0], 4)] for i in expection])
        return np.array(total_expection)



