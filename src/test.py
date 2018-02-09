import tensorflow as tf
import pickle
import numpy as np

import reader

flags = tf.app.flags


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


# train_data = '../data/corpus/excel/multi-labeled.xlsx'
# corpus = reader.preprocess(reader.read_excel(train_data, text_column=1, label_column=0),
#                                    seq_lenth=FLAGS.seq_lenth, seq_num=1, overlap_lenth=0, input_label=True, output_index=False)
# with open('../data/multi-labeled', 'wb') as fp:
#     pickle.dump(corpus, fp)
with open('../data/multi-labeled', 'rb') as fp:
    corpus = pickle.load(fp)
    id2word, word2id = reader.read_glossary()
for i in range(100):
    print([id2word[j] for j in corpus[i][0]])