import tensorflow as tf
import time
import numpy as np

import reader

flags = tf.app.flags

flags.DEFINE_string("tensorb_dir", "../tensorb", "")
flags.DEFINE_string("ckpt_dir", "../save/test/relu", "")

flags.DEFINE_integer("embedding_size", 400, "")
flags.DEFINE_integer("hidden_size", 300, "")
flags.DEFINE_integer("attn_lenth", 350, "")

'''
    33272(known)+(1876+8)(unknown)=35156
'''
flags.DEFINE_integer("glossary_size", 35156, "")
flags.DEFINE_integer("batch_size", 132, "")
flags.DEFINE_integer("max_seq_size", 120, "")
flags.DEFINE_integer("epoches", 10000, "")

FLAGS = flags.FLAGS


def restore_from_checkpoint(sess, saver, dir):
    ckpt = tf.train.get_checkpoint_state(dir)
    if not ckpt or not ckpt.model_checkpoint_path:
        print('No checkpoint found at {}'.format(dir))
        return False
    saver.restore(sess, ckpt.model_checkpoint_path)
    return True


def main(_):
    glossary_size = 35156
    embedding_size = 400
    attn_lenth = 350

    with tf.Graph().as_default(), tf.Session() as sess:

        with tf.name_scope('embeddings'), tf.variable_scope('embeddings'):
            # embeddings = tf.Variable(pretrained_wv, name='embeddings')
            embeddings = tf.Variable(tf.truncated_normal([glossary_size, embedding_size], stddev=0.1),
                                          name='embeddings')
        with tf.name_scope('attention'), tf.variable_scope('attention'):
            u1_w = tf.Variable(tf.truncated_normal([embedding_size, attn_lenth], stddev=0.1),
                                    name='attention_w')
            u1_b = tf.Variable(tf.constant(0.1, shape=[attn_lenth]), name='attention_b')
            u2_w = tf.Variable(tf.truncated_normal([attn_lenth, 1], stddev=0.1), name='attention_u')

        attned_1 = tf.matmul(tf.nn.relu(tf.matmul(embeddings, u1_w) + u1_b), u2_w)
        # attned_2 = tf.nn.softmax(attned_1, dim=relu)

        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=10)
        if restore_from_checkpoint(sess, saver, FLAGS.ckpt_dir):
            result = [i[0] for i in sess.run(attned_1)]
            glossary, _ = reader.read_glossary()
            g_r = [(k, v) for k, v in zip(glossary, result)]
            g_r = sorted(g_r, key=lambda x: x[1])
            f = open('words.txt', 'w', encoding='utf8')
            for i in range(len(glossary)):
                f.write(g_r[i][0] + '\t' + str(g_r[i][1]) + '\n')
        else:
            print('errrrrror@@')



if __name__ == '__main__':
    tf.app.run()
