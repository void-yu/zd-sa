import tensorflow as tf
import time
import numpy as np
import pickle

import reader
from test.model_sent import EncoderModel as em_sent

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





def get_batch(corpus):
    batch_size = FLAGS.batch_size
    np.random.shuffle(corpus)
    copy_inputs = []
    copy_lenths = []
    copy_labels = []
    num = 0
    for i in corpus:
        copy_labels.append(0 if i[2] == 'T' else 1)
        copy_inputs.append(i[0])
        copy_lenths.append(i[1])
        num += 1
    i = 0
    while i + batch_size <= num:
        yield copy_inputs[i:i+batch_size], \
              copy_lenths[i:i+batch_size], \
              copy_labels[i:i+batch_size]
        i += batch_size


def get_test_batch(inputs, lenth, labels, num):
    batch_size = FLAGS.batch_size
    count_num = 0

    while(count_num < num):
        batch_inputs = []
        batch_lenth = []
        batch_labels = []
        for j in range(min(batch_size, num - count_num)):
            batch_inputs.append(inputs[count_num])
            batch_lenth.append(lenth[count_num])
            batch_labels.append(labels[count_num])
            count_num += 1
        yield batch_inputs, batch_lenth, batch_labels

def train_sent(sess, corpus, test_corpus):
    print("Read file --")
    start = time.time()

    # id2word, word2id = reader.read_glossary()
    pretrained_wv = reader.read_initw2v()

    ##########################
    test_inputs = []
    test_lenth = []
    test_labels = []
    test_num = 0
    for item in test_corpus:
        test_inputs.append(item[0])
        test_lenth.append(int(item[1]))
        if item[2] in [0, 'T', 0.0]:
            test_labels.append(0)
        elif item[2] in [1, 'F', 1.0]:
            test_labels.append(1)
        test_num += 1
    test_writer = tf.summary.FileWriter(logdir=FLAGS.tensorb_dir + '/test')
    test_P_writer = tf.summary.FileWriter(logdir=FLAGS.tensorb_dir + '/test_P')
    test_R_writer = tf.summary.FileWriter(logdir=FLAGS.tensorb_dir + '/test_R')
    test_F_writer = tf.summary.FileWriter(logdir=FLAGS.tensorb_dir + '/test_F')
    ##########################

    end = time.time()
    print("Read finished -- {:.4f} sec".format(end-start))

    # Build model
    print("Building model --")
    start = end

    model = em_sent(
        seq_size=FLAGS.seq_lenth,
        glossary_size=FLAGS.glossary_size,
        embedding_size=FLAGS.embedding_size,
        hidden_size=FLAGS.hidden_size,
        attn_lenth=FLAGS.attn_lenth
    )
    model.buildTrainGraph()

    init = tf.global_variables_initializer()
    sess.run(init, feed_dict={model.pretrained_wv: pretrained_wv})
    # sess.run(init)

    saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=10)
    train_writer = tf.summary.FileWriter(logdir=FLAGS.tensorb_dir + '/train', graph=sess.graph)

    end = time.time()
    print("Building model finished -- {:.4f} sec".format(end - start))

    # if not reader.restore_from_checkpoint(sess, saver, FLAGS.ckpt_dir):
    #     return
    step_global = 0
    sum_loss = 0
    sum_acc_t = 0

    print("Training initialized")
    start = time.time()

    for epoch in range(FLAGS.epoches):
        for inputs, lenths, labels in get_batch(corpus):
            step_global += 1
            feed_dict = {
                model.inputs: inputs,
                model.lenths: lenths,
                model.labels: labels,
                model.learning_rate: 0.001
            }

            loss, _, t_scalar, t_acc = sess.run([model.loss,
                                                model.optimizer,
                                                model.train_scalar,
                                                model.accuracy],
                                         feed_dict=feed_dict)
            sum_loss += loss
            sum_acc_t += t_acc


            if step_global % FLAGS.save_every_n == 0:
                end = time.time()
                print("Training: Average loss at step {}: {};".format(step_global, sum_loss / FLAGS.save_every_n),
                      "time: {:.4f} sec;".format(end - start),
                      "accuracy rate: {:.4f}".format(sum_acc_t / FLAGS.save_every_n))
                # print("Validation: Average loss: {};".format(sum_dev_loss / FLAGS.save_every_n),
                #       "accuracy rate: {:.4f}".format(sum_acc_d / FLAGS.save_every_n))

                saver.save(sess, FLAGS.ckpt_dir + "/step{}.ckpt".format(step_global))

                ############################################
                total_test_loss = 0
                total_accuracy = 0
                total_expection = []
                for piece_inputs, piece_lenths, piece_labels in get_test_batch(test_inputs, test_lenth, test_labels,
                                                                               test_num):
                    piece_num = len(piece_inputs)
                    test_feed_dict = {
                        model.inputs: piece_inputs,
                        model.lenths: piece_lenths,
                        model.labels: piece_labels
                    }
                    test_loss, accuracy, expection, w2v = sess.run(
                        [model.loss, model.accuracy, model.expection, model.embeddings],
                        feed_dict=test_feed_dict)
                    total_test_loss += test_loss * piece_num
                    total_accuracy += accuracy * piece_num
                    total_expection.extend(expection)

                total_test_loss /= test_num
                total_accuracy /= test_num

                def f_value():
                    # 真正例
                    TP = 0
                    # 假正例
                    FP = 0
                    # 假反例
                    FN = 0
                    # 真反例
                    TN = 0

                    # We pay more attention on negative samples.
                    for i in range(test_num):
                        if test_labels[i] == 0 and total_expection[i] == 0:
                            TP += 1
                        elif test_labels[i] == 0 and total_expection[i] == 1:
                            FN += 1
                        elif test_labels[i] == 1 and total_expection[i] == 0:
                            FP += 1
                        elif test_labels[i] == 1 and total_expection[i] == 1:
                            TN += 1

                    P = TP / (TP + FP + 0.0001)
                    R = TP / (TP + FN + 0.0001)
                    F = 2 * P * R / (P + R + 0.0001)
                    P_ = TN / (TN + FN + 0.0001)
                    R_ = TN / (TN + FP + 0.0001)
                    F_ = 2 * P_ * R_ / (P_ + R_ + 0.0001)
                    # ACC = (TP + TN) / (TP + FP + TN + FN + 0.0001)
                    # print("Validation: Average loss: {};".format(total_test_loss))
                    # print("     accuracy rate: {:.4f}".format(total_accuracy))
                    # print("About positive samples:")
                    # print("     precision rate: {:.4f}".format(P))
                    # print("     recall rate: {:.4f}".format(R))
                    # print("     f-value: {:.4f}".format(F))
                    #
                    # print("About negative samples:")
                    # print("     precision rate: {:.4f}".format(P_))
                    # print("     recall rate: {:.4f}".format(R_))
                    # print("     f-value: {:.4f}".format(F_))

                    t_loss_scalar = tf.Summary(value=[tf.Summary.Value(tag="test_loss", simple_value=total_test_loss)])
                    test_writer.add_summary(t_loss_scalar, step_global)
                    t_ac_scalar = tf.Summary(value=[tf.Summary.Value(tag="test_accuracy", simple_value=total_accuracy)])
                    test_writer.add_summary(t_ac_scalar, step_global)
                    t_P_scalar = tf.Summary(value=[tf.Summary.Value(tag="test_negative", simple_value=P_)])
                    test_P_writer.add_summary(t_P_scalar, step_global)
                    t_R_scalar = tf.Summary(value=[tf.Summary.Value(tag="test_negative", simple_value=R_)])
                    test_R_writer.add_summary(t_R_scalar, step_global)
                    t_F_scalar = tf.Summary(value=[tf.Summary.Value(tag="test_negative", simple_value=F_)])
                    test_F_writer.add_summary(t_F_scalar, step_global)
                f_value()
                ############################################

                train_writer.add_summary(t_scalar, step_global)

                sum_loss = 0
                # sum_dev_loss = 0
                sum_acc_t = 0
                # sum_acc_d = 0

                start = time.time()


def main(_):
    with tf.Graph().as_default(), tf.Session() as sess:
        flags.DEFINE_string("tensorb_dir", "../tensorb/bd_ce_w=1_tanh_init_mean", "")
        flags.DEFINE_string("ckpt_dir", "../save/bd_ce_w=1_tanh_init_mean", "")

        # train_data = '../data/corpus/excel/multi-labeled.xlsx'
        test_data = '../data/corpus/check/klb.xlsx'
        # corpus = reader.preprocess(reader.read_excel(train_data, text_column=1, label_column=0),
        #                            seq_lenth=FLAGS.seq_lenth, seq_num=1, overlap_lenth=0, input_label=True, output_index=False)
        with open('../data/multi-labeled', 'rb') as fp:
            corpus = pickle.load(fp)

        test_corpus = reader.preprocess(reader.read_excel(test_data, text_column=1, label_column=0),
                                   seq_lenth=FLAGS.seq_lenth, seq_num=1, overlap_lenth=0, input_label=True, output_index=False)
        train_sent(sess, corpus, test_corpus)



if __name__ == '__main__':
    tf.app.run()
