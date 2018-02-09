import numpy as np
import pandas as pd
import re
import jieba
import pickle
import tensorflow as tf

import reader

import time



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


def test(filepath):
    corpus = reader.preprocess(reader.read_excel(filepath, text_column=1, label_column=0),
                        seq_lenth=FLAGS.seq_lenth, seq_num=1, overlap_lenth=0, input_label=True, output_index=False)
    # vocab, word2id = reader.read_glossary()

    test_inputs = []
    test_lenths = []
    test_labels = []
    test_num = 0
    for item in corpus:
        test_inputs.append(item[0])
        test_lenths.append(item[1])
        if item[2] in [0, 'T', 0.0]:
            test_labels.append(0)
        elif item[2] in [1, 'F', 1.0]:
            test_labels.append(1)
        test_num += 1

    with tf.Graph().as_default(), tf.Session() as sess:
        model = em_sent(
            # batch_size=FLAGS.batch_size,
            seq_size=FLAGS.seq_lenth,
            glossary_size=FLAGS.glossary_size,
            embedding_size=FLAGS.embedding_size,
            hidden_size=FLAGS.hidden_size,
            attn_lenth=FLAGS.attn_lenth,
            is_training=False
        )
        model.buildTrainGraph()

        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=10)

        if reader.restore_from_checkpoint(sess, saver, FLAGS.ckpt_dir):
            total_test_loss = 0
            total_accuracy = 0
            total_expection = []
            print(test_num)
            for piece_inputs, piece_lenths, piece_labels in \
                    get_test_batch(test_inputs, test_lenths, test_labels, test_num):
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
            # for i in range(test_num):
            #     print(i, [vocab[word] for word in test_inputs[i]])
            #     print(test_inputs[i])
            #     # print(alpha[i])
            #     print(test_labels[i], total_expection[i])

            def f_value():
                # 真正例
                TP = 0
                # 假正例
                FP = 0
                # 假反例
                FN = 0
                # 真反例
                TN = 0

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
                ACC = (TP + TN) / (TP + FP + TN + FN + 0.0001)
                print("Validation: Average loss: {};".format(total_test_loss))
                print("     accuracy rate: {:.4f}".format(total_accuracy))
                print("About positive samples:")
                print("     precision rate: {:.4f}".format(P))
                print("     recall rate: {:.4f}".format(R))
                print("     f-value: {:.4f}".format(F))

                print("About negative samples:")
                print("     precision rate: {:.4f}".format(P_))
                print("     recall rate: {:.4f}".format(R_))
                print("     f-value: {:.4f}".format(F_))

            f_value()
        else:
            print("error!")


def predict(input_path, output_path):
    df_i = pd.read_excel(input_path)
    corpus_i = df_i.iloc[:, [1]]
    corpus_i = np.array(corpus_i).tolist()

    corpus = reader.preprocess(reader.read_excel(input_path, text_column=1),
                        seq_lenth=FLAGS.seq_lenth, seq_num=1, overlap_lenth=0, input_label=False, output_index=True)
    # vocab, word2id = reader.read_glossary()

    test_inputs = []
    test_lenths = []
    test_num = 0
    for item in corpus:
        test_inputs.append(item[0])
        test_lenths.append(item[1])
        test_num += 1

    with tf.Graph().as_default(), tf.Session() as sess:
        model = em_sent(
            seq_size=FLAGS.seq_lenth,
            glossary_size=FLAGS.glossary_size,
            embedding_size=FLAGS.embedding_size,
            hidden_size=FLAGS.hidden_size,
            attn_lenth=FLAGS.attn_lenth,
            is_training=False
        )
        model.buildTrainGraph()

        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=10)

        if reader.restore_from_checkpoint(sess, saver, FLAGS.ckpt_dir):
            total_expection = []
            print(test_num)
            for piece_inputs, piece_lenths in get_test_batch(test_inputs, test_lenths, None, test_num, input_label=False):
                test_feed_dict = {
                    model.inputs: piece_inputs,
                    model.lenths: piece_lenths
                }
                expection = sess.run(model.expection, feed_dict=test_feed_dict)
                total_expection.extend(expection)

            zipped = []
            for index in range(test_num):
                zipped.append([corpus_i[corpus[index][2]], 'T' if total_expection[index][0] == 0 else 'F'])
            df_o = pd.DataFrame(zipped)
            writer = pd.ExcelWriter(output_path)
            df_o.to_excel(writer, 'Sheet1')
            writer.save()


def test_onesent(text):
    corpus = reader.preprocess([[text]],
                        seq_lenth=FLAGS.seq_lenth, seq_num=1, overlap_lenth=0, input_label=False, output_index=False)
    vocab, word2id = reader.read_glossary()

    print(corpus)
    test_inputs = []
    test_lenths = []
    test_num = 0
    for item in corpus:
        test_inputs.append(item[0])
        test_lenths.append(item[1])
        test_num += 1

    with tf.Graph().as_default(), tf.Session() as sess:
        model = em_sent(
            seq_size=FLAGS.seq_lenth,
            glossary_size=FLAGS.glossary_size,
            embedding_size=FLAGS.embedding_size,
            hidden_size=FLAGS.hidden_size,
            attn_lenth=FLAGS.attn_lenth,
            is_training=False
        )
        model.buildTrainGraph()

        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=10)

        if reader.restore_from_checkpoint(sess, saver, FLAGS.ckpt_dir):
            test_feed_dict = {
                model.inputs: test_inputs,
                model.lenths: test_lenths
            }
            expection, alpha = sess.run([model.expection, model.alpha], feed_dict=test_feed_dict)
            print([vocab[word] for word in test_inputs])
            print(alpha)
            if (expection[0][0] == 1):
                print('负面')
            else:
                print('正面')



if __name__ == '__main__':
    from model_sent import EncoderModel as em_sent

    flags = tf.app.flags

    flags.DEFINE_string("ckpt_dir", "model", "")
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

    # test('data/yxcd.xlsx')
    # predict(input_path='data/klb.xlsx', output_path='out/x.xlsx')
    test_onesent('发表了博文《康宝莱是传销》自己家人有做的，可我不愿意违背自己良心去宣传。确实很坑人~以下为好多被坑的人们的心声~希望大家不要为了急于追求苗条的身材去花冤枉钱了~实话告诉你都是骗人的，因为他们是金字塔模http://t.cn/RK2YJ8t')


