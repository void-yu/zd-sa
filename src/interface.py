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

flags.DEFINE_string("ckpt_dir", "../save/bd_ce_w=1_tanh_init_mean_fixed_1", "")
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

def test(corpus):
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
                    model.lenths_weight: padded_ones_list_like(piece_lenths, FLAGS.seq_lenth),
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
                    model.lenths: piece_lenths,
                    model.lenths_weight: padded_ones_list_like(piece_lenths, FLAGS.seq_lenth),
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
                model.lenths: test_lenths,
                model.lenths_weight: padded_ones_list_like(test_lenths, FLAGS.seq_lenth),
            }
            expection, alpha, logits = sess.run([model.expection, model.alpha, model.logits], feed_dict=test_feed_dict)

            print([vocab[i] for i in test_inputs[0]])

            for i in range(len(test_inputs[0])):
                print(vocab[test_inputs[0][i]], alpha[0][i], logits[0][i])


            # print([vocab[word] for word in test_inputs])
            if (expection[0][0] == 1):
                print('负面')
            else:
                print('正面')

            return expection[0]


def get_word_sentiment_polarity(words):
    vocab, word2id = reader.read_glossary()
    words = [[word2id[word]] for word in words if word in vocab]

    with tf.Graph().as_default(), tf.Session() as sess:
        model = wm(
            glossary_size=FLAGS.glossary_size,
            embedding_size=FLAGS.embedding_size,
            hidden_size=FLAGS.hidden_size,
        )
        model.buildGraph()
        saver = tf.train.Saver(tf.trainable_variables())
        sp = []
        if reader.restore_from_checkpoint(sess, saver, FLAGS.ckpt_dir):
            test_feed_dict = {model.inputs: words}
            raw_expection, expection = sess.run([model.raw_expection, model.expection], feed_dict=test_feed_dict)
            for index in range(len(words)):
                temp = [vocab[words[index]][0]]
                print(vocab[words[index]], end='\t')
                print(raw_expection[index][0], end='\t')
                temp.append(raw_expection[index][0])
                if (expection[index][0] == 1):
                    print('负面')
                    temp.append('负面')
                elif (expection[index][0] == 0):
                    print('非负面')
                    temp.append('非负面')
                sp.append(temp)

            return sp




if __name__ == '__main__':
    text = ['康宝莱】河南郑州', '花季少女死于康宝莱']
    corpus = reader.preprocess(
        reader.read_excel('../data/corpus/check/yxcd.xlsx', text_column=1, label_column=0),
        seq_lenth=FLAGS.seq_lenth, seq_num=1, overlap_lenth=0, input_label=True,
        output_index=False, de_duplicated=True)
    # with open('../data/corpus/latest/test.pickle', 'rb') as fp:
    #     corpus = pickle.load(fp)
    test(corpus)
    # predict(input_path='../data/corpus/check/klb.xlsx', output_path='out/x.xlsx')
    # test_onesent('【康宝莱】河南郑州花季少女死于康宝莱传销相关部门难逃其咎： 我叫张云成，男，汉族，现年47岁，家住河南省淮阳县冯塘乡蔡李庄村。2017年6月5日,我的儿子张旭(17岁)从郑州回到家里,向家里要钱,...文字版>> http://t.cn/RonRbWK （新浪长微博>> http://t.cn/zOXAaic）')
    # test_onesent('走进康宝莱之前，我听到了台上的分享嘉宾说了这句话：“也许做康宝莱这件事不是你的梦想，但做好康宝莱能实现你所有的梦想！” 我信了。 果然这是真的。 同时，在这个过程中，帮助更多人获得好身材、健康、财富、和精彩人生，也一并成为了我的梦想！和我其他的梦想一起实现了！——肖珂宇 网页链接')

    # vocab, word2id = reader.read_glossary()
    # sp = get_word_sentiment_polarity(vocab)
    #
    # import pandas as pd
    #
    # df_o = pd.DataFrame(sp)
    # writer = pd.ExcelWriter('../data/corpus/latest/words.xlsx')
    # df_o.to_excel(writer, 'Sheet1')
    # writer.save()


    # get_word_sentiment_polarity(['"', '我', '有', '型', '我', '健康', '"', '^替', '^替', '^替', '^替', '圆满', '落幕', '^填'])
#     get_word_sentiment_polarity(['【', '^替', '】', '河南', '郑州', '^替', '死', '于', '^替', '传销', '相关', '部门', '难逃', '其咎', '：', '我', '叫', '^替', '，', '男', '，', '汉族', '，', '现年', '^数', '岁', '，', '家住', '河南省', '^替', '^替', '乡', '蔡', '^替', '。', '^数', '年', '^数', '月', '^数', '日', ',', '我', '的', '儿子', '张旭', '(', '^数', '岁', ')', '从', '郑州', '回到', '家里', ',', '向', '家里', '要钱', ',', '...', '^替', '^替', '^替', 'http', ':', '/', '/', 't', '.', 'cn', '/', '^替', '（', '新浪', '^替', '^替', '^替', 'http', ':', '/', '/', 't', '.', 'cn', '/', '^替', '）', '^填', '^填', '^填', '^填', '^填', '^填', '^填', '^填', '^填', '^填', '^填', '^填', '^填', '^填', '^填', '^填', '^填'])

