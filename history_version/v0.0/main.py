import time

import numpy as np
import reader
import tensorflow as tf
from model import EncoderModel

flags = tf.app.flags

flags.DEFINE_string("tensorboard_dir", "tensorb/test",
                    "Directory to write the model and training summaries.")
flags.DEFINE_string("ckpt_dir", "save/test",
                    "Directory to write the model and training summaries.")
flags.DEFINE_integer("save_every_n", 10,
                     "Save all parameters every n iterations")

flags.DEFINE_integer("embedding_size", 400,
                     "The embedding layer size.")
flags.DEFINE_integer("hidden_size", 300,
                     "The hidden layer size.")
flags.DEFINE_integer("attn_lenth", 350,
                     "The size of an attention window.")

'''
    33272(known)+(1876+8)(unknown)=35156
'''
flags.DEFINE_integer("glossary_size", 35156,
                     "The glossary dimension size.")
flags.DEFINE_integer("batch_size", 128,
                     "Number of sentences per batch. And each time we feed the whole sentence.")
flags.DEFINE_integer("seq_size", 60, "")
flags.DEFINE_integer("epochs_batch", 2000,
                     "Number of epochs processed. I set only One batch is processed in One epoch.")
flags.DEFINE_integer("max_batch_size", 1024, "")

FLAGS = flags.FLAGS



def get_batches(inputs, lenth, labels, num):
    batch_size = FLAGS.batch_size
    count_num = 0

    for i in range(FLAGS.epochs_batch):
        batch_inputs = []
        batch_lenth = []
        batch_labels = []
        for j in range(batch_size):
            count_num = count_num % num
            batch_inputs.append(inputs[count_num])
            batch_lenth.append(lenth[count_num])
            batch_labels.append(labels[count_num])
            count_num += 1
        yield batch_inputs, batch_lenth, batch_labels

def get_splited_batches(inputs, lenth, num):
    batch_size = FLAGS.max_batch_size
    count_num = 0

    while(count_num < num):
        batch_inputs = []
        batch_lenth = []
        for j in range(min(batch_size, num - count_num)):
            batch_inputs.append(inputs[count_num])
            batch_lenth.append(lenth[count_num])
            count_num += 1
        yield batch_inputs, batch_lenth

def get_test_batches(inputs, lenth, labels, num):
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

def restore_from_checkpoint(sess, saver, dir):
    ckpt = tf.train.get_checkpoint_state(dir)
    if not ckpt or not ckpt.model_checkpoint_path:
        print('No checkpoint found at {}'.format(dir))
        return False
    saver.restore(sess, ckpt.model_checkpoint_path)
    return True



def train(sess):

    # Pretreatment
    print("Read file --")
    start = time.time()

    id2word, word2id = reader.read_glossary()
    train_corpus, valid_corpus, _ = reader.read_corpus(index='0', pick_test=False)
    pretrained_wv = reader.read_initw2v()

    train_inputs = []
    train_lenth = []
    train_labels = []
    train_num = 0
    for item in train_corpus:
        train_inputs.append(item[1])
        train_lenth.append(int(item[2]))
        train_labels.append(1 if item[0] is 'T' else 0)
        train_num += 1

    valid_inputs = []
    valid_lenth = []
    valid_labels = []
    valid_num = 0
    for item in valid_corpus:
        valid_inputs.append(item[1])
        valid_lenth.append(int(item[2]))
        valid_labels.append(1 if item[0] is 'T' else 0)
        valid_num += 1
    end = time.time()
    print("Read finished -- {:.4f} sec".format(end-start))

    # Build model
    print("Building model --")
    start = end

    model = EncoderModel(
        batch_size=FLAGS.batch_size,
        glossary_size=FLAGS.glossary_size,
        embedding_size=FLAGS.embedding_size,
        hidden_size=FLAGS.hidden_size,
        attn_lenth=FLAGS.attn_lenth
    )
    model.build_train_graph()
    model.build_validate_graph(valid_num)

    init = tf.global_variables_initializer()
    sess.run(init, feed_dict={model.pretrained_wv: pretrained_wv})
    # sess.run(init)
    saver = tf.train.Saver(max_to_keep=10)
    train_writer = tf.summary.FileWriter(logdir=FLAGS.tensorboard_dir, graph=sess.graph)

    end = time.time()
    print("Building model finished -- {:.4f} sec".format(end - start))

    # if not restore_from_checkpoint(sess, saver, FLAGS.ckpt_dir):
    #     return
    step_global = 0
    sum_loss = 0
    sum_dev_loss = 0
    sum_acc_t = 0
    sum_acc_d = 0
    # max_acc = 0
    lr = 0.001
    valid_labels = np.reshape(valid_labels, [valid_num, 1])
    dev_feed_dict = {
        model.dev_inputs: valid_inputs,
        model.dev_lenth: valid_lenth,
        model.dev_labels: valid_labels
    }

    print("Training initialized")
    start = time.time()

    for inputs, lenth, labels in get_batches(train_inputs, train_lenth, train_labels, train_num):
        step_global += 1
        labels = np.reshape(labels, [FLAGS.batch_size, 1])
        feed_dict = {
            model.inputs: inputs,
            model.lenth: lenth,
            model.labels: labels,
            model.learning_rate: lr
        }
        loss, _, t_scalar, t_acc = sess.run([model.loss,
                                            model.optimizer,
                                            model.train_scalar,
                                            model.train_accuracy],
                                     feed_dict=feed_dict)
        dev_loss, d_scalar, d_acc, w2v = sess.run([model.dev_loss,
                                                   model.dev_scalar,
                                                   model.dev_accuracy,
                                                   model.embeddings],
                                                  feed_dict=dev_feed_dict)

        sum_loss += loss
        sum_dev_loss += dev_loss
        sum_acc_t += t_acc
        sum_acc_d += d_acc

        def eval_ws(ws_list):
            from scipy import stats
            from numpy import linalg as LA

            logits = []
            real = []
            eval = []

            for iter_ws in ws_list:
                if iter_ws[0] not in id2word or iter_ws[1] not in id2word:
                    continue
                else:
                    A = word2id[iter_ws[0]]
                    B = word2id[iter_ws[1]]
                    real.append(iter_ws[2])
                    logits.extend([w2v[A], w2v[B]])

            for i in range(len(logits) // 2):
                A_vec = logits[2 * i]
                B_vec = logits[2 * i + 1]
                normed_A_vec = LA.norm(A_vec, axis=0)
                normed_B_vec = LA.norm(B_vec, axis=0)
                sim = sum(np.multiply(A_vec, B_vec))
                eval.append(sim / normed_A_vec / normed_B_vec)

            pearsonr = stats.pearsonr(real, eval)[0]
            spearmanr = stats.spearmanr(real, eval).correlation
            return pearsonr, spearmanr


        if step_global % FLAGS.save_every_n == 0:
            end = time.time()
            print("Training: Average loss at step {}: {};".format(step_global, sum_loss / FLAGS.save_every_n),
                  "time: {:.4f} sec;".format(end - start),
                  "accuracy rate: {:.4f}".format(sum_acc_t / FLAGS.save_every_n))
            print("Validation: Average loss: {};".format(sum_dev_loss / FLAGS.save_every_n),
                  "accuracy rate: {:.4f}".format(sum_acc_d / FLAGS.save_every_n))

            saver.save(sess, FLAGS.ckpt_dir + "/step{}.ckpt".format(step_global))

            train_writer.add_summary(t_scalar, step_global)
            train_writer.add_summary(d_scalar, step_global)
            ac_scalar = tf.Summary(value=[tf.Summary.Value(tag="accuracy rate", simple_value=sum_acc_d / FLAGS.save_every_n)])
            train_writer.add_summary(ac_scalar, step_global)
            # p_240, s_240 = eval_ws(reader.read_wordsim240())
            # p_297, s_297 = eval_ws(reader.read_wordsim297())
            # p_240_scalar = tf.Summary(value=[tf.Summary.Value(tag="ws240 pearsonr rate", simple_value=p_240)])
            # s_240_scalar = tf.Summary(value=[tf.Summary.Value(tag="ws240 spearmanr rate", simple_value=s_240)])
            # p_297_scalar = tf.Summary(value=[tf.Summary.Value(tag="ws297 pearsonr rate", simple_value=p_297)])
            # s_297_scalar = tf.Summary(value=[tf.Summary.Value(tag="ws297 spearmanr rate", simple_value=s_297)])
            # print("eval_ws240:")
            # print('pearsonr:%s' % p_240)
            # print('spearmanr:%s' % s_240)
            # print("eval_ws297:")
            # print('pearsonr:%s' % p_297)
            # print('spearmanr:%s' % s_297)
            # train_writer.add_summary(p_240_scalar, step_global)
            # train_writer.add_summary(s_240_scalar, step_global)
            # train_writer.add_summary(p_297_scalar, step_global)
            # train_writer.add_summary(s_297_scalar, step_global)

            sum_loss = 0
            sum_dev_loss = 0
            sum_acc_t = 0
            sum_acc_d = 0

            start = time.time()


def test(sess):
    # _, _, test_corpus = reader.read_corpus(index='1_0.2', pick_train=False, pick_valid=False, pick_test=True)
    # test_corpus, _, _ = reader.read_corpus(index=0, pick_train=True, pick_valid=False, pick_test=False)
    _, _, test_corpus = reader.read_corpus(index='klb_150', pick_train=False, pick_valid=False, pick_test=True)

    glossary, word2id = reader.read_glossary()

    test_inputs = []
    test_lenth = []
    test_labels = []
    test_num = 0
    for item in test_corpus:
        # test_inputs.append(item[1])
        # test_lenth.append(item[2])
        # if item[0] in [1, 'T', 1.0]:
        #     test_labels.append(1)
        # elif item[0] in [0, 'F', 0.0]:
        #     test_labels.append(0)
        # test_num += 1
        test_inputs.append(item[0][0])
        test_lenth.append(int(item[0][1]))
        if item[1] in [1, 'T', 1.0]:
            test_labels.append(1)
        elif item[1] in [0, 'F', 0.0]:
            test_labels.append(0)
        test_num += 1

    model = EncoderModel(
        batch_size=FLAGS.batch_size,
        glossary_size=FLAGS.glossary_size,
        embedding_size=FLAGS.embedding_size,
        hidden_size=FLAGS.hidden_size,
        attn_lenth=FLAGS.attn_lenth
    )
    model.build_test_graph(150)

    saver = tf.train.Saver()
    test_labels = np.reshape(test_labels, [test_num, 1])


    if restore_from_checkpoint(sess, saver, 'save/pt_bi_lstm_attn/1'):
        # test_loss, accuracy, expection, w2v, alpha = sess.run(
        #     [model.test_loss, model.test_accuracy, model.expection, model.embeddings, model.alpha],
        #                               feed_dict=test_feed_dict)
        total_test_loss = 0
        total_accuracy = 0
        total_expection = []
        for piece_inputs, piece_lenth, piece_labels in get_test_batches(test_inputs, test_lenth, test_labels, test_num):
            piece_num = len(piece_inputs)
            test_feed_dict = {
                model.test_inputs: piece_inputs,
                model.test_lenth: piece_lenth,
                model.test_labels: piece_labels
            }
            test_loss, accuracy, expection, w2v = sess.run(
                [model.test_loss, model.test_accuracy, model.expection, model.embeddings],
                feed_dict=test_feed_dict)
            total_test_loss += test_loss * piece_num
            total_accuracy += accuracy * piece_num
            total_expection.extend(expection)
        total_test_loss /= test_num
        total_accuracy /= test_num

        for i in range(test_num):
            print(i, [glossary[word] for word in test_inputs[i]])
            print(test_inputs[i])
            # print(alpha[i])
            print(test_labels[i], total_expection[i])

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

            print("Validation: Average loss: {};".format(total_test_loss))
            print("     accuracy rate: {:.4f}".format(total_accuracy))
            print("About negative samples:")
            print("     precision rate: {:.4f}".format(P))
            print("     recall rate: {:.4f}".format(R))
            print("     f-value: {:.4f}".format(F))

            print("About positive samples:")
            print("     precision rate: {:.4f}".format(P_))
            print("     recall rate: {:.4f}".format(R_))
            print("     f-value: {:.4f}".format(F_))

        def eval_ws(ws_list):
            from scipy import stats
            from numpy import linalg as LA

            logits = []
            real = []
            eval = []

            for iter_ws in ws_list:
                if iter_ws[0] not in glossary or iter_ws[1] not in glossary:
                    continue
                else:
                    A = word2id[iter_ws[0]]
                    B = word2id[iter_ws[1]]
                    real.append(iter_ws[2])
                    logits.extend([w2v[A], w2v[B]])

            for i in range(len(logits) // 2):
                A_vec = logits[2 * i]
                B_vec = logits[2 * i + 1]
                normed_A_vec = LA.norm(A_vec, axis=0)
                normed_B_vec = LA.norm(B_vec, axis=0)
                sim = sum(np.multiply(A_vec, B_vec))
                eval.append(sim / normed_A_vec / normed_B_vec)
                # print(sim/normed_A_vec/normed_B_vec)

            print('pearsonr:%s' % (stats.pearsonr(real, eval)[0]))
            print('spearmanr:%s' % (stats.spearmanr(real, eval).correlation))

        f_value()
        eval_ws(reader.read_wordsim240())
        eval_ws(reader.read_wordsim297())

        return total_expection
    else:
        print("error!")

def test_without_eval(sess):
    _, _, test_corpus = reader.read_corpus(index=1, pick_train=False, pick_valid=False, pick_test=True)

    test_inputs = []
    test_lenth = []
    test_num = 0
    for item in test_corpus:
        test_inputs.append(item[0][0])
        test_lenth.append(int(item[0][1]))
        test_num += 1

    model = EncoderModel(
        batch_size=FLAGS.batch_size,
        glossary_size=FLAGS.glossary_size,
        embedding_size=FLAGS.embedding_size,
        hidden_size=FLAGS.hidden_size,
        attn_lenth=FLAGS.attn_lenth
    )
    model.build_test_graph()

    saver = tf.train.Saver()

    expection = []
    if restore_from_checkpoint(sess, saver, 'save/pt_bi_lstm_attn/relu'):
        for piece_inputs, piece_lenth in get_splited_batches(test_inputs, test_lenth, test_num):
            test_feed_dict = {
                model.test_inputs: piece_inputs,
                model.test_lenth: piece_lenth
            }
            piece_expect = np.reshape(sess.run([model.expection], feed_dict=test_feed_dict), [-1])
            expection.extend(piece_expect)
    return expection



def test_onesent(sess, sent):
    import jieba
    import re
    glossary, word2id = reader.read_glossary()

    temp = list(jieba.cut(sent))
    for index, item in enumerate(temp):
        if item in [' ', '\u3000', '    ']:
            temp.remove(item)
    for index, item in enumerate(temp):
        word = item.lower()
        word = re.sub(r'[0-9]+', '^数', word)
        if word not in glossary:
            word = '^替'
        temp[index] = word
    temp.append('^终')
    num = len(temp)
    while (len(temp)) < 150:
        temp.append('^填')

    print(sent)

    sent = np.array([word2id[item] for item in temp])
    print(sent)
    print([glossary[i] for i in sent])

    model = EncoderModel(
        batch_size=FLAGS.batch_size,
        glossary_size=FLAGS.glossary_size,
        embedding_size=FLAGS.embedding_size,
        hidden_size=FLAGS.hidden_size,
        attn_lenth=FLAGS.attn_lenth
    )
    model.build_test_graph(150)

    saver = tf.train.Saver()

    if restore_from_checkpoint(sess, saver, 'save/pt_bi_lstm_attn/2'):
        result, alpha = sess.run([model.expection, model.alpha], feed_dict={model.test_inputs:[sent], model.test_lenth:[num]})
        print("Predict result: ")
        print(alpha)
        print(result[0])


def compile(stage):
    model = EncoderModel(
        batch_size=FLAGS.batch_size,
        glossary_size=FLAGS.glossary_size,
        embedding_size=FLAGS.embedding_size,
        hidden_size=FLAGS.hidden_size,
        attn_lenth=FLAGS.attn_lenth
    )
    if stage is 'train':
        model.build_train_graph()
        # model.build_validate_graph(relu)
    elif stage is 'test':
        model.build_test_graph(1)


def load_model(sess):
    model = EncoderModel(
        batch_size=FLAGS.batch_size,
        glossary_size=FLAGS.glossary_size,
        embedding_size=FLAGS.embedding_size,
        hidden_size=FLAGS.hidden_size,
        attn_lenth=FLAGS.attn_lenth
    )
    model.build_test_graph(89)

    saver = tf.train.Saver(
        [model.embeddings,
         model.lstm_fw_cell.weights[0],
         model.lstm_fw_cell.weights[1],
         model.lstm_bw_cell.weights[0],
         model.lstm_bw_cell.weights[1],
         model.u1_w,
         model.u1_b,
         model.u2_w
         ]
    )
    if restore_from_checkpoint(sess, saver, 'save/pt_bi_lstm_attn/relu'):
        print(sess.run(model.lstm_fw_cell.weights))
    #     embeddings = sess.run(model.embeddings)
    #     lstm_fw_cell = sess.run(model.lstm_fw_cell.weights)
    #     lstm_bw_cell = sess.run(model.lstm_bw_cell.weights)
    #     u1_w = sess.run(model.u1_w)
    #     u1_b = sess.run(model.u1_b)
    #     u2_w = sess.run(model.u2_w)
    #
    #     import pickle
    #     with open('data/temp/embeddings', 'wb') as fp:
    #         pickle.dump(embeddings, fp)
    #     with open('data/temp/lstm_fw_cell', 'wb') as fp:
    #         pickle.dump(lstm_fw_cell, fp)
    #     with open('data/temp/lstm_bw_cell', 'wb') as fp:
    #         pickle.dump(lstm_bw_cell, fp)
    #     with open('data/temp/attn_w', 'wb') as fp:
    #         pickle.dump(u1_w, fp)
    #     with open('data/temp/attn_b', 'wb') as fp:
    #         pickle.dump(u1_b, fp)
    #     with open('data/temp/attn_u', 'wb') as fp:
    #         pickle.dump(u2_w, fp)


def main(_):
    with tf.Graph().as_default(), tf.Session() as sess:
        # train(sess)
        test(sess)
        # load_model(sess)
        # test_onesent(sess, '手机电池炸伤机主 商家称便宜货不保质量')
        # test_onesent(sess, '乐利来案例:医科大学朱志华副教授就自身股骨头坏死改善的讲述')
        # test_onesent(sess, '茅台经营成功')
        # test_onesent(sess, '【康宝莱】河南郑州花季少女死于康宝莱传销相关部门难逃其咎： 我叫张云成，男，汉族，现年47岁，家住河南省淮阳县冯塘乡蔡李庄村。2017年6月5日,我的儿子张旭(17岁)从郑州回到家里,向家里要钱,...文字版>> http://t.cn/RonRbWK （新浪长微博>> http://t.cn/zOXAaic）')
        # test_onesent(sess, '人生不能靠心情或者，而要靠心态去生活。或者不是靠泪水博得同情，而是靠汗水赢得掌声！！干杯！')
        # compile('test')

        # ext = test_without_eval(sess)
        # # file = open('results/test_1_results', 'w', encoding='utf8')
        # import pandas as pd
        # df = pd.read_excel('data/corpus/relu.xlsx')
        # text = df.iloc[:, [0]]
        # text = [i[0] for i in np.array(text).tolist()]
        # data = [[ext[i], text[i]] for i in range(len(text))]
        # print(len(ext))
        # print(len(text))
        # for i in range(len(text)):
        #     line = str(i+relu) + ' ' + str(ext[i]) + ' ' + str(text[i]) + '\n'
        #     file.write(line)
        # pd.DataFrame(data).to_excel('results/test_1_results.xlsx', index=True, header=False)

        # expection = test(sess)
        # import pickle
        # with open('data/corpus/test_klb_150_raw', 'rb') as fp:
        #     text = pickle.load(fp)
        # import pandas as pd
        # data = []
        # for i in range(len(expection)):
        #     data.append([text[i][0], text[i][relu], expection[i]])
        # print(len(data))
        # pd.DataFrame(data).to_excel('results/klb_150_1.xlsx', index=True, header=['原文', '标记', '预测标记'])


if __name__ == '__main__':
    tf.app.run()
