import reader
import tensorflow as tf
from model_sgd import EncoderModel
import time
import numpy as np


flags = tf.app.flags

flags.DEFINE_string("tensorboard_dir", "tensorb/me-sum",
                    "Directory to write the model and training summaries.")
flags.DEFINE_string("ckpt_dir", "save/pt-me-sum",
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
flags.DEFINE_integer("batch_size", 1,
                     "Number of sentences per batch. And each time we feed the whole sentence.")
flags.DEFINE_integer("max_seq_size", 120, "")
flags.DEFINE_integer("epoches", 100,
                     "Number of epochs processed. I set only One batch is processed in One epoch.")

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

# def get_nolabel_batches(inputs, lenth, num):
#     batch_size = FLAGS.max_batch_size
#     count_num = 0
#
#     while(count_num < num):
#         batch_inputs = []
#         batch_lenth = []
#         for j in range(min(batch_size, num - count_num)):
#             batch_inputs.append(inputs[count_num])
#             batch_lenth.append(lenth[count_num])
#             count_num += 1
#         yield batch_inputs, batch_lenth


def restore_from_checkpoint(sess, saver, dir):
    ckpt = tf.train.get_checkpoint_state(dir)
    if not ckpt or not ckpt.model_checkpoint_path:
        print('No checkpoint found at {}'.format(dir))
        return False
    saver.restore(sess, ckpt.model_checkpoint_path)
    return True

def get_piece(corpus):
    for index in range(len(corpus)):
        title_input = [corpus[index][1][0][0]]
        title_lenth = [corpus[index][1][0][1]]
        text_inputs = [item[0] for item in corpus[index][2]]
        text_lenths = [item[1] for item in corpus[index][2]]
        label = [1 if corpus[index][0] is 'T' else 0]
        yield title_input, title_lenth, text_inputs, text_lenths, label


def train(sess):
    # Pretreatment
    print("Read file --")
    start = time.time()

    # id2word, word2id = reader.read_glossary()
    train_corpus, _, _ = reader.read_corpus(index='1', pick_valid=False, pick_test=False)
    pretrained_wv = reader.read_initw2v()

    end = time.time()
    print("Read finished -- {:.4f} sec".format(end-start))

    # Build model
    print("Building model --")
    start = end

    model = EncoderModel(
        max_seq_size=120,
        glossary_size=FLAGS.glossary_size,
        embedding_size=FLAGS.embedding_size,
        hidden_size=FLAGS.hidden_size,
        attn_lenth=FLAGS.attn_lenth,
        learning_rate=0.01
    )
    model.buildTrainGraph()

    init = tf.global_variables_initializer()
    # sess.run(init, feed_dict={model.pretrained_wv: pretrained_wv})
    sess.run(init)

    saver = tf.train.Saver(tf.trainable_variables(),
        # [
        #     model.embeddings,
        #     model.lstm_fw_cell.weights,
        #     model.lstm_bw_cell.weights,
        #     model.attn_w,
        #     model.attn_b,
        #     model.attn_u,
        #     model.inte_attn_w,
        #     model.inte_attn_b,
        #     model.inte_attn_u,
        #     model.merge_inde_w,
        #     model.merge_inde_b,
        #     model.merge_inte_w,
        #     model.merge_inte_b
        # ],
        max_to_keep=10)
    train_writer = tf.summary.FileWriter(logdir=FLAGS.tensorboard_dir, graph=sess.graph)

    end = time.time()
    print("Building model finished -- {:.4f} sec".format(end - start))

    if not restore_from_checkpoint(sess, saver, FLAGS.ckpt_dir):
        return
    step_global = 0
    sum_loss = 0
    # sum_dev_loss = 0
    sum_acc_t = 0
    # sum_acc_d = 0
    # max_acc = 0

    print("Training initialized")
    start = time.time()

    for epoch in range(FLAGS.epoches):
        for train_title_input, train_title_lenth, train_text_inputs, train_text_lenths, train_label in get_piece(train_corpus):
            step_global += 1
            feed_dict = {
                model.title_input: train_title_input,
                model.title_lenth: train_title_lenth,
                model.text_inputs: train_text_inputs,
                model.text_lenths: train_text_lenths,
                model.label: train_label
            }

            loss, _, t_scalar, t_acc = sess.run([model.loss,
                                                model.optimizer,
                                                model.train_scalar,
                                                model.train_accuracy],
                                         feed_dict=feed_dict)
            # print(aaaa, bbbb, loss)
            sum_loss += loss
            sum_acc_t += t_acc

            # for dev_inputs, dev_lenth, dev_labels in get_batches(valid_inputs, valid_lenth, valid_labels, valid_num):
            #     dev_feed_dict = {
            #         model.dev_inputs: dev_inputs,
            #         model.dev_lenth: dev_lenth,
            #         model.dev_labels: dev_labels
            #     }
            #     dev_loss, d_scalar, d_acc, w2v = sess.run([model.dev_loss,
            #                                                model.dev_scalar,
            #                                                model.dev_accuracy,
            #                                                model.embeddings],
            #                                               feed_dict=dev_feed_dict)
            #     sum_dev_loss += dev_loss
            #     sum_acc_d += d_acc
            #
            # sum_dev_loss /= valid_num
            # sum_acc_d /= valid_num

            # def eval_ws(ws_list):
            #     from scipy import stats
            #     from numpy import linalg as LA
            #
            #     logits = []
            #     real = []
            #     eval = []
            #
            #     for iter_ws in ws_list:
            #         if iter_ws[0] not in id2word or iter_ws[1] not in id2word:
            #             continue
            #         else:
            #             A = word2id[iter_ws[0]]
            #             B = word2id[iter_ws[1]]
            #             real.append(iter_ws[2])
            #             logits.extend([w2v[A], w2v[B]])
            #
            #     for i in range(len(logits) // 2):
            #         A_vec = logits[2 * i]
            #         B_vec = logits[2 * i + 1]
            #         normed_A_vec = LA.norm(A_vec, axis=0)
            #         normed_B_vec = LA.norm(B_vec, axis=0)
            #         sim = sum(np.multiply(A_vec, B_vec))
            #         eval.append(sim / normed_A_vec / normed_B_vec)
            #
            #     pearsonr = stats.pearsonr(real, eval)[0]
            #     spearmanr = stats.spearmanr(real, eval).correlation
            #     return pearsonr, spearmanr


            if step_global % FLAGS.save_every_n == 0:
                end = time.time()
                print("Training: Average loss at step {}: {};".format(step_global, sum_loss[0] / FLAGS.save_every_n),
                      "time: {:.4f} sec;".format(end - start),
                      "accuracy rate: {:.4f}".format(sum_acc_t[0] / FLAGS.save_every_n))
                # print("Validation: Average loss: {};".format(sum_dev_loss / FLAGS.save_every_n),
                #       "accuracy rate: {:.4f}".format(sum_acc_d / FLAGS.save_every_n))

                saver.save(sess, FLAGS.ckpt_dir + "/step{}.ckpt".format(step_global))

                train_writer.add_summary(t_scalar, step_global)
                # ac_scalar = tf.Summary(value=[tf.Summary.Value(tag="accuracy rate", simple_value=sum_acc_d / FLAGS.save_every_n)])
                # train_writer.add_summary(ac_scalar, step_global)

                sum_loss = 0
                # sum_dev_loss = 0
                sum_acc_t = 0
                # sum_acc_d = 0

                start = time.time()

def test(sess):
    _, _, test_corpus = reader.read_corpus(index='1', pick_train=False, pick_valid=False, pick_test=True)
    # _, test_corpus, _ = reader.read_corpus(index='1', pick_train=False, pick_valid=True, pick_test=False)

    model = EncoderModel(
        max_seq_size=120,
        glossary_size=FLAGS.glossary_size,
        embedding_size=FLAGS.embedding_size,
        hidden_size=FLAGS.hidden_size,
        attn_lenth=FLAGS.attn_lenth,
        learning_rate=0.01
    )
    model.buildTrainGraph()

    init = tf.global_variables_initializer()
    # sess.run(init, feed_dict={model.pretrained_wv: pretrained_wv})
    sess.run(init)
    saver = tf.train.Saver(tf.trainable_variables())

    if not restore_from_checkpoint(sess, saver, FLAGS.ckpt_dir):
        return
    sum_loss = 0
    sum_acc_t = 0
    test_logits = []
    test_labels = []
    i = 2

    print("Test initialized")
    start = time.time()

    for test_title_input, test_title_lenth, test_text_inputs, test_text_lenths, test_label in get_piece(test_corpus):
        feed_dict = {
            model.title_input: test_title_input,
            model.title_lenth: test_title_lenth,
            model.text_inputs: test_text_inputs,
            model.text_lenths: test_text_lenths,
            model.label: test_label
        }

        loss, t_acc, logit, label = sess.run([model.loss,
                                                        model.train_accuracy,
                                                        model._logits,
                                                        model._labels],
                                                 feed_dict=feed_dict)
        print(i, logit, label)
        i+= 1
        sum_loss += loss
        sum_acc_t += t_acc
        test_logits.append(logit)
        test_labels.append(label)

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
        for i in range(len(test_corpus)):
            if test_labels[i] == 0 and test_logits[i] == 0:
                TP += 1
            elif test_labels[i] == 0 and test_logits[i] == 1:
                FN += 1
            elif test_labels[i] == 1 and test_logits[i] == 0:
                FP += 1
            elif test_labels[i] == 1 and test_logits[i] == 1:
                TN += 1
        
        P = TP / (TP + FP + 0.0001)
        R = TP / (TP + FN + 0.0001)
        F = 2 * P * R / (P + R)
        P_ = TN / (TN + FN + 0.0001)
        R_ = TN / (TN + FP + 0.0001)
        F_ = 2 * P_ * R_ / (P_ + R_)

        print("About negative samples:")
        print("     precision rate: {:.4f}".format(P))
        print("     recall rate: {:.4f}".format(R))
        print("     f-value: {:.4f}".format(F))
        
        print("About positive samples:")
        print("     precision rate: {:.4f}".format(P_))
        print("     recall rate: {:.4f}".format(R_))
        print("     f-value: {:.4f}".format(F_))
        
    end = time.time()
    print("Average loss;".format(sum_loss[0] / len(test_corpus)),
          "time: {:.4f} sec;".format(end - start),
          "accuracy rate: {:.4f}".format(sum_acc_t[0] / len(test_corpus)))
    f_value()



def compile(sess, stage):
    model = EncoderModel(
        max_seq_size=100,
        glossary_size=30000,
        embedding_size=400,
        hidden_size=300,
        attn_lenth=300,
        learning_rate=0.1
    )
    if stage is 'train':
        model.buildTrainGraph()


def reload_model(sess):
    model = EncoderModel(
        max_seq_size=120,
        glossary_size=FLAGS.glossary_size,
        embedding_size=FLAGS.embedding_size,
        hidden_size=FLAGS.hidden_size,
        attn_lenth=FLAGS.attn_lenth,
        learning_rate=0.01
    )
    # model.loadPreTrainedParameters()
    model.buildTrainGraph()
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()
    if restore_from_checkpoint(sess, saver, 'save/temp'):
        print(sess.run(model.lstm_fw_cell.weights))

    # embeddings, attn_w, attn_b, attn_u = reader.read_pretrained()
    # sess.run(init, feed_dict={
    #     model.pretrained_embeddings: embeddings,
    #     model.pretrained_attn_w: attn_w,
    #     model.pretrained_attn_b: attn_b,
    #     model.pretrained_attn_u: attn_u
    # })
    #
    # saver = tf.train.Saver(
    #     [model.rnn_fw_cell.weights[0],
    #      model.rnn_fw_cell.weights[1],
    #      model.rnn_bw_cell.weights[0],
    #      model.rnn_bw_cell.weights[1]]
    # )
    # save1 = tf.train.Saver()

    # if restore_from_checkpoint(sess, saver, 'save/temp/1'):
    #     print(sess.run(model.rnn_fw_cell.weights))
    #     print(sess.run(model.lstm_fw_cell.weights))
    # save1.save(sess, 'save/temp/pretrained.ckpt')



def main(_):
    with tf.Graph().as_default(), tf.Session() as sess:
        # train(sess)
        test(sess)
        # reload_model(sess)
        # compile(sess, 'train')

if __name__ == '__main__':
    tf.app.run()
