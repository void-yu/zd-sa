# import tensorflow as tf
# import pickle
# import numpy as np
#
# import reader
#
# flags = tf.app.flags
#
#
# flags.DEFINE_integer("save_every_n", 10, "")
#
# flags.DEFINE_integer("embedding_size", 400, "")
# flags.DEFINE_integer("hidden_size", 300, "")
# flags.DEFINE_integer("attn_lenth", 350, "")
#
# '''
#     33272(known)+(1876+8)(unknown)=35156
# '''
# flags.DEFINE_integer("glossary_size", 35156, "")
# flags.DEFINE_integer("batch_size", 128, "")
# flags.DEFINE_integer("seq_lenth", 150, "")
# flags.DEFINE_integer("epoches", 100, "")
#
# FLAGS = flags.FLAGS
#
# flags.DEFINE_string("ckpt_dir", "../save/bd_ce_w=1_tanh_init_mean_truncated", "")

# from test.model_sent import EncoderModel as em_sent
#
# with tf.Graph().as_default(), tf.Session() as sess:
#     model = em_sent(
#         seq_size=FLAGS.seq_lenth,
#         glossary_size=FLAGS.glossary_size,
#         embedding_size=FLAGS.embedding_size,
#         hidden_size=FLAGS.hidden_size,
#         attn_lenth=FLAGS.attn_lenth,
#         is_training=True
#     )
#     model.buildTrainGraph()

# import pickle
# import jieba
#
# # with open('', 'rb') as fp:
# #     corpus = pickle.load(fp)
# with open('D:/Codes/git/fastText-0.1.0/data/raw/test.txt', 'r') as f:
#     corpus = f.readlines()
#     with open('D:/Codes/git/fastText-0.1.0/data/raw/test_.txt', 'w', encoding='utf8') as fw:
#         for i in corpus:
#             fw.write(' '.join(jieba.cut(i)) + '\n')

# def f_value(test_num, test_labels, total_expection):
#     # 真正例
#     TP = 0
#     # 假正例
#     FP = 0
#     # 假反例
#     FN = 0
#     # 真反例
#     TN = 0
#
#     for i in range(test_num):
#         if test_labels[i] == 0 and total_expection[i] == 0:
#             TP += 1
#         elif test_labels[i] == 0 and total_expection[i] == 1:
#             FN += 1
#         elif test_labels[i] == 1 and total_expection[i] == 0:
#             FP += 1
#         elif test_labels[i] == 1 and total_expection[i] == 1:
#             TN += 1
#
#     P = TP / (TP + FP + 0.0001)
#     R = TP / (TP + FN + 0.0001)
#     F = 2 * P * R / (P + R + 0.0001)
#     P_ = TN / (TN + FN + 0.0001)
#     R_ = TN / (TN + FP + 0.0001)
#     F_ = 2 * P_ * R_ / (P_ + R_ + 0.0001)
#     ACC = (TP + TN) / (TP + FP + TN + FN + 0.0001)
#     print("     accuracy rate: {:.4f}".format(ACC))
#     print("About positive samples:")
#     print("     precision rate: {:.4f}".format(P))
#     print("     recall rate: {:.4f}".format(R))
#     print("     f-value: {:.4f}".format(F))
#
#     print("About negative samples:")
#     print("     precision rate: {:.4f}".format(P_))
#     print("     recall rate: {:.4f}".format(R_))
#     print("     f-value: {:.4f}".format(F_))
#
# O = []
# E = []
# f = open('../data/out', 'r', encoding='utf8')
# lines = f.readlines()
# for i in lines:
#     spd = i.split('\t')
#     o = 1 if spd[0] is 'F' else 0
#     e = 1 if spd[1].split(' ')[0][-1] is 'F' else 0
#     O.append(o)
#     E.append(e)
# f_value(len(O), O, E)


import lime
import sklearn
import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.metrics

from sklearn.datasets import fetch_20newsgroups
categories = ['alt.atheism', 'soc.religion.christian']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)
class_names = ['atheism', 'christian']
vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False)
train_vectors = vectorizer.fit_transform(newsgroups_train.data)
test_vectors = vectorizer.transform(newsgroups_test.data)
rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
rf.fit(train_vectors, newsgroups_train.target)
pred = rf.predict(test_vectors)
sklearn.metrics.f1_score(newsgroups_test.target, pred, average='binary')
from lime import lime_text
from sklearn.pipeline import make_pipeline
c = make_pipeline(vectorizer, rf)
print(c.predict_proba([newsgroups_test.data[0]]))
from lime.lime_text import LimeTextExplainer
explainer = LimeTextExplainer(class_names=class_names)
idx = 83
print('-------------------------------------------')
print(newsgroups_test.data[idx])
print('-------------------------------------------')
exp = explainer.explain_instance(newsgroups_test.data[idx], c.predict_proba, num_features=6)
print('Document id: %d' % idx)
print('Probability(christian) =', c.predict_proba([newsgroups_test.data[idx]])[0,1])
print('True class: %s' % class_names[newsgroups_test.target[idx]])
print(exp.as_list())
print('Original prediction:', rf.predict_proba(test_vectors[idx])[0,1])
tmp = test_vectors[idx].copy()
tmp[0,vectorizer.vocabulary_['Posting']] = 0
tmp[0,vectorizer.vocabulary_['Host']] = 0
print('Prediction removing some features:', rf.predict_proba(tmp)[0,1])
print('Difference:', rf.predict_proba(tmp)[0,1] - rf.predict_proba(test_vectors[idx])[0,1])

fig = exp.as_pyplot_figure()
exp.show_in_notebook(text=False)
exp.save_to_file('oi.html')
exp.show_in_notebook(text=True)