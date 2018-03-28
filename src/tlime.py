import sklearn
import sklearn.ensemble
import sklearn.metrics
from sklearn.pipeline import make_pipeline
import interface_tlime

from sklearn.datasets import fetch_20newsgroups
# categories = ['alt.atheism', 'soc.religion.christian']
# newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
# newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)
class_names = ['atheism', 'christian']
#
# vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False)
# train_vectors = vectorizer.fit_transform(newsgroups_train.data)
# test_vectors = vectorizer.transform(newsgroups_test.data)
#
# rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
# rf.fit(train_vectors, newsgroups_train.target)
#
# pred = rf.predict(test_vectors)
# sklearn.metrics.f1_score(newsgroups_test.target, pred, average='binary')
#
# c = make_pipeline(vectorizer, rf)
# for i in c.steps:
#     print(i)
# print(c.predict_proba([newsgroups_test.data[0]]))



from lime.lime_text import LimeTextExplainer
import jieba
explainer = LimeTextExplainer(class_names=['Negative', 'Positive'], split_expression=' ')

idx = 83
# sss = ' '.join(jieba.cut('家住河南省淮阳县冯塘乡蔡李庄村。2017年6月5日,我的儿子张旭(17岁)从郑州回到家里,向家里要钱,...文字版>> http://t.cn/RonRbWK （新浪长微博>> http://t.cn/zOXAaic）'))
sss = ' '.join(jieba.cut('走进康宝莱之前，我听到了台上的分享嘉宾说了这句话：“也许做康宝莱这件事不是你的梦想，但做好康宝莱能实现你所有的梦想！” 我信了。 果然这是真的。 同时，在这个过程中，帮助更多人获得好身材、健康、财富、和精彩人生，也一并成为了我的梦想！和我其他的梦想一起实现了！——肖珂宇 网页链接'))
exp = explainer.explain_instance(sss, interface_tlime.test_for_lime, labels=(1,), num_samples=5000)

fig = exp.as_pyplot_figure()
exp.show_in_notebook(text=False)
exp.save_to_file('oi.html')
exp.show_in_notebook(text=True)