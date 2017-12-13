import numpy as np
import pandas as pd
import pickle
import reader
import jieba
import re

def read_exls(with_labels=False):
    df = pd.read_excel('data/corpus/2.xlsx')
    if with_labels is False:
        text = df.iloc[:, [0]]
        text = [[i[0]] for i in np.array(text).tolist()]
    else:
        text = df.iloc[:, [0, 1]]
        text = [[i[0], i[1]] for i in np.array(text).tolist()]
    glossary, word2id = reader.read_glossary()

    for index, item in enumerate(text):
        temp = list(jieba.cut(item[0]))
        for jndex, jtem in enumerate(temp):
            if jtem in [' ', '\u3000', '    ']:
                temp.remove(jtem)
        for jndex, jtem in enumerate(temp):
            word = jtem.lower()
            word = re.sub(r'[0-9]+', '^数', word)
            if word not in glossary:
                word = '^替'
            temp[jndex] = word
        temp.append('^终')
        num = len(temp)
        text[index][0] = [temp, num]

    max_len = max([i[0][1] for i in text])
    for index, item in enumerate(text):
        while(len(item[0][0])) < max_len:
            item[0][0].append('^填')

    for item in text:
        item[0][0] = [word2id[word] for word in item[0][0]]

    with open('data/corpus/test_1', 'wb') as fp:
        pickle.dump(text, fp)


def read_text():
    with open('data/corpus/test_1', 'rb') as fp:
        text = pickle.load(fp)
    print(text)
    print(np.shape(text))
    print(len(text))
    print(max([i[0][1] for i in text]))


def read_results():
    file = open('results/test_1_results', 'r', encoding='utf8')
    content = file.readlines()
    for i in content:
        item = i.split()
        if item[1] == '0.0':
            print(item[0], item[1], item[2])

