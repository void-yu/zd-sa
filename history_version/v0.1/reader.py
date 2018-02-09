import numpy as np
import pickle


def read_glossary():
    with open('data/glossary', 'rb') as fp:
        id2word = pickle.load(fp)
        word2id = {}
        for index in range(len(id2word)):
            word2id[id2word[index]] = index
    return id2word, word2id


def read_corpus(index='', pick_train=True, pick_valid=True, pick_test=True):
    train_data = None
    valid_data = None
    test_data = None
    if pick_train is True:
        with open('data/corpus/train_%s' % index, 'rb') as fp:
            train_data = pickle.load(fp)
    if pick_valid is True:
        with open('data/corpus/valid_%s' % index, 'rb') as fp:
            valid_data = pickle.load(fp)
    if pick_test is True:
        with open('data/corpus/test_%s' % index, 'rb') as fp:
            test_data = pickle.load(fp)
    return train_data, valid_data, test_data

def read_file(filepath=''):
    test_data = None
    with open(filepath, 'rb') as fp:
        test_data = pickle.load(fp)
    return test_data

def read_initw2v():
    with open('data/initw2v_use', 'rb') as fp:
        embedding = pickle.load(fp)
    return embedding




def read_wordsim240():
    file_sim = open('data/words-240/Words-240.txt', 'r', encoding='utf8')
    ws240_list = []
    for iter in file_sim.readlines():
        temp = iter[:-1].split()
        temp[2] = float(temp[2])
        ws240_list.append(temp)
    return ws240_list


def read_wordsim297():
    file_sim = open('data/words-297/297.txt', 'r', encoding='utf8')
    ws297_list = []
    for iter in file_sim.readlines():
        temp = iter[:-1].split()
        temp[2] = float(temp[2])
        ws297_list.append(temp)
    return ws297_list


def read_pretrained():
    with open('data/pretrained/embeddings', 'rb') as fp:
        embeddings = pickle.load(fp)
    with open('data/pretrained/attn_w', 'rb') as fp:
        attn_w = pickle.load(fp)
    with open('data/pretrained/attn_b', 'rb') as fp:
        attn_b = pickle.load(fp)
    with open('data/pretrained/attn_u', 'rb') as fp:
        attn_u = pickle.load(fp)
    return embeddings, attn_w, attn_b, attn_u
