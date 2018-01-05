import reader
import re
import jieba
import numpy as np
import pickle

VOCABULARY, WORD2ID = reader.read_glossary()

def splitSentence(text):
    list_temp = []
    str_temp = []
    last_special_flag = None
    quotes_flag = False

    def _add(_word, _seq):
        _seq.append(_word)
        return _seq

    def _new(_word, _seq, add_after_push=False):
        if _seq == []:
            _seq.append(_word)
        else:
            if add_after_push is True:
                _seq.append(_word)
                list_temp.append(''.join(_seq))
                _seq = []
            else:
                list_temp.append(''.join(_seq))
                _seq = [_word]
        return _seq

    punkt = set(u'。！？')

    for index in range(len(text)):

        if text[index] in punkt:
            special_flag = 'p'
        elif text[index] == '“':
            special_flag = 'l'
            quotes_flag = True
        elif text[index] == '”':
            special_flag = 'r'
            quotes_flag = False
        else:
            special_flag = 'a'

        if last_special_flag is None and special_flag is 'p':
            continue
        elif last_special_flag is None and special_flag in ['a', 'l']:
            str_temp = _new(text[index], str_temp)
        elif last_special_flag in ['a', 'l'] and special_flag in ['a', 'l']:
            str_temp = _add(text[index], str_temp)
        elif last_special_flag in ['a', 'l'] and special_flag is 'p':
            str_temp = _add(text[index], str_temp)
        elif last_special_flag in ['a', 'l'] and special_flag is 'r':
            str_temp = _add(text[index], str_temp)
        elif last_special_flag is 'p' and special_flag is 'a':
            if quotes_flag is True:
                str_temp = _add(text[index], str_temp)
            elif quotes_flag is False:
                str_temp = _new(text[index], str_temp)
        elif last_special_flag is 'p' and special_flag is 'p':
            str_temp = _add(text[index], str_temp)
        elif last_special_flag is 'p' and special_flag is 'l':
            str_temp = _new(text[index], str_temp)
        elif last_special_flag is 'p' and special_flag is 'r':
            str_temp = _new(text[index], str_temp, add_after_push=True)
        elif last_special_flag is 'r':
            str_temp = _add(text[index], str_temp)

        last_special_flag = special_flag
    if str_temp != []:
        list_temp.append(''.join(str_temp))
    return list_temp

def sent2list(batch_text):
    if isinstance(batch_text, list) is True:
        return_text = []
        for index, item in enumerate(batch_text):
            temp = list(jieba.cut(item))
            temp = [jtem for jtem in temp if jtem not in [' ', '\u3000', '    ', '\xa0']]
            if temp == []:
                continue
            for jndex, jtem in enumerate(temp):
                word = jtem.lower()
                word = re.sub(r'[0-9]+', '^数', word)
                if word not in VOCABULARY:
                    word = '^替'
                temp[jndex] = word
            temp.append('^终')
            num = len(temp)
            return_text.append([temp, num])
        return return_text

    elif isinstance(batch_text, str) is True:
        temp = list(jieba.cut(batch_text))
        for jndex, jtem in enumerate(temp):
            if jtem in [' ', '\u3000', '    ']:
                temp.remove(jtem)
        for jndex, jtem in enumerate(temp):
            word = jtem.lower()
            word = re.sub(r'[0-9]+', '^数', word)
            if word not in VOCABULARY:
                word = '^替'
            temp[jndex] = word
        temp.append('^终')
        num = len(temp)
        return [temp, num]

def paddingList(batch_text, seq_size):
    result = []
    for index, item in enumerate(batch_text):
        if (item != None):
            if (len(item[0]) <= seq_size):
                while (len(item[0])) < seq_size:
                    item[0].append('^填')
                result.append(item)
        else:
            result.append([[['^填']*seq_size, 0]])
    return result




def test():
    with open('data/corpus/test_1', 'rb') as fp:
        text = pickle.load(fp)
    write_to_file = []
    for item in text:
        item_flag = item[0]
        item_title = [sent2list(item[1])]
        item_text = sent2list(splitSentence(item[2]))
        item_title = paddingList(item_title, seq_size=120)
        item_text = paddingList(item_text, seq_size=120)
        write_to_file.append([item_flag, item_title, item_text])

    with open('data/corpus/test_1_raw', 'wb') as fp:
        pickle.dump(write_to_file, fp)

def afterProcess():
    with open('data/corpus/test_1_raw', 'rb') as fp:
        text = pickle.load(fp)
    for item in text:
        item[0] = 'T' if item[0] == '1' else 'F'
        item[1][0][0] = [WORD2ID[word] for word in item[1][0][0]]
        for jtem in item[2]:
            jtem[0] = [WORD2ID[word] for word in jtem[0]]
        if item[2] == []:
            item[2] = [[[WORD2ID['^填']]*120, 0]]
    with open('data/corpus/test_1', 'wb') as fp:
        pickle.dump(text, fp)


def write():
    with open('data/corpus/raw_', 'rb') as fp:
        text = pickle.load(fp)
    text_true = []
    text_false = []
    for item in text:
        if item[0] == 'T':
            text_true.append(item)
        elif item[0] == 'F':
            text_false.append(item)
    with open('data/corpus/raw_true', 'wb') as fp:
        pickle.dump(text_true, fp)
    with open('data/corpus/raw_false', 'wb') as fp:
        pickle.dump(text_false, fp)

def sample_from_raw():
    import random
    with open('data/corpus/raw_true', 'rb') as fp:
        text_true = pickle.load(fp)
    with open('data/corpus/raw_false', 'rb') as fp:
        text_false = pickle.load(fp)

    print(round(len(text_false)*0.3))
    text_false = random.sample(text_false, round(len(text_false)*0.3))
    text = text_false + text_true
    random.shuffle(text)
    with open('data/corpus/train_0_subsample', 'wb') as fp:
        pickle.dump(text, fp)



def empty_check(filepath='raw_'):
    with open('data/corpus/' + filepath, 'rb') as fp:
        text = pickle.load(fp)
    for i in text:
        if len(i[2]) > 500:
            print(i)
        if i[1][0][1] == 0 and len(i[2]) == 0:
            print(i)

# empty_check('train_0_subsample')
