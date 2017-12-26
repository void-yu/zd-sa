import reader
import re
import jieba
import numpy as np

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

def paddingList(batch_text):
    max_len = max([i[1] for i in batch_text])
    for index, item in enumerate(batch_text):
        while (len(item[0])) < max_len:
            item[0].append('^填')
    return batch_text

def test():
    import pickle
    with open('data/corpus/train_2_raw', 'rb') as fp:
        text = pickle.load(fp)
    write_to_file = []
    item = text[250]
    item_flag = item[0]
    item_list = [sent2list(item[1])]
    for i in splitSentence(item[2]):
        print(i)
    item_list.extend(sent2list(splitSentence(item[2])))
    for i in item_list:
        print(i)

test()
