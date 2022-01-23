from spacy import displacy
from sklearn.metrics import *
import string
import unicodedata
import re
import numpy as np
import pickle
from define_name import *

############################################################
def preprocess_data(sent):
    sent = handle_bracket(sent)
    sent = re.sub(' +', ' ', sent)
    sent_out = ""
    parts = sent.split()
    sent_out = ' '.join([handle_character(i)[0] for i in parts])
    return sent_out

def process_unk(tokenizer, sq):
    temp = []
    for i in sq.split():
        if ['[UNK]'] == tokenizer.tokenize(i):
            temp.append(i[0]+i[1:].lower())
        else:
            temp.append(i)
    return ' '.join(temp)

def preprocessing_text(text):
    dictt = {'™': ' ', '‘': "'", '®': ' ', '×': ' ', '😀': ' ', '‑': ' - ','́': ' ', '—': ' - ', '̣': ' ', '–': ' - ', '`': "'",\
             '“': '"', '̉': ' ','’': "'", '̃': ' ', '\u200b': ' ', '̀': ' ', '”': '"', '…': '...', '\ufeff': ' ', '″': '"'}
    text = text.split('\n')
    text = [i.strip()  for i in text if i!='']
    out = ""
    for i in range(1,len(text)+1):
        out += text[i-1]+' .</x> '
    text = unicodedata.normalize('NFKC', out)
    res = ''
    for i in text:
        if i.isalnum() or i in string.punctuation or i == ' ':
            res += i
        elif i in dictt:
            res += dictt[i]
    text = preprocess_data(res)
    return text

def handle_bracket(test_str):
    res = re.findall(r'(\(|\[|\"|\'|\{)(.*?)(\}|\'|\"|\]|\))', test_str)
    if len(res) > 0:
        for r in res:
            sub_tring = "".join(r)
            start_index = test_str.find(sub_tring)
            end_index = start_index + len(r[1])
            test_str = test_str[: start_index+ 1] + " " + test_str[start_index+ 1:]
            test_str = test_str[: end_index + 2] + " " + test_str[end_index + 2:]
    return test_str

def handle_character(sub_string):
    char_end = [".", ",", ";", "?", "+", ":" ]
    count = 1
    for index in reversed(range(len(sub_string))):
        c = sub_string[index]
        if c not in char_end:
            break
        elif c in char_end:
            if sub_string[index -1] not in char_end:
                sub_string = sub_string[:index] + " " + sub_string[index:]
                count = 2
            break
    return sub_string, count

#################################
def isNotSubword(x, idx, sub = '▁'):
    return  idx < len(x) - 1 and sub in x[idx+1] and ((x[idx] in PUNCT_SPEC and idx > 0 and sub == x[idx-1]) or (sub in x[idx]))
def cutting_subword(X, sub = '▁', size=254):
    res_X = []
    st = 0
    cur = 0
    while (st < len(X)-size):
        flag = True
        for i in range(st+size-1, st-1, -1):
            if X[i][-1] in PUNCT_END and isNotSubword(X, i, sub):
                cur = i+1
                flag = False
                break
        if flag:
            for i in range(st+size-1, st-1, -1):
                if isNotSubword(X, i, sub):
                    cur = i+1
                    break
        if st == cur:
            cur += size
        res_X.append(X[st: cur])
        st = cur
    res_X.append(X[cur:])
    for i in range(len(res_X)):
        res_X[i].insert(0, TOKEN_START)
        res_X[i].append(TOKEN_END)
    return res_X
############################################################
def visualize_spacy(arr):
    if len(arr) < 1:
        return None
    text = ' '.join([i for i, j in arr])
    pos = 0
    start_end_labels = []
    for word, tag in arr:
        if len(start_end_labels) > 0 and tag == start_end_labels[-1][2]:
            temp = [start_end_labels[-1][0], pos+len(word), tag]
            start_end_labels[-1] = temp.copy()
        else:
            temp = [pos, pos+len(word), tag]
            start_end_labels.append(temp)
        pos += len(word) + 1
        
    ex = [{'text': text, 'ents': [{'start': x[0], 'end': x[1], 'label': x[2]} for x in start_end_labels if x[2]!= 0]}]
    return displacy.render(ex, manual=True, jupyter=True, style='ent', options = OPTIONS)#page=True
#########################################
def write_pickle(dt, path):
  with open(path, 'wb') as f:
    pickle.dump(dt, f)

def read_pickle(file):
  with open(file, 'rb') as f:
    return pickle.load(f)
#=========convert data for auto-labelling==============
def preprocessing_text2(text):
    dictt = {'™': ' ', '‘': "'", '®': ' ', '×': ' ', '😀': ' ', '‑': ' - ', '́': ' ', '—': ' - ', '̣': ' ', '–': ' - ', '`': "'",\
    '“': '"', '̉': ' ','’': "'", '̃': ' ', '\u200b': ' ', '̀': ' ', '”': '"', '…': '...', '\ufeff': ' ', '″': '"'}
    text = text.split('\n')
    text = ' '.join([i.strip() for i in text if i!=''])
    text = unicodedata.normalize('NFKC', text)
    res = ''
    for i in text:
        if i.isalnum() or i in string.punctuation or i == ' ':
            res += i
        elif i in dictt:
            res += dictt[i]
    text = preprocess_data(res)
    return text

def read_txt(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.readlines()

def write_txt(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        for i in data:
            f.write(i + '\n')

def convertfile2doccano(path, path2):
    res = []
    data = read_txt(path)
    for line in data:
        res.append(preprocessing_text2(line.strip()))
        write_txt(res, path2)

import pandas as pd
from tqdm.notebook import tqdm
import torch
import numpy as np
from sklearn.metrics import classification_report


def show_span_f1(dic):
    index = []
    da = []
    for tag, detail in dic.items():
        index.append(tag)
        da.append(detail)
    df = pd.DataFrame(da)
    df = df.set_index([pd.Index(index)])
    return df

def convert_spanformat(arr):
    if len(arr) < 1:
        return None
    text = ' '.join([i for i, j in arr])
    pos = 0
    start_end_labels = []
    for word, tag in arr:
        if len(start_end_labels) > 0 and tag == start_end_labels[-1][2]:
            temp = [start_end_labels[-1][0], pos+len(word), tag]
            start_end_labels[-1] = temp.copy()
        else:
            temp = [pos, pos+len(word), tag]
            start_end_labels.append(temp)
        pos += len(word) + 1

    res = dict()   
    for s, e, l in start_end_labels:
        if l != 'O' and l != 'PAD':
            if l not in res:
                res[l] = [(s, e)]
            else:
                res[l].append((s, e))
    return res
 
def compare_span(span1, span2, res, strict = FLAG_STRICT['MAX']):
    all_labels = set(list(span1.keys()) + list(span2.keys()))
    for l in all_labels:
        if l not in res:
            res[l] = [0, 0, 0, 0]
        if l not in span1:
            res[l][3] += len(span2[l])
            continue
        if l not in span2:
            res[l][2] += len(span1[l])
            continue
        res[l][2] += len(span1[l])
        res[l][3] += len(span2[l])
        for s, e in span1[l]:
            for s1, e1 in span2[l]:
                if strict == FLAG_STRICT['MAX']:
                    if s == s1 and e == e1:
                        res[l][0] += 1
                        res[l][1] += 1
                else:
                    temp0, temp1 = iou_single(s, e, s1, e1)
                    if strict == FLAG_STRICT['MEDIUM']:
                        temp0, temp1 = int(temp0), int(temp1)
                    res[l][0] += temp0
                    res[l][1] += temp1
    return res
 
def iou_single(s1, e1, s2, e2):
    smax = max(s1, s2)
    emin = min(e1, e2)
    return max(0, emin - smax) / (e1 - s1) if e1 - s1 > 0 else 0, max(0, emin - smax) / (e2 - s2) if e2 - s2 > 0 else 0
          
def span_f1(arr, strict = FLAG_STRICT['MAX'], labels= None, digits=4):
    all_labels = set()
    dictt = dict()
    for ar in arr:
        text, gt, pred = list(zip(*ar))
        gtSpan = convert_spanformat(list(zip(text, gt)))
        predSpan = convert_spanformat(list(zip(text, pred)))
        dictt = compare_span(predSpan, gtSpan, dictt, strict)

        all_labels.update(list(gtSpan.keys()))
    classfication_rp = dict()
    f1_avg = 0
    if labels is None:
        labels = all_labels
    for i in labels:
        precision = dictt[i][0] / dictt[i][2] if dictt[i][2] > 0 else 0
        recall = dictt[i][1] / dictt[i][3] if dictt[i][3] > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        classfication_rp[i] = {'precision': round(precision, digits), 'recall': round(recall, digits), 'f1': round(f1, digits), 'support': dictt[i][3]}
        f1_avg += f1
    return f1_avg / len(labels), classfication_rp

def merge_subtags(tag_values, tokens, sm, tags_true=None):
    tags = []
    tests = []
    sms = []
    temp = []
    trues = []
    for index in range(len(tokens)):
        if len(tests) == 0:
            if "▁" in tokens[index]:
                tests.append(tokens[index][1:])
            else:
                tests.append(tokens[index])
            temp.append(sm[index])
            if tags_true:
                trues.append(tags_true[index])
        
        elif "▁" in tokens[index] or "</s>" in tokens[index]:
            tests.append(tokens[index][1:])
            sms.append(np.mean(temp, axis=0))
            tags.append(tag_values[np.argmax(sms[-1])])
            temp = [sm[index]]
            if tags_true:
                trues.append(tags_true[index])
        else:
            tests[-1] = tests[-1] + tokens[index]
            temp.append(sm[index])
    sms.append(np.mean(temp, axis=0))
    tags.append(tag_values[np.argmax(sms[-1])])
    return tests, trues, tags, sms

def softmax(arr):
    return np.exp(arr) / sum(np.exp(arr))
