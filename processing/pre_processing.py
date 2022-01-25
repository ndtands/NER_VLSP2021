from pyvi import ViTokenizer, ViPosTagger
import re
from sklearn.metrics import *
import string
import unicodedata
import re
import numpy as np
import pickle

def process_unk(tokenizer, sq):
    temp = []
    for i in sq.split():
        if ['[UNK]'] == tokenizer.tokenize(i):
            temp.append(i[0]+i[1:].lower())
        else:
            temp.append(i)
    return ' '.join(temp)


def replace_special_character(text):
  sent_out = ""
  dictt = {'™': ' ', '‘': "'", '®': ' ', '×': ' ', '😀': ' ', '‑': ' - ','́': ' ', '—': ' - ', '̣': ' ', '–': ' - ', '`': "'",\
             '“': '"', '̉': ' ','’': "'", '̃': ' ', '\u200b': ' ', '̀': ' ', '”': '"', '…': '...', '\ufeff': ' ', '″': '"'}
    
  for i in text:
    if i.isalnum() or i in string.punctuation or i == ' ':

        sent_out += i
    elif i in dictt:
        sent_out += dictt[i]
  return sent_out

def preprocessing_text(text):
    sent_out = ""
    stack = []
  
    sents = text.split('\n')
    for index in range(len(sents)):
      sent = sents[index]
      part_sent = ""
      
      if sent.strip() != '':
        for word in sent.split(' '):
          if word.strip() != '':
            word = unicodedata.normalize('NFKC', word)
            word = handle_character(word)
            word = handle_bracket(word)
            word = replace_special_character(word)
            part_sent +=  word + ' '
        
        
        
        sent_out += part_sent + " . "
        sent_out = re.sub(' +', ' ', sent_out)
        stack.append(len(sent_out.split(' '))  - 2)
    return {"sent_out":sent_out.rstrip(), "stack": stack }
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

def handle_character(word):
    char_end = [".", ",", ";", "?", "+", ":" ]
    if word[-1] in char_end:
      word =  word[0:-1] + " " + word[-1]
    return word


def isNotSubword(x, idx, sub = '##'):
    return sub in x[idx] and idx < len(x) - 1 and sub in x[idx+1]

def cutting_subword(X, sub = '##', size=256):
    res_X = []
    punct = '.!?'
    st = 0
    cur = 0
    while (st < len(X)-size):
        flag = True
        for i in range(st+size-1, st-1, -1):
            if X[i] in punct and isNotSubword(X, i, sub):
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
    return res_X



def merge_word(sent, pred_out):
  '''
    :sent: is input sentences (hanlded pre-processing). example: 'pham van manh have email ( pvm26042000@gmail.com ) ....'
    :out : is input of predict, is list tuple. example: [('pham', 'O'), ('van', 'O'), ('manh', 'O'), ('have', 'O'), ('email', 'O'), ('(', 'O'),  ('pvm26042000', 'EMAIL'), ('@', 'EMAIL'),('gmail', 'EMAIL'), ('.', 'EMAIL'),('com', 'EMAIL'),(')', 'O'),('....', 'O')]
  '''
  out_merged = []
  parts = sent.split()

  for index in range(0, len(parts)):
    word = parts[index]

    
    for jndex in range(1, len(pred_out) + 1):
      token = pred_out[0:jndex]
      ws_token, ls_token = list(zip(*token))
      word_token = "".join(ws_token)
      if word_token == word:
        if len(token) == 1:
          out_merged.append(token[0])
        elif len(token) > 1:
          a, b = list(zip(*token))
          word_merged = "".join(a)
          l_merged = decide_label((word_merged, b))
          out_merged.append(l_merged)
        pred_out = pred_out[jndex:]
        break
  return out_merged
