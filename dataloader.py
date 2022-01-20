import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from modelling import *
from define_name import *


class BertDataLoader:
    def __init__(self,  tokenizer, tag_values, sub, max_len=256, batch_size=32, device='cuda', is_train = False):
        self.tag_values = tag_values
        self.tag2idx = {t: i for i, t in enumerate(self.tag_values)}
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.batch_size = batch_size
        self.device = device
        self.sub = sub

    def check_label(self):
        labels = set()
        for sq in self.data:
            for sample in sq:
                labels.add(sample[1])
        return labels

    def isNotSubword(self, x, idx, sub = '▁'):
        return sub in x[idx] and idx < len(x) - 1 and sub in x[idx+1]
    
    def cutting_subword(self, X, y, punct = PUNCT_END):
        res_X, res_y = [], []
        st = 0
        cur = 0
        while (st < len(X)-self.max_len-2):
            flag = True
            for i in range(st+self.max_len-2-1, st-1, -1):
                if X[i] in punct and y[i] == LABEL_O:
                    cur = i+1
                    flag = False
                    break
            if flag:
                for i in range(st+self.max_len-2-1, st-1, -1):
                    if self.isNotSubword(X, i, sub = self.sub):
                        cur = i+1
                        if y[i] == LABEL_O:
                            cur = i+1
                            break
            if st == cur:
                cur += self.max_len-2

            res_X.append(X[st: cur])
            res_y.append(y[st: cur])
            st = cur
        res_X.append(X[cur:])
        res_y.append(y[cur:])
        for i in range(len(res_X)):
            res_X[i].insert(0, TOKEN_START)
            res_X[i].append(TOKEN_END)
            res_y[i].insert(0, LABEL_PAD)
            res_y[i].append(LABEL_PAD)
        return res_X, res_y
    
    def add_subword(self, sentence):
        '''
        input:
            sentence = ['Phạm', 'Văn', 'Mạnh']
            text_labels = ['B-PER', 'I-PER','I-PER']

        output: 
            ['Phạm', 'Văn', 'M', '##ạnh'],
            ['B-PER', 'I-PER', 'I-PER', 'I-PER']
        '''
        tokenized_sentence = []
        labels = []
        for word, label in sentence:
            subwords = self.tokenizer.tokenize(word)
            tokenized_sentence.extend(subwords)
            labels.extend([label] * len(subwords))
        return tokenized_sentence, labels

    def add_subword2data(self):
        tokenized_texts_and_labels = [self.add_subword(sentence) for sentence in self.data]
        return list(zip(*tokenized_texts_and_labels))
    
    def padding_data(self, X_subword, y_subword):
        '''
        input:
            X = [['Phạm', 'Văn', 'M', '##ạnh',..],....]
            Y = [['B-PER', 'I-PER','I-PER','I-PER',..],...]
        output: 
        [[10,20,30,40,0,0,0,0,0,0,0,0...],...],
        [[1, 2,3,4,5,5,5,5,5,5,5,5,5,...],...]
        '''
        X_padding = pad_sequences([self.tokenizer.convert_tokens_to_ids(txt) for txt in X_subword],
                          maxlen=self.max_len, dtype="long", value=self.tokenizer.pad_token_id,
                          truncating="post", padding="post")
        y_padding = pad_sequences([[self.tag2idx.get(l) for l in lab] for lab in y_subword],
                        maxlen=self.max_len, value=self.tag2idx[LABEL_PAD], padding="post",
                        dtype="long", truncating="post")
        attention_masks = [[float(i != 1.0) for i in ii] for ii in X_padding]
        return X_padding, y_padding, attention_masks
     
    def covert2tensor(self, X_padding, Y_padding, attention_masks, is_train):
        if is_train:
            X_tensor = torch.tensor(X_padding).to(self.device) 
            y_tensor = torch.tensor(Y_padding).to(self.device) 
            masks = torch.tensor(attention_masks).to(self.device)  
        else:
            X_tensor = torch.tensor(X_padding).type(torch.LongTensor).to(self.device) 
            y_tensor = torch.tensor(Y_padding).type(torch.LongTensor).to(self.device) 
            masks = torch.tensor(attention_masks).type(torch.LongTensor).to(self.device) 
        return  X_tensor, y_tensor, masks

    def create_dataloader(self, dir, is_train):
        with open(dir ,'rb') as f:
            self.data = pickle.load(f)
        labels = self.check_label()
        X_subword, y_subword = self.add_subword2data()
        X_subword_at, y_subword_at = [], []
        for i in range(len(X_subword)):
            res_x, res_y = self.cutting_subword(X_subword[i], y_subword[i])
            X_subword_at += res_x
            y_subword_at += res_y
        X_padding, y_padding, attention_masks = self.padding_data(X_subword_at, y_subword_at)
        X_tensor,y_tensor, masks = self.covert2tensor(X_padding, y_padding, attention_masks,is_train)
        train_data = TensorDataset(X_tensor, masks, y_tensor)
        train_dataloader = DataLoader(train_data, batch_size = self.batch_size, shuffle=True)
        return train_dataloader, labels