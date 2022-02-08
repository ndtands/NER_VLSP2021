
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
from torch.utils.data import TensorDataset, DataLoader
from modelling import *
from define_name import *


class BertDataLoaderReal:
    def __init__(self,  tokenizer, tag_values, sub, max_len=256, batch_size=32, device='cuda', is_train = False):
        self.tag_values = tag_values
        self.tag2idx = {t: i for i, t in enumerate(self.tag_values)}
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.batch_size = batch_size
        self.device = device
        self.sub = sub


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

        for word in sentence:
            subwords = self.tokenizer.tokenize(word)
            tokenized_sentence.extend(subwords)
      
        return tokenized_sentence

    def add_subword2data(self):
        tokenized_texts = [self.add_subword(sentence) for sentence in self.data]
        return tokenized_texts

    def isNotSubword(self, x, idx, sub = '▁'):
        return sub in x[idx] and idx < len(x) - 1 and sub in x[idx+1]

    def cutting_subword(self, X, sub = '▁'):
    
        res_X = []
        st = 0
        cur = 0
        while (st < len(X)-self.max_len):
            flag = True
            for i in range(st+self.max_len-1, st-1, -1):
                if X[i][-1] in PUNCT_END and self.isNotSubword(X, i, sub):
                    cur = i+1
                    flag = False
                    break
            if flag:
                for i in range(st+self.max_len-1, st-1, -1):
                    if self.isNotSubword(X, i, sub):
                        cur = i+1
                        break
            if st == cur:
                cur += self.max_len
            res_X.append(X[st: cur])
            st = cur
        res_X.append(X[cur:])
        for i in range(len(res_X)):
            res_X[i].insert(0, TOKEN_START)
            res_X[i].append(TOKEN_END)
        return res_X
    def padding_data(self, X_subword):
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
        
        attention_masks = [[float(i != 1.0) for i in ii] for ii in X_padding]
        return X_padding, attention_masks
    def covert2tensor(self, X_padding, attention_masks):
      
        X_tensor = torch.tensor(X_padding).type(torch.LongTensor).to(self.device) 
        masks = torch.tensor(attention_masks).type(torch.LongTensor).to(self.device) 
        return  X_tensor, masks
    
    def create_dataloader(self, data):
        
        X_padding, attention_masks = self.padding_data(data)
        X_tensor, masks = self.covert2tensor(X_padding, attention_masks)

        train_data = TensorDataset(X_tensor, masks)
        train_dataloader = DataLoader(train_data, batch_size = self.batch_size, shuffle=True)

        return train_dataloader