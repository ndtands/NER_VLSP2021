from tqdm.notebook import tqdm
import torch.nn as nn
import torch
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from transformers import AutoConfig, AutoModel, AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import pickle
from transformers import AdamW, get_linear_schedule_with_warmup

import torch.nn.functional as F
log_soft = F.log_softmax
import utils
import data_processing
from dataloader import BertDataLoader

import torch
from tqdm.notebook import tqdm
from tqdm import trange
import numpy as np
from define_name import *
from processing import pre_processing

class NER(nn.Module):
    def __init__(self, config):
        super(NER, self).__init__()
        self.hyper_parameter = config['hyper_parameter']
        self.tag_values = config['tag_values']
        self.max_len = config['max_len']
        self.device = config['device']
        self.batch_size = config['batch_size']
        self.tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'], do_lower_case=False,use_fast=False)
        self.config_pretrain = AutoConfig.from_pretrained(config['config'], output_hidden_states=True)
        self.base_model = AutoModel.from_pretrained(config['model'], config=self.config_pretrain, add_pooling_layer=True)
        self.sub = config['sub']
        self.model = BaseBertSoftmax(self.base_model, self.hyper_parameter['linear_dropout'], num_labels=len(self.tag_values))
        self.model.to(self.device)
        self.dataloader = BertDataLoader(self.tokenizer, self.tag_values, self.sub, self.max_len, self.batch_size, self.device)

    def predict(self, texts, interpret = False):
        preprocessing_texts = pre_processing.preprocessing_text(texts)
        texts = preprocessing_texts["sent_out"]
        stack = preprocessing_texts["stack"]
        subwords = self.tokenizer.tokenize(texts)
        sub_cut = utils.cutting_subword(subwords, sub = self.sub, size = self.max_len-2)
        tags_out = []
        words_out = []
        probs_out = []
        self.model.eval()
        for sub in sub_cut:
            input_ids = pad_sequences([self.tokenizer.convert_tokens_to_ids(sub)],
                                        maxlen=self.max_len, dtype="long", value=self.tokenizer.pad_token_id,
                                        truncating="post", padding="post")
            input_ids_tensor = torch.tensor(input_ids).type(torch.LongTensor).to(self.device)
            input_mask = [[float(i != self.tokenizer.pad_token_id) for i in ii] for ii in input_ids]
            input_mask_tensor = torch.tensor(input_mask).type(torch.LongTensor).to(self.device) 
            with torch.no_grad():
                outputs = self.model.forward_custom(input_ids_tensor, input_mask_tensor)
            logits = outputs[0].detach().cpu().numpy()
            len_subword = sum(input_ids[0] != self.tokenizer.pad_token_id)
            predict = np.argmax(logits, axis=2)[0][:len_subword]
            sm = [(utils.softmax(logits[0,i])) for i in range(len_subword)]
            tags_predict = [self.tag_values[i]  for i in  predict]
            tests, tags, probs = utils.merge_subtags_test(sub, tags_predict, sm)
            words_out += tests[1:-1]
            tags_out += tags[1:-1]
            probs_out += probs[1:-1]
        out1 = [(w,t,p) for w,t,p in zip(words_out,tags_out, probs_out)]
        out = pre_processing.span_cluster(out1)
        texts = " ".join([word for (word, _) in out])
        result = pre_processing.post_processing(texts, stack, out)
        
        if interpret:
            return result, probs_out
        else:
            return result

    def evaluate(self, dataloader, strict = FLAG_STRICT['MAX']):
        self.model.eval()
        eval_loss = 0
        out = []
        for batch in tqdm(dataloader, desc='Testing Process:'):
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():
                outputs = self.model.forward_custom(b_input_ids, b_input_mask, b_labels)
            eval_loss += outputs[0].mean().item()
            
            out += self.batch_processing(batch, outputs)
        eval_loss = eval_loss / len(dataloader)
        f1, result = utils.span_f1(arr = out, strict = strict, digits=4)
        return eval_loss, f1, result, out

    def train_model(self, train_loader, dev_loader, PATH, strict):
        optimizer = self.get_optimizer()
        total_steps = len(train_loader) * self.hyper_parameter['epochs']
        scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=int(total_steps/10), num_training_steps=total_steps)

        ## Store the average loss after each epoch so we can plot them.
        patience = self.hyper_parameter['patience']
        train_loss_values, valid_loss_values = [], []
        f1_max = 0
        f1_train_list, f1_dev_list = [], []
        history = {}
        epochs_no_improve = 0
        best_epoch = 0
        for epoch in trange(self.hyper_parameter['epochs'], desc="Epoch"):
            # ==========================================#
            #               Training                    #
            # ==========================================#
            self.model.train()
            train_loss = 0
            out = []
            for batch in tqdm(train_loader, desc='Traning Process:'):
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                self.model.zero_grad()
                outputs = self.model.forward_custom(input_ids=b_input_ids, attention_mask=b_input_mask, labels=b_labels, head_mask=None)
                loss = outputs[0]
                loss.backward()
                train_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(parameters = self.model.parameters(), max_norm=self.hyper_parameter['max_grad_norm'])
                # update parameters
                optimizer.step()
                # Update the learning rate.
                scheduler.step()
                out += self.batch_processing(batch, outputs)

            avg_train_loss = train_loss / len(train_loader)
            train_loss_values.append(avg_train_loss)
            f1_train, _ = utils.span_f1(arr = out, strict=strict, digits=4)
            # ========================================
            #               Validation
            # ========================================
            eval_loss, f1_dev, result_dev, out = self.evaluate(dataloader=dev_loader, strict=strict)
            valid_loss_values.append(eval_loss)
            print(utils.show_span_f1(result_dev))
            print(f'<===============> loss: {avg_train_loss:.4f}   f1_span{strict}: {f1_train:.4f}   val_loss: {eval_loss:.4f}   val_f1_span{strict}: {f1_dev:.4f}')
            f1_train_list.append(f1_train)
            f1_dev_list.append(f1_dev)

            if f1_dev > f1_max:
                print(f'f1_score improved from: {f1_max:.4f} to {f1_dev:.4f}')
                print(f'Best model saved to {PATH}')
                f1_max = f1_dev
                torch.save(self.model.state_dict(), PATH)
                epochs_no_improve = 0
                best_epoch = epoch
            else:
                print(f'f1_score dont improve from: {f1_max:.4f} to {f1_dev:.4f}')
                epochs_no_improve += 1
                if epochs_no_improve < patience:
                    print(f'EarlyStopping count: {epochs_no_improve}/{patience}')
                else:
                    print(f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with f1_score: {f1_max:.4f}')
                    break
        history['train_loss_values'] = train_loss_values
        history['valid_loss_values'] = valid_loss_values
        history['f1_train_list'] = f1_train_list
        history['f1_dev_list'] = f1_dev_list
        self.model.load_state_dict(torch.load(PATH), strict=False)
        return history

    def get_optimizer(self):
        param_optimizer1 = list(self.model.named_parameters())[:self.hyper_parameter['num_layer']]
        param_optimizer2 = list(self.model.named_parameters())[self.hyper_parameter['num_layer']:]
        no_decay = ['bias', 'LayerNorm.weight'] #['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer1 if not any(nd in n for nd in no_decay)],
            'weight_decay_rate': self.hyper_parameter['bert_weight_decay']},
            {'params': [p for n, p in param_optimizer1 if any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.0},
            
            {'params': [p for n, p in param_optimizer2 if not any(nd in n for nd in no_decay)],
            'weight_decay_rate': self.hyper_parameter['softmax_weight_decay'],
            'lr':  self.hyper_parameter['softmax_lr']},
            {'params': [p for n, p in param_optimizer2 if any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.0,
            'lr': self.hyper_parameter['softmax_lr']},
        ]
        optimizer = AdamW(
                    optimizer_grouped_parameters,
                    lr=self.hyper_parameter['bert_lr'],
                    eps = 1e-8)
        return optimizer

    def batch_processing(self, batch, outputs):
        b_input_ids, b_input_mask, b_labels = batch
        label_ids = b_labels.to('cpu').numpy().tolist()
        b_input_ids = b_input_ids.to('cpu').numpy().tolist()
        predict_labels = []
        logits = outputs[1].detach().cpu().numpy()
        out = []
        for predicts in np.argmax(logits, axis=2):
            predict_labels.append(predicts)
        count_idx = 0
        for b_input_id, preds, labels in zip(b_input_ids, predict_labels, label_ids):
            n = sum(np.array(b_input_id) != self.tokenizer.pad_token_id)
            sm = [(utils.softmax(logits[count_idx,ii])) for ii in range(n)]
            count_idx += 1
            tokens = self.tokenizer.convert_ids_to_tokens(b_input_id)[:n]
            labels = [self.tag_values[i] for i in labels][:n]
            preds = [self.tag_values[i] for i in preds]
            token_new, label_new, pred_new, _ = utils.merge_subtags_train(tag_values=self.tag_values, tags_true=labels, tokens=tokens, sm=sm)
            out.append(list(zip(token_new, label_new, pred_new)))   
        return out

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')), strict=False)
        # self.model = self.model.to(self.device)
        self.model.to(self.device)


#############################################################################################################################
class BaseBertSoftmax(nn.Module):
    def __init__(self, model, drop_out , num_labels ):
        super(BaseBertSoftmax, self).__init__()
        self.num_labels = num_labels
        self.model = model
        self.dropout = nn.Dropout(drop_out)
        self.classifier = nn.Linear(self.model.config.hidden_size, num_labels)
            
    def forward_custom(self, input_ids, attention_mask=None,
                        labels=None, head_mask=None):
        outputs = self.model(input_ids = input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs[0]) #no concat
        
        logits = self.classifier(sequence_output) # bsz, seq_len, num_labels
        loss_fct = nn.CrossEntropyLoss()
        outputs = (logits,)
        if labels is not None:
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs 

#########################################################
