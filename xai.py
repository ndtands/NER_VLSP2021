from processing.pre_processing import preprocessing_text
from eli5.lime import TextExplainer
from eli5.lime.samplers import MaskingTextSampler
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def get_predict_function(ner,word_index):
    def predict_func(texts):
     
        out = []
        for  sq in texts:
            
            _, logit = ner.predict(sq, interpret = True)
            logit_ = [norm_logit(i.tolist()) for i in logit]
            out.append(logit_)
        
        return np.array(out)[:,word_index,:]
    return predict_func


def norm_logit(logit):
    logit = np.array(logit)
    logit /= logit.sum()
    return logit

def xai_lime(ner, text, id_word):
    tag_values = ner.tag_values
    preprocessing_texts = preprocessing_text(text)
    text = preprocessing_texts["sent_out"]
    # stack = preprocessing_texts["stack"]
    # temp = ""
    
    sampler = MaskingTextSampler(
        replacement="<_s>",
        max_replace=0.5,
        token_pattern= r"[^\s]+",
        bow=False,
        random_state=0
    )

    te = TextExplainer(
        sampler=sampler,
        position_dependent=False,
        random_state=20,
        token_pattern=r"[^\s]+",
        
    )
    
    predict_func = get_predict_function(ner,id_word)
    te.n_samples = 200

    te.fit(text, predict_func)
    parts = text.split(" ")

    idx = {}
    for i in range(len(parts)):
        window = 2
        keyword = parts[i].lower()
        idx[i] = [te.vec_.vocabulary_[keyword]]
        if i == 0:
            keyword_next = " ".join(parts[i:i + window]).lower()
            idx[i].append(te.vec_.vocabulary_[keyword_next])

        elif i > 0 and i < len(parts) -1:
            keyword_prev = " ".join(parts[i -1:i + 1]).lower()
            idx[i].append(te.vec_.vocabulary_[keyword_next])

            keyword_next = " ".join(parts[i:i + window]).lower()
            idx[i].append(te.vec_.vocabulary_[keyword_next])
        elif  i == len(parts) -1:

            keyword_prev = " ".join(parts[i -1:i + 1]).lower()
            idx[i].append(te.vec_.vocabulary_[keyword_next])

                
        classes = te.clf_.classes_
        rss = {}
        rss['sent'] = text

        for indx in range(len(te.clf_.coef_)):
            class_n = classes[indx]
            for weight_label in te.clf_.coef_:
                rs = []
                for key in idx:
                    indexs = idx[key]
                    L = te.clf_.coef_[indx]
                    T = [L[i] for i in indexs]
                    rs.append(sum(T))
    
                proba = te.explain_prediction().targets[indx].proba
                class_name = te.explain_prediction().targets[indx].target
    
                if len(classes) == 2 and class_n == class_name:
                    rs = [-i for i in rs]
    
                rss[tag_values[class_name]] = {
                    "proba": proba,
                    "scores": rs
                }
    return rss