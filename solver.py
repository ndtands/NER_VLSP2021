from model.model import NerModel
from abc import ABC, abstractmethod
from utils.text.pre_process import preprocessing_text
from utils.text.post_process import post_processing
from utils.text.reference_process import cutting_subword, merge_subtags_3column

class BaseSolver(ABC):
    def __init__(self):
        self.setup()

    @abstractmethod
    def setup(self):
        raise NotImplementedError
    
    @abstractmethod
    def solve(self):
        raise NotImplementedError

class NerSolver(BaseSolver):
    def __init__(self, config):
        self.config = config
        super(NerSolver, self).__init__()
        
    def setup(self):
        self.model = NerModel(self.config)
    
    def solve(self, texts):
       
        texts_processed = preprocessing_text(texts)
        texts = texts_processed["sent_out"]
        stack = texts_processed["stack"]

        subwords = self.model.tokenizer.tokenize(texts)
        subs_cut = cutting_subword(subwords, sub = self.model.sub, size = self.model.max_len-2)

        results = self.model.predict(subs_cut = subs_cut)
        
        words_out = []
        tags_out = []

        for i in range(len(results)):

            token_new, pred_new = merge_subtags_3column(tokens=subs_cut[i], tags_predict=results[i])
            words_out += token_new[1:-1]
            tags_out += pred_new[1:-1]
   
        output_process = [(w,self.model.label_classes[t]) for w,t in zip(words_out,tags_out)]
        texts = " ".join([word for (word, _) in output_process])
    
        result_final = post_processing(texts, stack, output_process)
        return result_final
