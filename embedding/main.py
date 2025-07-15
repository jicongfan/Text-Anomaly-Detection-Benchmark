# -*- coding: utf-8 -*-
__author__ = 'XF'
__date__ = '2024/09/04'

'''
text embedding.
'''
import time
from embedding.llms_embedding import openai_llms, opensource_llms
from embedding.lms_embedding import LMS_Embedding
from embedding.text_data import get_text_data
from embedding.configs_llms import SAVE_PATH
from tools import new_dir


class TextEmbedding(object):

    def __init__(self, dataset, model, **kw):

        self.dataset = dataset
        self.model = model
        self.kw = kw
        self.time = None
        
        self.train_data, self.train_target, self.test_data, self.test_target = get_text_data(self.dataset)
        
    def embedding(self):
    
        dataset = [self.train_data, self.train_target, self.test_data, self.test_target]
        start = time.time()
        if self.model in ['bert', 'sbert', 'glove_6b']:

            train_path = new_dir(SAVE_PATH, f'{self.dataset}/{self.model}/train')
            test_path = new_dir(SAVE_PATH, f'{self.dataset}/{self.model}/test')
            LMS_Embedding(self.model, *dataset, train_path=train_path, test_path=test_path, device=self.kw['device'], batch_size=self.kw['batch_size'])

        elif self.model in ['text-embedding-3-small', 'text-embedding-3-large', 'text-embedding-ada-002']:

            train_path = new_dir(SAVE_PATH, f'{self.dataset}/{self.kw["model_from"]}/train')
            test_path = new_dir(SAVE_PATH, f'{self.dataset}/{self.kw["model_from"]}/test')
            openai_llms(self.model, *dataset, train_path=train_path, test_path=test_path, request_size=self.kw['batch_size'], 
                        begin_position=self.kw['begin_position'])

        elif self.model in ['Llama3-8b', 'Llama2-7b', 'Mistral-7b']:

            train_path = new_dir(SAVE_PATH, f'{self.dataset}/{self.model}/train')
            test_path = new_dir(SAVE_PATH, f'{self.dataset}/{self.model}/test')
            opensource_llms(self.model, self.kw['ft_llm'], *dataset, train_path=train_path, test_path=test_path, 
                            request_size=self.kw['batch_size'], max_size=self.kw['max_size'], pooling=self.kw['pooling'], 
                            device=self.kw['device'], begin_position=self.kw['begin_position'])
            
        else:
            raise Exception(f'Unknown language model [{self.model}]!')
        self.time = time.time() - start


