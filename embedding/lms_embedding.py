__author__ = 'XF'
__date__ = '2024/09/04'

'''
Text embedding using Language models, such as [bert, sequence_bert, ...]

'''

import torch
from tqdm import tqdm
import numpy as np
from os import path as osp
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
from os import path as osp
from tools import obj_save
from embedding.configs_llms import LM_CACHE_DIR, CHUNK_EMBEDDING
from torchnlp.word_to_vector import GloVe
from torch.utils.data import DataLoader
from .data_preprocess import clean_text



# 1. CLS Token
# 2. average word embedding
# 3. TF-IDF weighted average word embedding
# 4. Sentence BERT
# 5. Text Embedding from LLMs

class LMS_Embedding(object):

    def __init__(self, model, train_data, train_target, test_data, test_target, train_path, test_path, max_length=512, device=None, batch_size=1, begin_position=0):
        
        self.model = model
        self.train_path = train_path
        self.test_path = test_path
        self.max_length = max_length
        self.batch_size = batch_size
        self.begin_position = begin_position
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.train_data = train_data
        self.train_target = train_target
        self.test_data = test_data
        self.test_target = test_target

        self.train_embedding = None
        self.train_labels = None
        self.train_cls_embedding = None


        self.test_embedding = None
        self.test_labels = None
        self.test_cls_embedding = None
        self.embedding()
    
    def embedding(self):

        if self.model == 'bert':
            self.bert()
        elif self.model == 'glove_6b':
            self.glove()
        
        # cls embedding
        # self.save_embedding()
        # average word embedding
        # self.average_embedding()
        # tf-idf weighted word embedding
        # self.tf_idf_weighted_embedding()

    def bert(self):

        """
        using pretrained bert model to get word embedding and cls token.
        """
        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', cache_dir=LM_CACHE_DIR['bert'])
        model = BertModel.from_pretrained('bert-large-uncased', cache_dir=LM_CACHE_DIR['bert'])
        
        
        model.to(self.device)
        for _, (text_data, text_target, save_path, bp) in enumerate(zip([self.train_data, self.test_data], [self.train_target, self.test_target], [self.train_path, self.test_path], [self.begin_position, 0])):
            # word_embeddings = []
            cls_tokens = None
            mean_embedding = None
            targets = []
            position = bp
            if position > len(text_data):
                continue
            print(f'Data Size: {len(text_data)}')
            for begin_pos in tqdm(range(0, len(text_data), self.batch_size)):
                if self.batch_size < len(text_data) - begin_pos:
                    end_pos = begin_pos + self.batch_size
                else:
                    end_pos = -1
                
                text = text_data[begin_pos:end_pos]
                label = text_target[begin_pos:end_pos]
                position += self.batch_size
                # encoding text，get inputs of bert
                inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)
                inputs = inputs.to(self.device)
                # print(f'inputs: {type(inputs)}')
                # get the output of model
                model.eval()
                with torch.no_grad():
                    outputs = model(**inputs)
                    
                # get the hidden states of last hidden layer
                hidden_states = outputs.last_hidden_state
                # print(f'hidden states: {hidden_states.shape}')
                
                # get the embedding of cls token
                cls_token = hidden_states[:, 0, :].squeeze(0).detach().cpu().numpy()
                if cls_tokens is None:
                    cls_tokens = cls_token
                else:
                    cls_tokens = np.concatenate((cls_tokens, cls_token))
                
                # get the embedding of all tokens
                token_embeddings = hidden_states[:,1:-1,:].squeeze(0).detach().cpu().numpy()
                # print(f'token_embeddings:{token_embeddings.shape}')
                sentence_embedding = np.nanmean(token_embeddings, keepdims=False, axis=1)
                # print(f'sentence_embedding: {sentence_embedding.shape}')
                if mean_embedding is None:
                    mean_embedding = sentence_embedding
                else:
                    mean_embedding = np.concatenate((mean_embedding, sentence_embedding))
                # word_embeddings.append(token_embeddings)
                targets.extend(label)

                # if position % CHUNK_EMBEDDING == 0:
                #     obj_save(osp.join(save_path, f'mean_embeddings_{position}.npy'), mean_embedding)
                #     obj_save(osp.join(save_path, f'cls_token_embeddings_{position}.npy'), cls_tokens)
                #     obj_save(osp.join(save_path, f'embedding_labels_{position}.npy'), targets)
                #     print(f'Save embedding to [{save_path}].')
                #     mean_embedding = None
                #     cls_tokens = None
                #     targets = []


            print(f'Embedding info =====================')
            print(f'mean embedding: {mean_embedding.shape}')
            print(f'cls token embedding: {cls_tokens.shape}')
            print(f'labels: {len(targets)}')
            # obj_save(osp.join(save_path, f'mean_embeddings_end.npy'), mean_embedding)
            # obj_save(osp.join(save_path, f'cls_token_embeddings_end.npy'), cls_tokens)
            # obj_save(osp.join(save_path, f'embedding_labels_end.npy'), targets)
            obj_save(osp.join(save_path, f'mean_embeddings.npy'), mean_embedding)
            obj_save(osp.join(save_path, f'cls_token_embeddings.npy'), cls_tokens)
            obj_save(osp.join(save_path, f'embedding_labels.npy'), targets)
            print(f'Save embedding to [{save_path}].')


    def glove(self, embedding_size=300):

        # data clean
        valid_train_data = []
        valid_train_label = []
        valid_test_data = []
        valid_test_label = []

        for text, label in zip(self.train_data, self.train_target):
            valid_text = clean_text(text, rm_numbers=True)
            if len(valid_text) > 1 and len(valid_text.split(' ')) > 1:
                valid_train_data.append(valid_text)
                valid_train_label.append(label)

        for text, label in zip(self.test_data, self.test_target):
            valid_text = clean_text(text, rm_numbers=True)
            if len(valid_text) > 1 and len(valid_text.split(' ')) > 1:
                valid_test_data.append(valid_text)
                valid_test_label.append(label)
        
        print(f'valid train data: {len(valid_train_data)}')
        print(f'valid test data: {len(valid_test_data)}')

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=LM_CACHE_DIR['bert'])
        glove = GloVe(name='6B', dim=embedding_size, cache=LM_CACHE_DIR['glove_6b']) 

        for _, (text_data, text_target, save_path, bp) in enumerate(zip([valid_train_data, valid_test_data], [valid_train_label, valid_test_label], [self.train_path, self.test_path], [self.begin_position, 0])):
            mean_embedding = None
            targets = []
            position = bp
            if position > len(text_data):
                continue
            print(f'Data Size: {len(text_data)}')
            for begin_pos in tqdm(range(0, len(text_data), self.batch_size)):
                if self.batch_size < len(text_data) - begin_pos:
                    end_pos = begin_pos + self.batch_size
                else:
                    end_pos = -1
                
                text = text_data[begin_pos:end_pos]
                label = text_target[begin_pos:end_pos]
                position += self.batch_size
                # encoding text，get inputs of bert
                # tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)
                print(f'[{_ + 1}]: {text}')
                # tokens = tokenizer.tokenize(text)
                tokens = tokenizer(text, return_tensors='pt',)
                print(type(tokens))
                for em in tokens:
                    print(em)
                print(f'token: {tokens.shape}')

                valid_embeddings = [
                    glove[token].numpy() for token in tokens if token in glove
                ]
                
                # If no tokens have embeddings, return a zero vector
                if not valid_embeddings:
                    valid_embeddings =  np.zeros(embedding_size)
                
                print(valid_embeddings.shape)
                # Apply mean pooling over the embeddings
                sentence_embedding = np.nanmean(valid_embeddings, axis=0)
             

                if mean_embedding is None:
                    mean_embedding = sentence_embedding
                else:
                    mean_embedding = np.concatenate((mean_embedding, sentence_embedding))
                # word_embeddings.append(token_embeddings)
                targets.extend(label)

                if position % CHUNK_EMBEDDING == 0:
                    obj_save(osp.join(save_path, f'mean_embeddings_{position}.npy'), mean_embedding)
                    obj_save(osp.join(save_path, f'embedding_labels_{position}.npy'), targets)
                    print(f'Save embedding to [{save_path}].')
                    mean_embedding = None
                    targets = []


            print(f'Embedding info =====================')
            print(f'mean embedding: {mean_embedding.shape}')
            print(f'labels: {len(targets)}')
            obj_save(osp.join(save_path, f'mean_embeddings_end.npy'), mean_embedding)
            obj_save(osp.join(save_path, f'embedding_labels_end.npy'), targets)
            print(f'Save embedding to [{save_path}].')


    def cls_token(self):

        print(f'Embedding info [train data]========================')
        print(f'CLS tokens embedding: {self.train_cls_embedding.shape}')
        print(f'Targets size: {len(self.train_labels)}')
        print(f'Embedding info [test data]========================')
        print(f'CLS tokens embedding: {self.test_cls_embedding.shape}')
        print(f'Targets size: {len(self.test_labels)}')

        # save train data embedding
        obj_save(osp.join(self.train_path, f'word_embeddings.npy'), self.train_embedding)
        obj_save(osp.join(self.train_path, f'cls_token_embeddings.npy'), self.train_cls_embedding)
        obj_save(osp.join(self.train_path, f'cls_embedding_labels.npy'), self.train_labels)

        # save test data embedding
        obj_save(osp.join(self.test_path, f'word_embeddings.npy'), self.test_embedding)
        obj_save(osp.join(self.test_path, f'cls_token_embeddings.npy'), self.test_cls_embedding)
        obj_save(osp.join(self.test_path, f'cls_embedding_labels.npy'), self.test_labels)


    def average_embedding(self):

        print(f'train embedding: {len(self.train_embedding)}')
        print(f'test embedding: {len(self.test_embedding)}')
        train_average_embedding = None
        test_average_embedding = None
        train_lab = []
        test_lab = []
        # train data embedding =======================================================================
        for i, (word_embeddings, label) in enumerate(zip(self.train_embedding, self.train_labels)):
            
            sentence_embedding = np.nanmean(word_embeddings, axis=0).reshape((1, -1))
            if np.any(np.isnan(sentence_embedding)):
                print(f'[{i + 1}]: nan')
                continue
            # print(f'sentence embedding: {sentence_embedding.shape}')
            if train_average_embedding is None:
                train_average_embedding = sentence_embedding
            else:
                train_average_embedding = np.concatenate((train_average_embedding, sentence_embedding))
            train_lab.append(label)
        # test data embedding =======================================================================
        for i, (word_embeddings, label) in enumerate(zip(self.test_embedding, self.test_labels)):
            
            sentence_embedding = np.nanmean(word_embeddings, axis=0).reshape((1, -1))
            if np.any(np.isnan(sentence_embedding)):
                print(f'[{i + 1}]: nan')
                continue
            if test_average_embedding is None:
                test_average_embedding = sentence_embedding
            else:
                test_average_embedding = np.concatenate((test_average_embedding, sentence_embedding))
            test_lab.append(label)

        print(f'train sentence embedding: {train_average_embedding.shape}')
        print(f'train label: {len(train_lab)}')
        print(f'test sentence embedding: {test_average_embedding.shape}')
        print(f'test label: {len(test_lab)}')
        obj_save(osp.join(self.train_path, f'average_word_embeddings.npy'), train_average_embedding)
        obj_save(osp.join(self.train_path, f'average_word_embedding_labels.npy'), train_lab)
        obj_save(osp.join(self.test_path, f'average_word_embeddings.npy'), test_average_embedding)
        obj_save(osp.join(self.test_path, f'average_word_embedding_labels.npy'), test_lab)

