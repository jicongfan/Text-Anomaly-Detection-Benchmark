# -*- coding: utf-8 -*-
__author__ = 'XF'
__date__ = '2022/08/25'

'the default configrations for this project.'

# built-in library
import os.path as osp

# third-party library

# self-defined library

# dir
ROOT_DIR = osp.dirname(osp.abspath(__file__))
DATA_DIR = osp.join('/mnt/data/xiaofeng/data/Text_AD/data')
RESULTS_DIR = osp.join(ROOT_DIR, 'results')
ARGUMENTS_DIR = osp.join(ROOT_DIR, 'arguments')
LOG_DIR = osp.join(ROOT_DIR, 'logs')
EMBEDDING_DIR = osp.join('/mnt/data/xiaofeng/data/Text_AD', 'embedding')
XL_TEMPLATE_PATH = osp.join(RESULTS_DIR, 'results_template.xlsx')
# the maximum contamination rate
CONTAMINATION = 0.1

# datasets used
DATASETS = [
    '20newsgroups',
    'reuters21578',
    'imdb',
    'sst2',
    'dbpedia14',
    'sms_spam',
    'enron',
    'wos'
    
]

# metric

METRIC = [
    'auroc',
    'auprc',
    'f1',
    'acc',
    'fnr',
    'fpr'
]

# dbpedia14_chunk

DBPEDIA14_CHUNK = [
    '50000',
    '100000',
    '150000',
    '200000',
    '250000',
    '300000',
    '350000',
    '400000',
    '450000',
    '500000',
    '550000',
    'end'
]


# max sample size of single class of dbpedia
MAX_SIZE_SINGLE_TRAIN_CLASS_DBPEDIA14 = 40000
MAX_SIZE_SINGLE_TEST_CLASS_DBPEDIA14 = 5000

# class used in DBpedia14
DBPEDIA14_CLASS = [
    0, 1, 2, 3, 4
]


# Anoamly detection algorithm
CLASSIC_ML_ALGORITHMS = [
    'ocsvm',
    'iforest',
    'pca',
    'lof',
    'ecod'
]

AD_ALGORITHMS = [
    'ocsvm',
    'iforest',
    'pca',
    'lof',
    'ecod',
    'ae',
    'dsvdd',
    'dpad'

]

# embedding techniques
EMBEDDING = {

    'cls': 'cls_tokens_tr',
    'avg_emb': 'average_word_embedding',
    'tfidf_emd': 'tfidf_weighted_word_embedding',
    'tfidf-cls': 'tfidf_weighted_word_embedding_concate_cls_embedding',
    'openai-small': 'openai_text-embedding-3-small_embedding',
    'openai-large': 'openai_text-embedding-3-large_embedding',
    'openai-ada': 'openai_text-embedding-ada-002_embedding',
    'meta-llama3-8b-unsup': 'meta_Llama3-8b_mntp-unsup-simcse_weighted_mean_embedding',
    'meta-llama3-8b-sup': 'meta_Llama3-8b_mntp-supervised_weighted_mean_embedding',
    'meta-llama2-7b-unsup': 'meta_Llama2-7b_mntp-unsup-simcse_weighted_mean_embedding',
    'meta-llama2-7b-sup': 'meta_Llama2-7b_mntp-supervised_weighted_mean_embedding',
    'mistral-7b-unsup': 'meta_Mistral-7b_mntp-unsup-simcse_weighted_mean_embedding',
    'mistral-7b-sup': 'meta_Mistral-7b_mntp-supervised_weighted_mean_embedding',

}

MODEL = [
    'glove_6b',
    'bert',
    'Llama2-7b',
    'Llama3-8b',
    'Mistral-7b',
    'openai'

]

FT_MODEL = {
    'glove_6b': [None],
    'bert': [None],
    'openai': ['text-embedding-3-small', 'text-embedding-ada-002', 'text-embedding-3-large'],
    'Llama2-7b': ['mntp', 'mntp-supervised', 'mntp-unsup-simcse'],
    'Llama3-8b': ['mntp', 'mntp-supervised', 'mntp-unsup-simcse'],
    'Mistral-7b': ['mntp', 'mntp-supervised', 'mntp-unsup-simcse'],
}

POOLING = {
    'glove_6b': ['mean'],
    'bert': ['cls_token', 'mean'],
    'openai': [None],
    'Llama2-7b': ['eos_token', 'mean', 'weighted_mean'],
    'Llama3-8b': ['eos_token', 'mean', 'weighted_mean'],
    'Mistral-7b': ['eos_token', 'mean', 'weighted_mean'],
}

# 20newsgroups class
CLASS_20newsgroups = {
    'normal': [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'abnormal': [6],
}

# enron clsss
CLASS_enron = {
    'normal': [0, 1],
    'abnormal': [2]
}

CLASS_imdb = {
    'normal': [0],
    'abnormal': [1]
}

CLASS_reuters21578 = {

    'normal': [0, 1],
    'abnormal': [2]

}


CLASS_sst2 = {
    
    'normal': [1],
    'abnormal': [0]
}

CLASS_wos = {

    'normal': [0, 1, 3, 6, 4, 5],
    'abnormal': [2]
}

XL_STRUCTURE = {
    'Llama2-7b_mntp_eos_token': {
        'auroc': 'C4',
        'auprc': 'C5',
        'f1': 'C6',
        'acc': 'C7',
        'fnr': 'C8',
        'fpr': 'C9'
    },
    'Llama2-7b_mntp_mean': {
        'auroc': 'D4',
        'auprc': 'D5',
        'f1': 'D6',
        'acc': 'D7',
        'fnr': 'D8',
        'fpr': 'D9'
    },
    'Llama2-7b_mntp_weighted_mean': {
        'auroc': 'E4',
        'auprc': 'E5',
        'f1': 'E6',
        'acc': 'E7',
        'fnr': 'E8',
        'fpr': 'E9'
    },

    'Llama2-7b_mntp-unsup-simcse_eos_token': {
        'auroc': 'F4',
        'auprc': 'F5',
        'f1': 'F6',
        'acc': 'F7',
        'fnr': 'F8',
        'fpr': 'F9'
    },
    'Llama2-7b_mntp-unsup-simcse_mean': {
        'auroc': 'G4',
        'auprc': 'G5',
        'f1': 'G6',
        'acc': 'G7',
        'fnr': 'G8',
        'fpr': 'G9'
    },
    'Llama2-7b_mntp-unsup-simcse_weighted_mean': {
        'auroc': 'H4',
        'auprc': 'H5',
        'f1': 'H6',
        'acc': 'H7',
        'fnr': 'H8',
        'fpr': 'H9'
    },

    'Llama2-7b_mntp-supervised_eos_token': {
        'auroc': 'I4',
        'auprc': 'I5',
        'f1': 'I6',
        'acc': 'I7',
        'fnr': 'I8',
        'fpr': 'I9'
    },
    'Llama2-7b_mntp-supervised_mean': {
        'auroc': 'J4',
        'auprc': 'J5',
        'f1': 'J6',
        'acc': 'J7',
        'fnr': 'J8',
        'fpr': 'J9'
    },
    'Llama2-7b_mntp-supervised_weighted_mean': {
        'auroc': 'K4',
        'auprc': 'K5',
        'f1': 'K6',
        'acc': 'K7',
        'fnr': 'K8',
        'fpr': 'K9'
    },

    'bert_None_cls_token': {
        'auroc': 'M4',
        'auprc': 'M5',
        'f1': 'M6',
        'acc': 'M7',
        'fnr': 'M8',
        'fpr': 'M9'
    },
    'bert_None_mean': {
        'auroc': 'N4',
        'auprc': 'N5',
        'f1': 'N6',
        'acc': 'N7',
        'fnr': 'N8',
        'fpr': 'N9'
    },
    'openai_text-embedding-3-small_None': {
        'auroc': 'M13',
        'auprc': 'M14',
        'f1': 'M15',
        'acc': 'M16',
        'fnr': 'M17',
        'fpr': 'M18'
    },
    'openai_text-embedding-ada-002_None': {
        'auroc': 'N13',
        'auprc': 'N14',
        'f1': 'N15',
        'acc': 'N16',
        'fnr': 'N17',
        'fpr': 'N18'
    },
    'openai_text-embedding-3-large_None': {
        'auroc': 'O13',
        'auprc': 'O14',
        'f1': 'O15',
        'acc': 'O16',
        'fnr': 'O17',
        'fpr': 'O18'
    },
}

# other
begining_line = '=============================== Begin ======================================='
ending_line =   '================================ End ========================================'

