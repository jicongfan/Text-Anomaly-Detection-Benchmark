__author__ = 'XF'
__date__ = '2024/09/13'

'''
Preprocess the raw text data.
'''

import sys
import re
import torch
import numpy as np
import string
import click
import nltk
import copy
from os import path as osp
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import reuters
from nltk import word_tokenize
from torchnlp.datasets import imdb_dataset
from datasets import load_dataset
# from torchnlp.datasets.dataset import Dataset

ROOT_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(ROOT_DIR)
from configs import DATASETS, DATA_DIR
from tools import obj_save, new_dir, obj_load

@click.command()
@click.option('--dataset', type=str, default=click.Choice([DATASETS]))
@click.option('--clean', type=bool, default=True)
@click.option('--max_length', type=int, default=8191)
@click.option('--min_length', type=int, default=0)

def main(dataset, clean, max_length, min_length):

    print(f'Preprocess dataset: [{dataset}]')

    data_path = osp.join(DATA_DIR, dataset)
    print(f'Original data info ==========================')
    if dataset == '20newsgroups':
        train_data, train_label, test_data, test_label = newsgroups20_dataset(data_path)
    elif dataset == 'reuters21578':
        train_data, train_label, test_data, test_label = reuters_21578(data_path)
    elif dataset == 'imdb':
        train_data, train_label, test_data, test_label = imdb(data_path)
    elif dataset == 'sst2':
        train_data, train_label, test_data, test_label = sst2(data_path)
    elif dataset == 'dbpedia14':
        train_data, train_label, test_data, test_label = dbpedia14(data_path)
    elif dataset == 'sms_spam':
        train_data, train_label, test_data, test_label = sms_spam(data_path)
    elif dataset == 'enron':
        train_data, train_label, test_data, test_label = enron_email(data_path)
    elif dataset == 'wos':
        train_data, train_label, test_data, test_label = wos(data_path)
    else:
        raise Exception(f'Unknown dataset [{dataset}].')

    # clean raw data
    if clean:    
        clean_train_data = []
        clean_test_data = []
        clean_train_label = []
        clean_test_label = []

        for text, label in zip(train_data, train_label):
            c_text = clean_text(text)
            if len(c_text) > min_length:
                clean_train_data.append(c_text)
                clean_train_label.append(label)
        
        for text, label in zip(test_data, test_label):
            c_text = clean_text(text)
            if len(c_text) > min_length:
                clean_test_data.append(clean_text(text))
                clean_test_label.append(label)
    else:
        clean_train_data = train_data
        clean_train_label = train_label
        clean_test_data = test_data
        clean_test_label = test_label

    # check max sentence length
    valid_train_data, valid_train_label = check_token_number(clean_train_data, clean_train_label, max_length=max_length)
    valid_test_data, valid_test_label = check_token_number(clean_test_data, clean_test_label, max_length=max_length)

    print(f'Preprocessed data info ==============================')
    print(f'train data: {len(valid_train_data)}')
    print(f'train label: {len(valid_train_label)}')
    print(f'test data: {len(valid_test_data)}')
    print(f'test label: {len(valid_test_label)}')


    # save data
    save_dir = new_dir(data_path, mk_dir='processed')
    obj_save(osp.join(save_dir, 'train_data.list'), valid_train_data)
    obj_save(osp.join(save_dir, 'train_label.list'), valid_train_label)
    obj_save(osp.join(save_dir, 'test_data.list'), valid_test_data)
    obj_save(osp.join(save_dir, 'test_label.list'), valid_test_label)
    print(f'save data in [{save_dir}].')


def newsgroups20_dataset(data_path):

    """
    The 20 Newsgroups data set is a collection of approximately 20,000 newsgroup documents, 
    partitioned (nearly) evenly across 20 different newsgroups.

    """
   
    train_set = fetch_20newsgroups(
                                data_home=data_path, 
                                subset='train', 
                                remove=('headers', 'footers', 'quotes'),
                                download_if_missing=False
                                 )  
    test_set = fetch_20newsgroups(
                                data_home=data_path, 
                                subset='test', 
                                remove=('headers', 'footers', 'quotes'),
                                download_if_missing=False
                                 )
    # print(f'Train set ================')
    # print(f'Size: {len(train_data.data)}')
    # print(f'A text: {train_data.data[0]}')
    # print(f'Target: {train_data.target[0]}')
    # print(f'Target name: {train_data.target_names}')
    # print(f'Test set ================')
    # print(f'Size: {len(test_data.data)}')
    # print(f'A text: {test_data.data[0]}')
    # print(f'Target: {test_data.target[0]}')
    # print(f'Target name: {test_data.target_names}')

    # print(f'data info =================')
    print(f'train set: {len(train_set.data)}')
    print(f'test set: {len(test_set.data)}')
    # print(f'============================')
    # # clean data
    # if clean:    
    #     train_data = []
    #     test_data = []

    #     for text in train_set.data:
    #         train_data.append(clean_text(text))
        
    #     for text in test_set.data:
    #         test_data.append(clean_text(text))
    # else:
    #     train_data = train_set.data
    #     test_data = test_set.data
    train_data = train_set.data
    test_data = test_set.data
    
    print(f'target names: {train_set.target_names}')
    print(f'target names: {test_set.target_names}')
    
    return train_data, train_set.target, test_data, test_set.target


def reuters_21578(data_path):


    '''
      The documents in the Reuters-21578 collection appeared on the
        Reuters newswire in 1987.
    '''

    # 59 classes
    # normal class: earn, acq, outliers: others
    # classes = {'earn': 0, 'acq': 1, 'others': 2}
    nltk.download('reuters', download_dir=data_path)
    if data_path not in nltk.data.path:
        nltk.data.path.append(data_path)

    doc_ids = reuters.fileids()

    splits = ['train', 'test']
    train_data = []
    train_label = []
    test_data = []
    test_label = []

    for split_set in splits:
        split_set_doc_ids = list(filter(lambda doc: doc.startswith(split_set), doc_ids))
        data = []
        label = []
        for id in split_set_doc_ids:
            target = reuters.categories(id)

            if len(target) == 1:
                data.append(reuters.raw(id))
                if target[0] == 'earn':
                    label.append(0)
                elif target[0] == 'acq':
                    label.append(1)
                else:
                    label.append(2)

        print(f'[{split_set}] ========================')
        print(f'data: [{len(data)}]')
        print(f'topics: {len(set(label)), set(label)}')

        if split_set == 'train':
            train_data = copy.deepcopy(data)
            train_label = copy.deepcopy(label)
        else:
            test_data = copy.deepcopy(data)
            test_label = copy.deepcopy(label)

    return train_data, train_label, test_data, test_label
        

def imdb(data_path):

    '''
    This is a dataset for binary sentiment classification 
    containing substantially more data than previous benchmark datasets.
    '''

    train_set, test_set = imdb_dataset(data_path, train=True, test=True)

    # normal class: pos, abnormal class: neg
    classes = {'pos': 0, 'neg': 1}

    train_data = []
    train_label = []
    test_data = []
    test_label = []
    normal_train = 0
    normal_test = 0

    for e in train_set:
        train_data.append(e['text'])
        train_label.append(classes[e['sentiment']])
        if e['sentiment'] == 'pos':
            normal_train += 1
    
    for e in test_set:
        test_data.append(e['text'])
        test_label.append(classes[e['sentiment']])
        if e['sentiment'] == 'pos':
            normal_test += 1

    print(f'train set ========================')
    print(f'Size: {len(train_data)}')
    print(f'normal data: {normal_train}')
    # print(f'{train_set[0]}')

    print(f'test set ========================')
    print(f'Size: {len(test_data)}')
    print(f'normal data: {normal_test}')
    # print(f'{test_set[0]}')
    return train_data, train_label, test_data, test_label


def sst2(data_path):

    '''
    Binary classification experiments on full sentences 
    (negative or somewhat negative vs somewhat positive or 
    positive with neutral sentences discarded) refer to the dataset as SST-2 or SST binary.
    '''
    # 0: negative
    # 1: postive
    ds = load_dataset('stanfordnlp/sst2', cache_dir=data_path)
    print(type(ds), ds.shape)
    train_set = ds['train']
    test_set = ds['test']
    val_set = ds['validation']

    train_data = []
    train_label = []
    test_data = []
    test_label = []
    normal_train = 0
    normal_test = 0

    for e in train_set:

        train_data.append(e['sentence'])
        train_label.append(e['label'])
        if e['label'] == 1:
            normal_train += 1

    for e in test_set:

        test_data.append(e['sentence'])
        test_label.append(e['label'])
        if e['label'] == 1:
            normal_test += 1

    for e in val_set:

        test_data.append(e['sentence'])
        test_label.append(e['label'])
        if e['label'] == 1:
            normal_test += 1
    
    print(f'train set =======================')
    print(f'Size: {len(train_data)}')
    print(f'Positive: {normal_train}')
    print(f'Negative: {len(train_data) - normal_train}')

        
    print(f'test set =======================')
    print(f'Size: {len(test_data)}')
    print(f'Positive: {normal_test}')
    print(f'Negative: {len(test_data) - normal_test}')

    return train_data, train_label, test_data, test_label


def dbpedia14(data_path):

    '''
    The DBpedia ontology classification dataset is constructed by picking 14 non-overlapping classes from DBpedia 2014. 
    They are listed in classes.txt. From each of thse 14 ontology classes, we randomly choose 40,000 training samples and 5,000 testing samples.
    Therefore, the total size of the training dataset is 560,000 and testing dataset 70,000. 
    There are 3 columns in the dataset (same for train and test splits), corresponding to class index (1 to 14), 
    title and content. The title and content are escaped using double quotes ("), and any internal double quote is escaped by 2 double quotes (""). 
    There are no new lines in title or content.
    '''
    # 14 classes
    ds = load_dataset("fancyzhx/dbpedia_14", cache_dir=data_path)
    print(type(ds), ds.shape)
    train_set = ds['train']
    test_set = ds['test']

    train_data = []
    train_label = []
    test_data = []
    test_label = []

    for e in train_set:

        train_data.append(e['content'])
        train_label.append(e['label'])

    for e in test_set:

        test_data.append(e['content'])
        test_label.append(e['label'])

    
    print(f'train set =======================')
    print(f'Size: {len(train_data)}')
    print(f'Classes: {set(train_label)}')

        
    print(f'test set =======================')
    print(f'Size: {len(test_data)}')
    print(f'Classes: {set(test_label)}')

    return train_data, train_label, test_data, test_label


def sms_spam(data_path):

    '''
    The SMS Spam Collection v.1 is a public set of SMS labeled messages 
    that have been collected for mobile phone spam research. 
    It has one collection composed by 5,574 English, real and non-enconded messages, 
    tagged according being legitimate (ham) or spam.
    '''
    # binary class: spam or ham
    # ham: 0, spam: 1
    ds = load_dataset("ucirvine/sms_spam", cache_dir=data_path)
    print(type(ds), ds.shape, ds.column_names)
    train_set = ds['train']
    ham_data = []
    spam_data = []
    train_data = []
    train_label = []
    test_data = []
    test_label = []

    for e in train_set:
        if e['label'] == 0:
            ham_data.append(e['sms'])
        else:
            spam_data.append(e['sms'])
    
    print(f'Spam data ====================')
    print(f'Size: {len(spam_data)}')
    print(f'Ham data ====================')
    print(f'Size: {len(ham_data)}')

    test_data = spam_data + ham_data[0:len(spam_data)]
    test_label = np.concatenate((np.ones(len(spam_data)), np.zeros(len(spam_data)))).tolist()
    train_data = ham_data[len(spam_data):]
    train_label = np.zeros(len(ham_data) - len(spam_data)).tolist()

    print(f'train data ===================')
    print(f'Size: {len(train_data)}')
    print(f'label: {len(train_label)}')

    print(f'test data ===================')
    print(f'Size: {len(test_data)}')
    print(f'label: {len(test_label)}')

    return train_data, train_label, test_data, test_label
    

def enron_email(data_path):

    '''
    It contains data from about 150 users, mostly senior management of Enron, 
    organized into folders. The corpus contains a total of about 0.5M messages. 
    '''
    # normal class: kay.mann@enron.com : 0, vince.kaminski@enron.com: 1 (sample size > 10000)
    # abnormal class: others with only one email: 2 (sample size == 1)
    ds = load_dataset("Hellisotherpeople/enron_emails_parsed", cache_dir=data_path)
    print(type(ds), ds.shape, ds.column_names)

    data_set = ds['train']
    dict_user = {}
    label = set()

    normal_data = []
    normal_label = []

    for e in data_set:
        
        email_list = e['from'].split(' ')
        for part in email_list:
            if part.find('.com') > 0 and part.find('mailto') == -1:
                part = part.replace('@ENRON', '')
                part = part.replace('<', '')
                part = part.replace('>', '')
                part = part.replace('[', '')
                part = part.replace(']', '')
                if part == 'kay.mann@enron.com':
                    normal_data.append(e['body'])
                    normal_label.append(0)
                elif part == 'vince.kaminski@enron.com':
                    normal_data.append(e['body'])
                    normal_label.append(1)
                else:
                    if part not in label:
                        dict_user.setdefault(part, e['body'])
                    else:
                        try:
                            del dict_user[part]
                        except KeyError:
                            pass
                    label.add(part)
                break

    abnormal_data = list(dict_user.values())

    print(f'data info ===============')
    print(f'normal data: {len(normal_data)}')
    print(f'normal label: {set(normal_label), len(normal_label)}')
    print(f'abnormal data: {len(abnormal_data)}')

    train_data = normal_data
    train_label = normal_label
    test_data = abnormal_data
    test_label = (2 * np.ones(len(test_data))).tolist()

    return train_data, train_label, test_data, test_label


def wos(data_path):

    '''
    Web of Science Dataset WOS-46985
        -This dataset contains 46,985 documents with 134 categories which include 7 parents categories.
            - Computer Science (CS): 0
            - Electrical Engineering (ECE): 1
            - Psychology: 2
            - Mechanical Engineering (MAE): 3
            - Civil Engineering (Civil): 4
            - Medical Science (Medical): 5
            - biochemistry: 6
    '''

    from datasets import load_dataset

    ds = load_dataset("river-martin/web-of-science-with-label-texts", cache_dir=data_path)
    print(type(ds), ds.shape, ds.column_names)

    train_set = ds['train']
    test_set = ds['test']
    val_set = ds['validate']

    train_data = []
    train_label = []
    test_data = []
    test_label = []



    for e in train_set:

        train_data.append(e['abstract'])
        if e['domain'] == 'CS':
            train_label.append(0)
        elif e['domain'] == 'Psychology':
            train_label.append(1)
        elif e['domain'] == 'biochemistry':
            train_label.append(2)
        elif e['domain'] == 'Medical':
            train_label.append(3)
        elif e['domain'] == 'ECE':
            train_label.append(4)
        elif e['domain'] == 'MAE':
            train_label.append(5)
        elif e['domain'] == 'Civil':
            train_label.append(6)

    for e in test_set:

        test_data.append(e['abstract'])
        if e['domain'] == 'CS':
            test_label.append(0)
        elif e['domain'] == 'Psychology':
            test_label.append(1)
        elif e['domain'] == 'biochemistry':
            test_label.append(2)
        elif e['domain'] == 'Medical':
            test_label.append(3)
        elif e['domain'] == 'ECE':
            test_label.append(4)
        elif e['domain'] == 'MAE':
            test_label.append(5)
        elif e['domain'] == 'Civil':
            test_label.append(6)

    for e in val_set:

        test_data.append(e['abstract'])
        if e['domain'] == 'CS':
            test_label.append(0)
        elif e['domain'] == 'Psychology':
            test_label.append(1)
        elif e['domain'] == 'biochemistry':
            test_label.append(2)
        elif e['domain'] == 'Medical':
            test_label.append(3)
        elif e['domain'] == 'ECE':
            test_label.append(4)
        elif e['domain'] == 'MAE':
            test_label.append(5)
        elif e['domain'] == 'Civil':
            test_label.append(6)


    print(f'train data =====================')
    print(f'Size: {len(train_data)}')
    print(f'label: {len(train_label), set(train_label)}')

    print(f'test data =====================')
    print(f'Size: {len(test_data)}')
    print(f'label: {len(test_label), set(test_label)}')

    return train_data, train_label, test_data, test_label


def check_token_number(data, target, max_length=8191):

    valid_data = []
    valid_target = []
    invalid_text_num = 0
    for i, (e, t) in enumerate(zip(data, target)):
        if 0 < len(e) and 0 < len(e.split(' ')) < max_length:
            valid_data.append(e)
            valid_target.append(t)
        else:
            invalid_text_num += 1
            # print(f"[{i}]: {len(e.split(' '))}")
    # print(f'Invalid text: [{invalid_text_num}]')
    return valid_data, valid_target


def compute_tfidf_weights(train_set, test_set, vocab_size):
    """ Compute the Tf-idf weights (based on idf vector computed from train_set)."""

    transformer = TfidfTransformer()

    # fit idf vector on train set
    counts = np.zeros((len(train_set), vocab_size), dtype=np.int64)
    for i, row in enumerate(train_set):
        counts_sample = torch.bincount(row['text'])
        counts[i, :len(counts_sample)] = counts_sample.cpu().data.numpy()
    tfidf = transformer.fit_transform(counts)

    for i, row in enumerate(train_set):
        row['weight'] = torch.tensor(tfidf[i, row['text']].toarray().astype(np.float32).flatten())

    # compute tf-idf weights for test set (using idf vector from train set)
    counts = np.zeros((len(test_set), vocab_size), dtype=np.int64)
    for i, row in enumerate(test_set):
        counts_sample = torch.bincount(row['text'])
        counts[i, :len(counts_sample)] = counts_sample.cpu().data.numpy()
    tfidf = transformer.transform(counts)

    for i, row in enumerate(test_set):
        row['weight'] = torch.tensor(tfidf[i, row['text']].toarray().astype(np.float32).flatten())

    
def clean_text(text: str, rm_numbers=False, rm_punct=True, rm_stop_words=True, rm_short_words=True):

    """ Function to perform common NLP pre-processing tasks. """

    # make lowercase
    text = text.lower()

    # remove punctuation
    if rm_punct:
        text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))

    # remove numbers
    if rm_numbers:
        text = re.sub(r'\d+', '', text)

    # remove whitespaces
    text = text.strip()

    # remove stopwords
    if rm_stop_words:
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text)
        text_list = [w for w in word_tokens if not w in stop_words]
        text = ' '.join(text_list)

    # remove short words
    if rm_short_words:
        text_list = [w for w in text.split() if len(w) >= 3]
        text = ' '.join(text_list)
    return text


def get_data(dataset):

    print(f'[{dataset}]==============================')
    data_dir = osp.join(DATA_DIR, f'{dataset}/processed')
    train_data = obj_load(osp.join(data_dir, 'train_data.list'))
    train_label = obj_load(osp.join(data_dir, 'train_label.list'))
    test_data = obj_load(osp.join(data_dir, 'test_data.list'))
    test_label = obj_load(osp.join(data_dir, 'test_label.list'))

    print(f'data info =================')
    print(f'train set: {len(train_data)}')
    print(f'test set: {len(test_data)}')
    print(f'Sample size: {len(train_data) + len(test_data)}')
    print(f'train classes: {set(train_label)}')
    print(f'test classes: {set(test_label)}')
    print(f'============================')
    pass


if __name__ == '__main__':

    main()