__author__ = 'XF'
__date__ = '2024/09/20'

'''

'''
import numpy as np
from os import path as osp
from configs import EMBEDDING_DIR, DATA_DIR, CLASS_20newsgroups, CLASS_enron, CLASS_imdb, CLASS_reuters21578, CLASS_sst2, CLASS_wos
from configs import DBPEDIA14_CHUNK, MAX_SIZE_SINGLE_TRAIN_CLASS_DBPEDIA14, MAX_SIZE_SINGLE_TEST_CLASS_DBPEDIA14, DBPEDIA14_CLASS
from tools import obj_load, obj_save, new_dir
CHUNK_EMBEDDING = 50000

model_dirs = [
    'bert',
    'Llama2-7b',
    'Llama3-8b',
    'Mistral-7b',
    'openai'
]
llm_embeddings = [
    'mntp_eos_token_embeddings',
    'mntp_mean_embeddings',
    'mntp_weighted_mean_embeddings',

    'mntp-unsup-simcse_eos_token_embeddings',
    'mntp-unsup-simcse_mean_embeddings',
    'mntp-unsup-simcse_weighted_mean_embeddings',

    'mntp-supervised_eos_token_embeddings',
    'mntp-supervised_mean_embeddings',
    'mntp-supervised_weighted_mean_embeddings',    

]
bert_embeddings = [
    'cls_token_embeddings',
    'mean_embeddings'
]
openai_embeddings = [
    'openai_text-embedding-3-large_embedding',
    'openai_text-embedding-3-small_embedding',
    'openai_text-embedding-ada-002_embedding'
]





def get_data(dataset, base_model, ft_model, pooling, normal_class=None):
    
    spliter = None
    if dataset == '20newsgroups':
        spliter = split_20newsgroups
    elif dataset == 'dbpedia14':
        spliter = split_dbpedia14
    elif dataset == 'enron':
        spliter = split_enron
    elif dataset == 'imdb':
        spliter = split_imdb
    elif dataset == 'reuters21578':
        spliter = split_reuters21578
    elif dataset == 'sms_spam':
        spliter = split_sms_spam
    elif dataset == 'sst2':
        spliter = split_sst2
    elif dataset == 'wos':
        spliter = split_wos
    
    data_path, label_path = get_embedding_path(dataset, base_model, ft_model, pooling)
    # data_path, label_path = get_data_path(dataset)
    
    if base_model == 'glove_6b':
        if dataset == 'dbpedia14':
            for i, (d_path, l_path) in enumerate(zip(data_path, label_path)):
                data_path[i] = d_path.replace('mean_embeddings', f'{str(normal_class)}/mean_embeddings.npy')
                label_path[i] = l_path.replace('embedding_labels', f'{str(normal_class)}/embedding_labels.npy')
        train_data, train_label, test_data, test_label = load_data_glove(data_path, label_path)
    else:
        if dataset == 'dbpedia14':
            train_data, train_label, test_data, test_label = spliter(data_path, label_path, normal_class)
        else:
            train_data, train_label, test_data, test_label = spliter(data_path, label_path)
    
    # save_valid_data(dataset, normal_class=normal_class, train_data=train_data, test_data=test_data, train_label=train_label, test_label=test_label)

    print(f'data info ========================')
    print(f'Training set:{train_data.shape} ')
    print(f'Test set: {test_data.shape}')
    print(f'==================================')
    # exit(0)
    

    return train_data, train_label, test_data, test_label, train_data.shape[1]

def load_data_glove(data_path, label_path):

    train_set = obj_load(data_path[0])
    train_label = obj_load(label_path[0])

    test_set = obj_load(data_path[1])
    test_label = obj_load(label_path[1])

    return train_set, train_label, test_set, test_label



def split_20newsgroups(data_path, label_path):

    train_set = obj_load(data_path[0])
    train_label = obj_load(label_path[0])

    test_set = obj_load(data_path[1])
    test_label = obj_load(label_path[1])


    train_normal_data = []
    train_abnormal_data = []
    for sample, label in zip(train_set, train_label):
        if label in CLASS_20newsgroups['normal']:
            train_normal_data.append(sample)
        elif label in CLASS_20newsgroups['abnormal']:
            train_abnormal_data.append(sample)
    
    test_normal_data = []
    test_abnormal_data = []
    for sample, label in zip(test_set, test_label):
        if label in CLASS_20newsgroups['normal']:
            test_normal_data.append(sample)
        elif label in CLASS_20newsgroups['abnormal']:
            test_abnormal_data.append(sample)
    

    train_data = np.array(train_normal_data, dtype=np.float32)
    train_label = np.zeros(len(train_data), dtype=np.int8)

    test_abnormal_data = np.concatenate((train_abnormal_data, test_abnormal_data))
    test_data = np.concatenate((test_normal_data, test_abnormal_data), dtype=np.float32)
    test_label = np.concatenate((np.zeros(len(test_normal_data), dtype=np.int8), np.ones(len(test_abnormal_data), dtype=np.int8)))

    # =======================
    # temporary chaning 
    # train_data = train_normal_data
    # train_label = np.zeros(len(train_data), dtype=np.int8).tolist()
    
    # test_abnormal_data.extend(train_abnormal_data)
    # test_label = np.concatenate((np.zeros(len(test_normal_data), dtype=np.int8), np.ones(len(test_abnormal_data), dtype=np.int8))).tolist()
    # test_normal_data.extend(test_abnormal_data)
    # test_data = test_normal_data
    

    # return train_data, train_label, test_data, test_label
    # =======================

    print('data splitting ========================================')
    for dp, lp in zip(data_path, label_path):
        print(f'data path: [{dp}]')
        print(f'label path: [{lp}]')
    print(f'train_data: {type(train_data), train_data.shape}')
    print(f'train_label: {type(train_label), train_label.shape}')
    print(f'set train label: {set(train_label)}')
    print(f'test_data: {type(test_data), test_data.shape}')
    print(f'test_label: {type(test_label), test_label.shape}')
    print(f'set test label: {set(test_label)}')
    print(f'anomaly ratio: {len(test_abnormal_data) / len(test_data)}')
    print(f'All size: {len(train_data) + len(test_data)}')
    print('data splitting ========================================')

    return train_data, train_label, test_data, test_label


def split_dbpedia14(data_path, label_path, normal_class):


    # =========================
    # temporary changing
    # train_set = obj_load(data_path[0])
    # train_label = obj_load(label_path[0])

    # test_set = obj_load(data_path[1])
    # test_label = obj_load(label_path[1])

    # train_idx = np.where(np.array(train_label) == normal_class)[0]
    
    # num_train_data = int(MAX_SIZE_SINGLE_TRAIN_CLASS_DBPEDIA14 / 2)
    # normal_data = [train_set[i] for i in train_idx[:MAX_SIZE_SINGLE_TRAIN_CLASS_DBPEDIA14]]
    # train_data = normal_data[:num_train_data]
    # train_label = np.zeros(len(train_data), dtype=np.int8).tolist()
    # test_normal_data = normal_data[num_train_data:]

    # test_data = []
    # test_data_label = []
    # for c in DBPEDIA14_CLASS:
    #     test_idx = np.where(np.array(test_label) == c)[0][:MAX_SIZE_SINGLE_TEST_CLASS_DBPEDIA14]
    #     test_data.extend([test_set[i] for i in test_idx])
    #     test_data_label.extend([test_label[i] for i in test_idx])
        
    
    # test_data_label = np.where(np.array(test_data_label) == normal_class, 0, 1)
    # test_data = np.concatenate((test_normal_data, test_data)).tolist()
    # test_label = np.concatenate((np.zeros(len(test_normal_data)), test_data_label)).tolist()
    

    # return train_data, train_label, test_data, test_label
    # =========================

    train_valid_chunks, test_valid_chunks = find_valid_chunks(label_path, normal_class)

    train_data_path = data_path[0]
    test_data_path = data_path[1]

    # train data
    normal_data = None
    for chunk, idxs in train_valid_chunks.items():
        data_chunk_path =  train_data_path + '_' + chunk + '.npy'
        temp_data = obj_load(data_chunk_path)
        if normal_data is None:
            normal_data = temp_data[idxs]
        else:
            normal_data = np.concatenate((normal_data, temp_data[idxs]))
    num_train_data = int(MAX_SIZE_SINGLE_TRAIN_CLASS_DBPEDIA14 / 2)
    train_data = normal_data[:num_train_data]
    train_label = np.zeros(len(train_data), dtype=np.int8)


    # test data
    test_data = None
    test_label = None
    for chunk, idxs in test_valid_chunks.items():
        data_chunk_path =  test_data_path + '_' + chunk + '.npy'
        label_chunk_path = label_path[1] + '_' + chunk + '.npy'
        temp_data = obj_load(data_chunk_path)
        temp_label = np.array(obj_load(label_chunk_path), dtype=np.int8)
        if test_data is None:
            test_data = temp_data[idxs]
        else:
            test_data = np.concatenate((test_data, temp_data[idxs]))
        if test_label is None:
            test_label = temp_label[idxs]
        else:
            test_label = np.concatenate((test_label, temp_label[idxs]))
    test_label = np.where(test_label == normal_class, 0, 1)
    test_data = np.concatenate((normal_data[num_train_data:], test_data))
    test_label = np.concatenate((np.zeros(len(normal_data[num_train_data:])), test_label))
    num_test_abnormal_data = np.count_nonzero(test_label)


    print('data splitting ========================================')
    print(f'normal class: [{normal_class}]')
    print(f'train_data: {type(train_data), train_data.shape}')
    print(f'train_label: {type(train_label), train_label.shape}')
    print(f'set train label: {set(train_label)}')
    print(f'test_data: {type(test_data), test_data.shape}')
    print(f'test_label: {type(test_label), test_label.shape}')
    print(f'set test label: {set(test_label)}')
    print(f'anomaly ratio: {num_test_abnormal_data / len(test_data)}')
    print(f'All size: {len(train_data) + len(test_data)}')
    print('data splitting ========================================')

    return train_data, train_label, test_data, test_label


def split_enron(data_path, label_path):

    train_set = obj_load(data_path[0])
    train_label = obj_load(label_path[0])

    test_set = obj_load(data_path[1])
    test_label = obj_load(label_path[1])

    test_abnormal_data = test_set
    num_normal_data_per_class = 5000
    class_0_num = num_normal_data_per_class
    class_1_num = num_normal_data_per_class
    test_normal_data = []
    train_normal_data = []

    for sample, label in zip(train_set, train_label):
        
        if label == CLASS_enron['normal'][0] and class_0_num > 0:
            train_normal_data.append(sample)
            class_0_num -= 1
        elif label == CLASS_enron['normal'][1] and class_1_num > 0:
            train_normal_data.append(sample)
            class_1_num -= 1
        else:
            test_normal_data.append(sample)
    

    train_data = np.array(train_normal_data, dtype=np.float32)
    train_label = np.zeros(len(train_data), dtype=np.int8)
    test_data = np.concatenate((test_normal_data, test_abnormal_data), dtype=np.float32)
    test_label = np.concatenate((np.zeros(len(test_normal_data), dtype=np.int8), np.ones(len(test_abnormal_data), dtype=np.int8)))

    # ==============================
    # temporary chaning
    # train_data = train_normal_data
    # train_label = np.zeros(len(train_data), dtype=np.int8).tolist()
    # test_label = np.concatenate((np.zeros(len(test_normal_data), dtype=np.int8), np.ones(len(test_abnormal_data), dtype=np.int8))).tolist()
    # test_normal_data.extend(test_abnormal_data)
    # test_data = test_normal_data
    # return train_data, train_label, test_data, test_label


    # ==============================

    print('data splitting ========================================')
    for dp, lp in zip(data_path, label_path):
        print(f'data path: [{dp}]')
        print(f'label path: [{lp}]')
    print(f'train_data: {type(train_data), train_data.shape}')
    print(f'train_label: {type(train_label), train_label.shape}')
    print(f'set train label: {set(train_label)}')
    print(f'test_data: {type(test_data), test_data.shape}')
    print(f'test_label: {type(test_label), test_label.shape}')
    print(f'set test label: {set(test_label)}')
    print(f'anomaly ratio: {len(test_abnormal_data) / len(test_data)}')
    print(f'All size: {len(train_data) + len(test_data)}')
    print('data splitting ========================================')

    return train_data, train_label, test_data, test_label


def split_imdb(data_path, label_path):

    train_set = obj_load(data_path[0])
    train_label = obj_load(label_path[0])

    test_set = obj_load(data_path[1])
    test_label = obj_load(label_path[1])

    normal_data = []
    abnormal_data = []

    for sample_1, label_1, sample_2, label_2 in zip(train_set, train_label, test_set, test_label):
        if label_1 in CLASS_imdb['normal']:
            normal_data.append(sample_1)
        elif label_1 in CLASS_imdb['abnormal']:
            abnormal_data.append(sample_1)
        if label_2 in CLASS_imdb['normal']:
            normal_data.append(sample_2)
        elif label_1 in CLASS_imdb['abnormal']:
            abnormal_data.append(sample_2)

    num_train_normal_data = 10000
    train_normal_data = normal_data[:num_train_normal_data]
    test_normal_data = normal_data[num_train_normal_data:]
    test_abnormal_data = abnormal_data

    train_data = np.array(train_normal_data, dtype=np.float32)
    train_label = np.zeros(len(train_data), dtype=np.int8)
    test_data = np.concatenate((test_normal_data, test_abnormal_data), dtype=np.float32)
    test_label = np.concatenate((np.zeros(len(test_normal_data), dtype=np.int8), np.ones(len(test_abnormal_data), dtype=np.int8)))

    
    # ==============================
    # temporary chaning
    # train_data = train_normal_data
    # train_label = np.zeros(len(train_data), dtype=np.int8).tolist()
    # test_label = np.concatenate((np.zeros(len(test_normal_data), dtype=np.int8), np.ones(len(test_abnormal_data), dtype=np.int8))).tolist()
    # test_normal_data.extend(test_abnormal_data)
    # test_data = test_normal_data
    # return train_data, train_label, test_data, test_label


    # ==============================
    
    print('data splitting ========================================')
    for dp, lp in zip(data_path, label_path):
        print(f'data path: [{dp}]')
        print(f'label path: [{lp}]')
    print(f'train_data: {type(train_data), train_data.shape}')
    print(f'train_label: {type(train_label), train_label.shape}')
    print(f'set train label: {set(train_label)}')
    print(f'test_data: {type(test_data), test_data.shape}')
    print(f'test_label: {type(test_label), test_label.shape}')
    print(f'set test label: {set(test_label)}')
    print(f'anomaly ratio: {len(test_abnormal_data) / len(test_data)}')
    print('data splitting ========================================')
    print(f'All size: {len(train_data) + len(test_data)}')

    return train_data, train_label, test_data, test_label


def split_reuters21578(data_path, label_path):

    train_set = obj_load(data_path[0])
    train_label = obj_load(label_path[0])

    test_set = obj_load(data_path[1])
    test_label = obj_load(label_path[1])

    train_normal_data = []
    test_abnormal_data = []
    print(set(train_label))
    
    for sample, label in zip(train_set, train_label):
        if label in CLASS_reuters21578['normal']:
            train_normal_data.append(sample)
        elif label in CLASS_reuters21578['abnormal']:
            test_abnormal_data.append(sample)

    train_data = np.array(train_normal_data, dtype=np.float32)
    train_label = np.zeros(len(train_data), dtype=np.int8)
    
    test_data = np.concatenate((test_set, test_abnormal_data), dtype=np.float32)

    test_label = [0 if e != 2 else 1 for e in test_label]
    test_label = np.concatenate((test_label, np.ones(len(test_abnormal_data), dtype=np.int8)), dtype=np.int8)


    
    # ==============================
    # temporary chaning
    # train_data = train_normal_data
    # train_label = np.zeros(len(train_data), dtype=np.int8).tolist()
    # test_label = [0 if e != 2 else 1 for e in test_label]
    # test_label = np.concatenate((test_label, np.ones(len(test_abnormal_data), dtype=np.int8)), dtype=np.int8).tolist()
    # test_set.extend(test_abnormal_data)
    # test_data = test_set
    # return train_data, train_label, test_data, test_label


    # ==============================
    print('data splitting ========================================')
    for dp, lp in zip(data_path, label_path):
        print(f'data path: [{dp}]')
        print(f'label path: [{lp}]')
    print(f'train_data: {type(train_data), train_data.shape}')
    print(f'train_label: {type(train_label), train_label.shape}')
    print(f'set train label: {set(train_label)}')
    print(f'test_data: {type(test_data), test_data.shape}')
    print(f'test_label: {type(test_label), test_label.shape}')
    print(f'set test label: {set(test_label)}')
    print(f'anomaly ratio: {len(test_abnormal_data) / len(test_data)}')
    print('data splitting ========================================')
    print(f'All size: {len(train_data) + len(test_data)}')

    return train_data, train_label, test_data, test_label


def split_sms_spam(data_path, label_path):


    train_set = obj_load(data_path[0])
    train_label = obj_load(label_path[0])

    test_set = obj_load(data_path[1])
    test_label = obj_load(label_path[1])

    num_train_normal_data = 3000

    train_data = np.array(train_set[:num_train_normal_data], dtype=np.float32)
    train_label = np.zeros(len(train_data), dtype=np.int8)
    test_data = np.concatenate((train_set[num_train_normal_data:], test_set), dtype=np.float32)

    test_label = np.concatenate((np.zeros(len(train_set[num_train_normal_data:]), dtype=np.int8), np.array(test_label, dtype=np.int8)))

    # ===========================
    # temporary chaning
    # train_data = train_set[:num_train_normal_data]
    # train_label = np.zeros(len(train_data), dtype=np.int8).tolist()
    # test_label = np.concatenate((np.zeros(len(train_set[num_train_normal_data:]), dtype=np.int8), np.array(test_label, dtype=np.int8))).tolist()
    # test_normal_data = train_set[num_train_normal_data:]
    # test_normal_data.extend(test_set)
    # test_data = test_normal_data
    # return train_data, train_label, test_data, test_label
    # ===========================

    num_abnormal_data = 0
    for label in test_label:
        if label == 1:
            num_abnormal_data += 1
    # print(f'test abnormal data:{num_abnormal_data}')
    print('data splitting ========================================')
    for dp, lp in zip(data_path, label_path):
        print(f'data path: [{dp}]')
        print(f'label path: [{lp}]')
    print(f'train_data: {type(train_data), train_data.shape}')
    print(f'train_label: {type(train_label), train_label.shape}')
    print(f'set train label: {set(train_label)}')
    print(f'test_data: {type(test_data), test_data.shape}')
    print(f'test_label: {type(test_label), test_label.shape}')
    print(f'set test label: {set(test_label)}')
    print(f'anomaly ratio: {num_abnormal_data / len(test_data)}')
    print(f'All size: {len(train_data) + len(test_data)}')
    print('data splitting ========================================')

    return train_data, train_label, test_data, test_label


def split_sst2(data_path, label_path):

    
    train_set = obj_load(data_path[0])
    train_label = obj_load(label_path[0])

    test_set = obj_load(data_path[1])
    test_label = obj_load(label_path[1])


    train_normal_data = []
    train_abnormal_data = []
    for sample, label in zip(train_set, train_label):

        if label in CLASS_sst2['normal']:
            train_normal_data.append(sample)
        elif label in CLASS_sst2['abnormal']:
            train_abnormal_data.append(sample)
    
    test_normal_data = []
    test_abnormal_data = []
    for sample, label in zip(test_set, test_label):

        if label in CLASS_sst2['normal']:
            test_normal_data.append(sample)
        elif label in CLASS_sst2['abnormal']:
            test_abnormal_data.append(sample)

    num_train_normal_data = 10000    
    train_data = np.array(train_normal_data[:num_train_normal_data], dtype=np.float32)
    train_label = np.zeros(len(train_data), dtype=np.int8)
    test_normal_data = np.concatenate((test_normal_data, train_normal_data[num_train_normal_data:]))
    test_abnormal_data = np.concatenate((train_abnormal_data, test_abnormal_data))
    test_data = np.concatenate((test_normal_data, test_abnormal_data), dtype=np.float32)
    test_label = np.concatenate((np.zeros(len(test_normal_data), dtype=np.int8), np.ones(len(test_abnormal_data), dtype=np.int8)))


    # =================================

    # train_data = train_normal_data[:num_train_normal_data]
    # train_label = np.zeros(len(train_data), dtype=np.int8).tolist()
    # test_normal_data.extend(train_normal_data[num_train_normal_data:])
    # test_abnormal_data.extend(train_abnormal_data)
    # test_label = np.concatenate((np.zeros(len(test_normal_data), dtype=np.int8), np.ones(len(test_abnormal_data), dtype=np.int8))).tolist()
    # test_normal_data.extend(test_abnormal_data)
    # test_data = test_normal_data
    # return train_data, train_label, test_data, test_label

    # =================================

    print('data splitting ========================================')
    for dp, lp in zip(data_path, label_path):
        print(f'data path: [{dp}]')
        print(f'label path: [{lp}]')
    print(f'train_data: {type(train_data), train_data.shape}')
    print(f'train_label: {type(train_label), train_label.shape}')
    print(f'set train label: {set(train_label)}')
    print(f'test_data: {type(test_data), test_data.shape}')
    print(f'test_label: {type(test_label), test_label.shape}')
    print(f'set test label: {set(test_label)}')
    print(f'anomaly ratio: {len(test_abnormal_data) / len(test_data)}')
    print(f'All size: {len(train_data) + len(test_data)}')
    print('data splitting ========================================')

    return train_data, train_label, test_data, test_label


def split_wos(data_path, label_path):

    train_set = obj_load(data_path[0])
    train_label = obj_load(label_path[0])

    test_set = obj_load(data_path[1])
    test_label = obj_load(label_path[1])

    train_normal_data = []
    train_abnormal_data = []
    for sample, label in zip(train_set, train_label):
        if label in CLASS_wos['normal']:
            train_normal_data.append(sample)
        elif label in CLASS_wos['abnormal']:
            train_abnormal_data.append(sample)
    
    test_normal_data = []
    test_abnormal_data = []
    for sample, label in zip(test_set, test_label):
        if label in CLASS_wos['normal']:
            test_normal_data.append(sample)
        elif label in CLASS_wos['abnormal']:
            test_abnormal_data.append(sample)

    train_data = np.array(train_normal_data, dtype=np.float32)
    train_label = np.zeros(len(train_data), dtype=np.int8)
    test_abnormal_data = np.concatenate((train_abnormal_data, test_abnormal_data))
    test_data = np.concatenate((test_normal_data, test_abnormal_data), dtype=np.float32)
    test_label = np.concatenate((np.zeros(len(test_normal_data), dtype=np.int8), np.ones(len(test_abnormal_data), dtype=np.int8)))

    # ==================================
    # temporary changing
    # train_data = train_normal_data
    # train_label = np.zeros(len(train_data), dtype=np.int8).tolist()
    # test_abnormal_data.extend(train_abnormal_data)
    # test_label = np.concatenate((np.zeros(len(test_normal_data), dtype=np.int8), np.ones(len(test_abnormal_data), dtype=np.int8))).tolist()
    # test_normal_data.extend(test_abnormal_data)
    # test_data = test_normal_data

    # return train_data, train_label, test_data, test_label

    # ==================================

    print('data splitting ========================================')
    for dp, lp in zip(data_path, label_path):
        print(f'data path: [{dp}]')
        print(f'label path: [{lp}]')
    print(f'train_data: {type(train_data), train_data.shape}')
    print(f'train_label: {type(train_label), train_label.shape}')
    print(f'set train label: {set(train_label)}')
    print(f'test_data: {type(test_data), test_data.shape}')
    print(f'test_label: {type(test_label), test_label.shape}')
    print(f'set test label: {set(test_label)}')
    print(f'anomaly ratio: {len(test_abnormal_data) / len(test_data)}')
    print(f'All size: {len(train_data) + len(test_data)}')
    print('data splitting ========================================')

    return train_data, train_label, test_data, test_label


def get_embedding_path(dataset, model, ft_model, pooling):
    
    data_path = None
    label_path = None
    if dataset in ['20newsgroups', 'enron', 'imdb', 'reuters21578', 'sms_spam', 'sst2', 'wos']:
        
        if model in ['bert', 'glove_6b']:
            train_embedding_path = osp.join(EMBEDDING_DIR, f'{dataset}/{model}/train/{pooling}_embeddings.npy')
            train_label_path = osp.join(EMBEDDING_DIR, f'{dataset}/{model}/train/embedding_labels.npy')
            test_embedding_path = osp.join(EMBEDDING_DIR, f'{dataset}/{model}/test/{pooling}_embeddings.npy')
            test_label_path = osp.join(EMBEDDING_DIR, f'{dataset}/{model}/test/embedding_labels.npy')


        elif model == 'openai':
            train_embedding_path = osp.join(EMBEDDING_DIR, f'{dataset}/{model}/train/openai_{ft_model}_embeddings.npy')
            train_label_path = osp.join(EMBEDDING_DIR, f'{dataset}/{model}/train/openai_{ft_model}_embedding_labels.npy')
            test_embedding_path = osp.join(EMBEDDING_DIR, f'{dataset}/{model}/test/openai_{ft_model}_embeddings.npy')
            test_label_path = osp.join(EMBEDDING_DIR, f'{dataset}/{model}/test/openai_{ft_model}_embedding_labels.npy')
        
        elif model in ['Llama2-7b', 'Llama3-8b', 'Mistral-7b']:
                
            train_embedding_path = osp.join(EMBEDDING_DIR, f'{dataset}/{model}/train/{ft_model}_{pooling}_embeddings.npy')
            train_label_path = osp.join(EMBEDDING_DIR, f'{dataset}/{model}/train/{ft_model}_embedding_labels.npy')
            test_embedding_path = osp.join(EMBEDDING_DIR, f'{dataset}/{model}/test/{ft_model}_{pooling}_embeddings.npy')
            test_label_path = osp.join(EMBEDDING_DIR, f'{dataset}/{model}/test/{ft_model}_embedding_labels.npy')
        else:
            raise Exception(f'Unknown model name [{model}].')

        data_path = [train_embedding_path, test_embedding_path]
        label_path = [train_label_path, test_label_path]
    elif dataset in ['dbpedia14']:
        if model in ['bert', 'glove_6b']:
            train_embedding_path = osp.join(EMBEDDING_DIR, f'{dataset}/{model}/train/{pooling}_embeddings')
            train_label_path = osp.join(EMBEDDING_DIR, f'{dataset}/{model}/train/embedding_labels')
            test_embedding_path = osp.join(EMBEDDING_DIR, f'{dataset}/{model}/test/{pooling}_embeddings')
            test_label_path = osp.join(EMBEDDING_DIR, f'{dataset}/{model}/test/embedding_labels')
        
        elif model == 'openai':
            train_embedding_path = osp.join(EMBEDDING_DIR, f'{dataset}/{model}/train/{ft_model}/openai_{ft_model}_embeddings')
            train_label_path = osp.join(EMBEDDING_DIR, f'{dataset}/{model}/train/{ft_model}/openai_{ft_model}_embeddings_labels')
            test_embedding_path = osp.join(EMBEDDING_DIR, f'{dataset}/{model}/test/{ft_model}/openai_{ft_model}_embeddings')
            test_label_path = osp.join(EMBEDDING_DIR, f'{dataset}/{model}/test/{ft_model}/openai_{ft_model}_embeddings_labels')
        
        elif model in ['Llama2-7b', 'Llama3-8b', 'Mistral-7b']:
                
            train_embedding_path = osp.join(EMBEDDING_DIR, f'{dataset}/{model}/train/{ft_model}/{pooling}/{ft_model}_{pooling}_embeddings')
            train_label_path = osp.join(EMBEDDING_DIR, f'{dataset}/{model}/train/{ft_model}/{pooling}/{ft_model}_{pooling}_embeddings_labels')
            test_embedding_path = osp.join(EMBEDDING_DIR, f'{dataset}/{model}/test/{ft_model}/{pooling}/{ft_model}_{pooling}_embeddings')
            test_label_path = osp.join(EMBEDDING_DIR, f'{dataset}/{model}/test/{ft_model}/{pooling}/{ft_model}_{pooling}_embeddings_labels')
        else:
            raise Exception(f'Unknown model name [{model}].')

        data_path = [train_embedding_path, test_embedding_path]
        label_path = [train_label_path, test_label_path]


    return data_path, label_path


def find_valid_chunks(label_path, normal_class):

    train_valid_chunks = {}
    test_valid_chunks = {}

    train_path = label_path[0]
    test_path = label_path[1]

    
    # train labels
    num_valid_sample = 0
    for chunk in DBPEDIA14_CHUNK:
        train_chunk_path = train_path + '_' + chunk + '.npy'
        valid_index = find_valid_normal_sample(train_chunk_path, [normal_class])
        if valid_index is not None:
            train_valid_chunks.setdefault(chunk, valid_index)
            num_valid_sample += len(valid_index)
        if num_valid_sample == MAX_SIZE_SINGLE_TRAIN_CLASS_DBPEDIA14:
            break
    
    for chunk, num in train_valid_chunks.items():
        print(f'{chunk}: [{len(num)}]')
    print(f'train data: {num_valid_sample}')


    # test labels
    num_valid_sample = 0
    for chunk in DBPEDIA14_CHUNK:
        test_chunk_path = test_path + '_' + chunk + '.npy'
        valid_index = find_valid_normal_sample(test_chunk_path, DBPEDIA14_CLASS)
        if valid_index is not None:
            test_valid_chunks.setdefault(chunk, valid_index)
            num_valid_sample += len(valid_index)
        
        if num_valid_sample == MAX_SIZE_SINGLE_TEST_CLASS_DBPEDIA14 * len(DBPEDIA14_CLASS):
            break
    print(f'test data: {num_valid_sample}')
    
    return train_valid_chunks, test_valid_chunks


def find_valid_normal_sample(path, _class):

    labels = obj_load(path)
    valid_index = []
    for i, label in enumerate(labels):
        if label in _class:
            valid_index.append(i)
    if len(valid_index) == 0:
        valid_index = None
    
    return valid_index


def get_data_path(dataset_name):

    data_path = [osp.join(DATA_DIR, f'{dataset_name}/processed/train_data.list'),
                 osp.join(DATA_DIR, f'{dataset_name}/processed/test_data.list')]
    
    label_path = [osp.join(DATA_DIR, f'{dataset_name}/processed/train_label.list'),
                 osp.join(DATA_DIR, f'{dataset_name}/processed/test_label.list')]
    
    return data_path, label_path

def save_valid_data(dataset_name, normal_class, train_data, test_data, train_label, test_label):

    save_dir = new_dir(DATA_DIR, f'{dataset_name}/processed/valid_data/{normal_class}')

    obj_save(osp.join(save_dir, 'train_data.list'), train_data)
    obj_save(osp.join(save_dir, 'train_label.list'), train_label)
    obj_save(osp.join(save_dir, 'test_data.list'), test_data)
    obj_save(osp.join(save_dir, 'test_label.list'), test_label)
    print(f'train_data: {len(train_data)}')
    print(f'train_label: {len(train_label)}')
    print(f'test_data: {len(test_data)}')
    print(f'test_label: {len(test_label)}')

    exit(0)
