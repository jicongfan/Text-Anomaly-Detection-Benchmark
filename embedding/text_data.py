__author__ = 'XF'
__date__ = '2024/09/09'

'''
Text data preprocess for text embedding.
'''


from os import path as osp
from configs import DATA_DIR
from tools import obj_load


def get_text_data(dataset):

    data_dir = osp.join(DATA_DIR, f'{dataset}/processed')
    train_data = obj_load(osp.join(data_dir, 'train_data.list'))
    train_label = obj_load(osp.join(data_dir, 'train_label.list'))
    test_data = obj_load(osp.join(data_dir, 'test_data.list'))
    test_label = obj_load(osp.join(data_dir, 'test_label.list'))

    print(f'data info =================')
    print(f'train set: {len(train_data)}')
    print(f'test set: {len(test_data)}')
    print(f'All data size: {len(train_data) + len(test_data)}')
    print(f'============================')

    return train_data, train_label, test_data, test_label


