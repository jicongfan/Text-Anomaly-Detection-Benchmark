# -*- coding: utf-8 -*-

__author__ = 'XF'
__date__ = '2022/08/25'

'The Script is to provide some functional tools for core part.'

# built-in library
import re
import os
import os.path as osp
import time
import pickle
import json
import openpyxl
from copy import copy
from builtins import print as b_print

# import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

# third-party library
import torch as th
from argparse import ArgumentParser, ArgumentTypeError

# self-defined libarary
from configs import begining_line, ending_line, XL_TEMPLATE_PATH, RESULTS_DIR, XL_STRUCTURE, FT_MODEL, POOLING
from configs import RESULTS_DIR, MODEL, METRIC

class Log(object):

    def __init__(self, log_dir, log_name):
        self.log_path = osp.join(log_dir, generate_filename('.txt', *[log_name], timestamp=True))
        self.print(begining_line)
        self.print('date: %s' % time.strftime('%Y/%m/%d-%H:%M:%S'))
    
    def print(self, *args, end='\n'):

        with open(file=self.log_path, mode='a', encoding='utf-8') as console:
            b_print(*args, file=console, end=end)
        b_print(*args, end=end)
    
    @property
    def ending(self):
        self.print('date: %s' % time.strftime('%Y/%m/%d-%H:%M:%S'))
        self.print(ending_line)


def generate_filename(suffix, *args, sep='_', timestamp=False):

    '''

    :param suffix: suffix of file
    :param sep: separator，default '_'
    :param timestamp: add timestamp for uniqueness
    :param args:
    :return:
    '''

    filename = sep.join(args).replace(' ', '_')
    if timestamp:
        filename += time.strftime('_%Y%m%d%H%M%S')
    if suffix[0] == '.':
        filename += suffix
    else:
        filename += ('.' + suffix)

    return filename
    

# object serialization
def obj_save(path, obj):

    if obj is not None:
        with open(path, 'wb') as file:
            pickle.dump(obj, file)
    else:
        print('object is None!')


# object instantiation
def obj_load(path):

    if os.path.exists(path):
        with open(path, 'rb') as file:
            obj = pickle.load(file)
        return obj
    else:
        raise OSError('no such path:%s' % path)


def linux_command(command, info_file):

    assert isinstance(command, str)

    command = ' '.join([command, '>', info_file])

    os.system(command=command)


def new_dir(father_dir, mk_dir=None):

    if mk_dir is not None:
        new_path = osp.join(father_dir, mk_dir)
    else:   
        new_path = osp.join(father_dir, time.strftime('%Y%m%d%H%M%S'))
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    return new_path


def save_checkpoint(checkpoint_dict, save_path, myprint=None):

    assert checkpoint_dict['model'] is not None
    default_checkpoint = {
        'model': None,
        'optimizer': None,
        'epoch': None,
        'min_loss': None,
        'best_epoch': None
    }
    default_checkpoint.update(checkpoint_dict)
    th.save(default_checkpoint, save_path)
    if myprint is not None:
        myprint(f'model save in [{save_path}]')
    else:
        print(f'model save in [{save_path}]')


def load_checkpoint(checkpoint_path, model, optimizer):

    model_ckpt = th.load(checkpoint_path)
    
    model.load_state_dict(model_ckpt['model'])
    optimizer.load_state_dict(model_ckpt['optimizer'])

    return model, optimizer, model_ckpt['epoch'] + 1, model_ckpt['min_loss'], model_ckpt['best_epoch']


def create_argparser(default_args):

    parser = ArgumentParser()
    add_dict_to_argparser(parser, default_args)
    return parser


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError("boolean value expected")


def json_load(path):
    
    with open(path, 'r', encoding='utf8') as f:
        content = json.load(f)
    return content


def json_dump(path, dict_obj):

    with open(path, 'a+', encoding='utf-8') as f:
        json.dump(dict_obj, f, indent=4, ensure_ascii=False)


def stat_results(results):

    """
    params:: 
        results: [dict], {metric_name: [values, ...], ...}
    """
    stats = {}
    mean_results = {}
    for metric, values in results.items():
        mean = np.mean(values).item()
        std = np.std(values).item()
        stats.setdefault('Avg_' + metric, mean)
        if metric in ['auroc', 'auprc', 'f1', 'acc', 'fnr', 'fpr']:
            mean_results.setdefault(metric, mean)
        stats.setdefault('Std_' + metric, std)

    results.update(stats)
    return results, mean_results


def save_part_embedding(save_dir, name, position, embeddings, labels):

    save_path_embedding = osp.join(save_dir, f'{name}_{str(position)}.npy')
    save_path_label = osp.join(save_dir, f'{name}_labels_{str(position)}.npy')

    obj_save(save_path_embedding, embeddings)
    obj_save(save_path_label, labels)
    print(f'Save embedding [{position}] to [{save_dir}].')


def create_xl(path, ad_name, results):

    if not osp.exists(path):

        command = f'cp {XL_TEMPLATE_PATH} {path}'
        os.system(command)

    workbook = openpyxl.load_workbook(path)
    template_sheet = workbook['template']

    # copy the template sheet to a specific sheet
    target_sheet = workbook.copy_worksheet(template_sheet, ad_name)

    update_xl_structure()

    # write results to sheet
    for model, metrics in results.items():
        for metric, value in metrics.items():
            target_sheet[XL_STRUCTURE[model][metric]] = value

    workbook.save(path)


def update_xl_structure():

    
    base_model = 'Llama2-7b'
    mistral_structure = {}
    # Mistral-7b
    for ft_model in FT_MODEL['Mistral-7b']:
        for pooling in POOLING['Mistral-7b']:
            key = f'Mistral-7b_{ft_model}_{pooling}'
            base_key = f'{base_model}_{ft_model}_{pooling}'
            mistral_structure.setdefault(key, change_coordinate(XL_STRUCTURE[base_key], 9))


    llama3_structure = {}
    # Llama3-8b
    for ft_model in FT_MODEL['Llama3-8b']:
        for pooling in POOLING['Llama3-8b']:
            key = f'Llama3-8b_{ft_model}_{pooling}'
            base_key = f'{base_model}_{ft_model}_{pooling}'
            llama3_structure.setdefault(key, change_coordinate(XL_STRUCTURE[base_key], 18))
    
    XL_STRUCTURE.update(mistral_structure)
    XL_STRUCTURE.update(llama3_structure)


def change_coordinate(pos, b):

    new_pos = {}
    for key, value in pos.items():
        new_value = value[0] + str(int(value[1:]) + b)
        new_pos.setdefault(key, new_value)

    return new_pos


def collect_mean_results(ad, dataset):

    all_mean_results = {}

    for model in MODEL:
        for ft_model in FT_MODEL[model]:
            for pooling in POOLING[model]:
                key = f'{model}_{ft_model}_{pooling}'
                all_mean_results.setdefault(key, None)
                if ft_model is None:
                    results_dir = osp.join(RESULTS_DIR, f'{dataset}/{model}/')
                else:
                    results_dir = osp.join(RESULTS_DIR, f'{dataset}/{model}/{ft_model}')
                file_name = filter_file(ad, pooling=pooling, father_dir=results_dir)

                results = json_load(osp.join(results_dir, file_name))
                all_mean_results[key] = {
                    'auroc': results['Avg_auroc'],
                    'auprc': results['Avg_auprc'],
                    'f1': results['Avg_f1'],
                    'acc': results['Avg_acc'],
                    'fnr': results['Avg_fnr'],
                    'fpr': results['Avg_fpr']
                }
    return all_mean_results


def filter_file(ad, pooling, father_dir):

    file_list = os.listdir(father_dir)

    valid_file = f'results_{ad}_{pooling}'

    for file_name in file_list:
        if re.search(valid_file, file_name) is not None:
            valid_file = file_name
            break
    return valid_file


if __name__ == '__main__':
    pass