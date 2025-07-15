# -*- coding: utf-8 -*-
__author__ = 'XF'
__date__ = '2024/09/02'


'''
The entrance of text anomaly detection via LLMs.
'''

import click

from embedding.main import TextEmbedding
from anomaly_detection.main import AnomalyDetection

from configs import osp, ARGUMENTS_DIR, LOG_DIR, RESULTS_DIR, DATASETS, MODEL, FT_MODEL, POOLING
from tools import json_load, json_dump, Log, stat_results, new_dir, generate_filename, create_xl, collect_mean_results
################################################################################
# Command line arguments
################################################################################
@click.command()
@click.option('--dataset', type=str, default=click.Choice([DATASETS]))
@click.option('--device', type=str, default=None, help='[cpu, cuda:0, cuda:1, ...]')
# arguments for text embedding
@click.option('--model_from', type=click.Choice(['openai', 'meta', 'google']), default=None)
@click.option('--model', type=str, default=None, help='base model')
@click.option('--ft_llm', type=str, default=None, help='fine-tuned llm')
@click.option('--pooling', type=click.Choice(['cls_token', 'eos_token', 'mean', 'weighted_mean', 'tf_idf']), default='mean')
@click.option('--batch_size', type=int, default=100)
@click.option('--max_size', type=int, default=32, help='The max size of 4GPUs multiprocess to infer the sentence')
@click.option('--begin_position', type=int, default=0, help='The begin sequence index for embedding.')
# arguments for anomaly detection
@click.option('--ad', type=str, default=None, help='anomaly detection method')
@click.option('--normal_class', type=int, default=None, help='normal_class')
@click.option('--rsd', type=str, default=None, help='results save dir')
@click.option('--wxl', type=bool, default=False, help='collect the mean results to excel file.')
@click.option('--hps', type=str, default=None, help='hyperparameters of anomaly detection method')
@click.option('--kernel', type=str, default='gaussian', help='kernel for KDE')
@click.option('--repeat', type=int, default=1, help='The repeat time.')

def main(dataset, device, model_from, model, ft_llm, pooling, batch_size, max_size, begin_position, ad, normal_class, rsd, wxl, hps, kernel, repeat):

    # load arguments
    te_args = {
        'dataset': dataset,
        'device': device,
        # '======== Text Embedding arguemnts ========': '',
        'model_from': model_from,
        'model': model,
        'ft_llm': ft_llm,
        'pooling': pooling,
        'batch_size': batch_size,
        'max_size': max_size,
        'begin_position': begin_position
    }
    ad_args = {
        'dataset': dataset,
        'device': device,
        # '======== Anomaly Detection arguemnts ========': '',
        'ad': ad,
        'normal_class': normal_class,
        'repeat': repeat,
        'model': None,
        'hps': hps,
        'kernel': kernel,
        'ft_model': None,
        'pooling': None,
        'print': None,
        'arg_model': None
    }    

    # log
    logger = Log(new_dir(LOG_DIR, dataset), dataset)
    print = logger.print

    # show details
    print(f'Dataset: {dataset}')
    if device is not None:
        print(f'Device: [{device}]')


    # text embedding
    if ad is None and model is not None:
        print(f'Model From : {model_from}')
        print(f'Model: {model}')
        if ft_llm is not None:
            print(f'Fine-tuned Model: {ft_llm}')
        print(f'Pooling: {pooling}')
        print(f'Batch size: {batch_size}')
        print(f'Max size: {max_size}')
        print(f'######## Text embedding... ########')
        te = TextEmbedding(**te_args)
        te.embedding()
        print(f'Text embedding time: [{te.time}s]')
    

    # anomaly detection
    if ad is not None:
        # print(f'AD method: {ad}')
        # ad_args.update(json_load(ARGUMENTS_DIR, osp.join(ad, f'{dataset}.json')))
        if normal_class is None:
            results_dir = new_dir(RESULTS_DIR, f'{dataset}')
        else:
            results_dir = new_dir(RESULTS_DIR, f'{dataset}_{normal_class}')
        if wxl:
            print(f'######## Collect results #########')
            all_mean_results = collect_mean_results(ad, dataset)
            mean_results_path = osp.join(results_dir, f"{generate_filename('.xlsx', *['results', dataset, 'collect'], timestamp=False)}")
        else:
            print(f'######## Anomaly detection... ########')
            models = MODEL
            all_mean_results = {}
            if hps is None:
                mean_results_path = osp.join(results_dir, f"{generate_filename('.xlsx', *['results', dataset], timestamp=True)}")
            else:
                mean_results_path = osp.join(results_dir, f"{generate_filename('.xlsx', *['results', dataset, hps.split('/')[-1]], timestamp=True)}")
            if model is not None:
                models = [model]
            for model in models:
                for ft_model in FT_MODEL[model]:
                    if ft_llm is not None:
                        if ft_model != ft_llm:
                            continue
                    for pooling in POOLING[model]:
                        
                        ad_args['model'] = model
                        ad_args['ft_model'] = ft_model
                        ad_args['pooling'] = pooling
                        ad_args['print'] = logger.print
                        
                        key = f'{model}_{ft_model}_{pooling}'
                        all_mean_results.setdefault(key, None)
                        print(f'[{model}]->[{ft_model}]->[{pooling}]')
                        # save dir
                        if model in ['openai', 'Llama2-7b', 'Llama3-8b', 'Mistral-7b'] :
                            save_dir = new_dir(results_dir, f'{model}/{ft_model}')
                            ad_args['arg_model'] = 'llm'
                        elif model in ['bert', 'glove_6b']:
                            save_dir = new_dir(results_dir, f'{model}')
                            ad_args['arg_model'] = 'bert'

                        save_path = osp.join(save_dir, generate_filename('.json', *['results', ad, str(pooling)], timestamp=True))
                        adm = AnomalyDetection(**ad_args)
        
                        results, mean_results = stat_results(adm.results)
                        json_dump(save_path, results)
                        all_mean_results[key] = mean_results

        print(f'Anomaly detection time: [{results["time"]}s]')
        create_xl(path=mean_results_path, ad_name=ad, results=all_mean_results)                 


        # save results
        save_path = osp.join(save_dir, generate_filename('.json', *['results', ad, dataset], timestamp=True))
        json_dump(save_path, stat_results(results))

    # save arguments
    te_args.update(ad_args)
    json_dump(osp.join(save_dir, 'arguments.json'), te_args)


    logger.ending


if __name__ == '__main__':

    main()