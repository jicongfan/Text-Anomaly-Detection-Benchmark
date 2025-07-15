__author__ = 'XF'
__date__ = '2024/07/17'

'''
Text Embedding using Large Language Models, such as [Llama2, Llama3, ...]
'''
from os import path as osp
import numpy as np
from embedding.openai_api import get_text_embedding as openai_text_embedding
from embedding.open_source_llm2vec import meta_text_embedding
from tools import obj_save, new_dir, save_part_embedding
# from embedding.configs_llms import HUGGINGFACE
# from embedding.huggingface_login import hf_login
from embedding.configs_llms import CHUNK_EMBEDDING


def openai_llms(model, train_data, train_target, test_data, test_target, train_path, test_path, request_size, begin_position=0):

    
    for data, targets, save_path, bp in zip([train_data, test_data], [train_target, test_target], [train_path, test_path], [begin_position, 0]):
        print(f'data size: [{len(data)}]')
        embeddings = None
        num_error = 0
        error_pos = []
        valid_targets = []
        position = bp
        print(f'Begin position: [{bp}]')
        if bp > len(data):
            continue


        encoder = openai_text_embedding
        for i, begin_pos in enumerate(range(bp, len(data), request_size)):
            print(f'Embedding .... [{begin_pos}]')
            if request_size  > len(data) - request_size * i:
                end_pos = -1
            else:
                end_pos = begin_pos + request_size
            batch_data = data[begin_pos: end_pos]
            position += request_size
            try: 
                if embeddings is None:
                    embeddings = encoder(texts=batch_data, model=model)
                else:
                    embeddings = np.concatenate((embeddings, encoder(texts=batch_data, model=model)))
            except Exception:
                for j, text in enumerate(batch_data):
                    try:
                        if embeddings is None:
                            embeddings = encoder(texts=text, model=model)
                        else:
                            embeddings = np.concatenate((embeddings, encoder(texts=text, model=model)))
                    except Exception as e:
                        print(f'Request error: [{e}]')
                        print(f'Error [{begin_pos + j}]: {text}')
                        num_error += 1
                        error_pos.append(begin_pos + j)
                        continue
                    valid_targets.append(targets[begin_pos + j])
                continue
            valid_targets.extend(targets[begin_pos: end_pos])
            if position % CHUNK_EMBEDDING == 0:
                save_part_embedding(new_dir(save_path, f'{model}'), f'openai_{model}_embeddings', position=position,
                                    embeddings=embeddings,
                                    labels=valid_targets
                                    )
                embeddings = None
                valid_targets = []


        print(f'Embedding: {embeddings.shape}')
        print(f'Targets: {len(valid_targets)}')
        print(f'request error: {num_error}')
        print(f'error position: {error_pos}')
        
        # save the last part
        # save_part_embedding(new_dir(save_path, f'{model}'), f'openai_{model}_embeddings', position='end',
        #                             embeddings=embeddings,
        #                             labels=valid_targets
        #                             )

        obj_save(osp.join(save_path, f'openai_{model}_embeddings.npy'), embeddings)
        obj_save(osp.join(save_path, f'openai_{model}_embedding_labels.npy'), valid_targets)
        print(f'Saving embedding to [{save_path}]!')


def opensource_llms(base_llm, ft_llm, train_data, train_target, test_data, test_target, pooling, train_path, test_path, request_size, max_size=32, device=None, begin_position=0):

    # hf_login(token=HUGGINGFACE['token'], mirror=HUGGINGFACE['mirror'])
    for data, targets, save_path, bp in zip([train_data, test_data], [train_target, test_target], [train_path, test_path], [begin_position, 0]):

        # print(f'data size: [{len(data)}]')
        # embeddings, valid_targets, num_error, error_pos = meta_text_embedding(
        #     data, targets, base_llm, ft_llm, pooling=pooling, batch_size=request_size, max_size=max_size, device=device, save_path=save_path, begin_position=begin_position)
        
        # eos_token_embedding, mean_embedding, weighted_mean_embedding = embeddings
        # print(f'Eos_token_Embedding: {eos_token_embedding.shape}')
        # print(f'Mean_Embedding: {mean_embedding.shape}')
        # print(f'Weighted_mean_Embedding: {weighted_mean_embedding.shape}')
        # print(f'Targets: {len(valid_targets)}')
        # print(f'request error: {num_error}')
        # print(f'error position: {error_pos}')

        num_error, error_pos = meta_text_embedding(
            data, targets, base_llm, ft_llm, pooling=pooling, batch_size=request_size, max_size=max_size, device=device, save_path=save_path, begin_position=bp)
        
        print(f'request error: {num_error}')
        print(f'error position: {error_pos}')
        

        # obj_save(osp.join(save_path, f'{ft_llm}_eos_token_embeddings.npy'), eos_token_embedding)
        # obj_save(osp.join(save_path, f'{ft_llm}_mean_embeddings.npy'), mean_embedding)
        # obj_save(osp.join(save_path, f'{ft_llm}_weighted_mean_embeddings.npy'), weighted_mean_embedding)
        # obj_save(osp.join(save_path, f'{ft_llm}_embedding_labels.npy'), valid_targets)
        print(f'Saving embedding to [{save_path}]!')
