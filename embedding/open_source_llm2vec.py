__author__ = 'XF'
__date__ = '2024/07/23'

'''
Obtaining the text embedding using open-source LLMs.
'''
import tqdm
import numpy as np
import click
import torch
from llm2vec import LLM2Vec
from embedding.configs_llms import META_EMBEDDING, META_BASE_LLMs, LLMs2VEC_FT_LLMs, CHUNK_EMBEDDING
from torch.utils.data import DataLoader
from embedding.huggingface_login import hf_login
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from tools import new_dir, save_part_embedding, obj_save, osp

# Hugging Face login
# huggingface-cli login
# import following command to .bashrc
# export HF_ENDPOINT=https://hf-mirror.com 


def meta_text_embedding(data, targets, base_llm, ft_llm, pooling='mean', max_length=8191, batch_size=100, max_size=32, device=None, save_path=None, begin_position=0):

    assert data is not None and base_llm in META_EMBEDDING['models'] and pooling in META_EMBEDDING['pooling_mode']
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Base LLM: {META_BASE_LLMs[base_llm]}')
    print(f'Fine Tuned LLM: {LLMs2VEC_FT_LLMs[base_llm + "-" + ft_llm]}')
    Encoder = LLM2Vec.from_pretrained(base_model_name_or_path=META_BASE_LLMs[base_llm],
                                    peft_model_name_or_path=LLMs2VEC_FT_LLMs[base_llm + '-' + ft_llm],
                                    device_map=device,
                                    torch_dtype=torch.bfloat16,
                                    pooling_mode=pooling,
                                    max_length=max_length,    
                                    cache_dir=META_EMBEDDING['cache_dir'])

    
    print(f'Max Sentence Length: {Encoder.max_length}')
    print(f'Begin position:[{begin_position}]')
    if begin_position > len(data):
        return 0, []
    
    eos_token_embeddings = None
    mean_embeddings = None
    weighted_mean_embeddings = None
    num_error = 0
    error_pos = []
    valid_targets = []
    position = begin_position
    for i, begin_pos in enumerate(range(begin_position, len(data), batch_size)):
        print(f'Embedding .... [{begin_pos}]')
        if batch_size  > len(data) - batch_size * i:
            end_pos = -1
        else:
            end_pos = begin_pos + batch_size
        batch_data = data[begin_pos: end_pos]
        position += batch_size
        try: 
            if eos_token_embeddings is None:
                eos_token_embeddings, mean_embeddings, weighted_mean_embeddings = Encoder.encode(batch_data, batch_size=max_size)
            else:
                embeddings = Encoder.encode(batch_data, batch_size=max_size)
                eos_token_embeddings = np.concatenate((eos_token_embeddings, embeddings[0]))
                mean_embeddings = np.concatenate((mean_embeddings, embeddings[1]))
                weighted_mean_embeddings = np.concatenate((weighted_mean_embeddings, embeddings[2]))
        except Exception as e:
            print(f'Exception happen: [{e}]')
            for j, text in enumerate(batch_data):
                try:
                    if eos_token_embeddings is None:
                        eos_token_embeddings, mean_embeddings, weighted_mean_embeddings = Encoder.encode([text], batch_size=1)
                    else:
                        embeddings = Encoder.encode([text], batch_size=1)
                        eos_token_embeddings = np.concatenate((eos_token_embeddings, embeddings[0]))
                        mean_embeddings = np.concatenate((mean_embeddings, embeddings[1]))
                        weighted_mean_embeddings = np.concatenate((weighted_mean_embeddings, embeddings[2]))
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
            save_part_embedding(new_dir(save_path, f'{ft_llm}/eos_token'), f'{ft_llm}_eos_token_embeddings', position=position,
                                    embeddings=eos_token_embeddings,
                                    labels=valid_targets
                                    )
            save_part_embedding(new_dir(save_path, f'{ft_llm}/mean'), f'{ft_llm}_mean_embeddings', position=position,
                                    embeddings=mean_embeddings,
                                    labels=valid_targets
                                    )
            save_part_embedding(new_dir(save_path, f'{ft_llm}/weighted_mean'), f'{ft_llm}_weighted_mean_embeddings', position=position,
                                    embeddings=weighted_mean_embeddings,
                                    labels=valid_targets
                                    )
            eos_token_embeddings = None
            mean_embeddings = None
            weighted_mean_embeddings = None
            valid_targets = []
    
    # save the last part
    print(f'eos_token: {eos_token_embeddings.shape}')
    print(f'mean: {mean_embeddings.shape}')
    print(f'weighted_mean: {weighted_mean_embeddings.shape}')
    if len(data) <= CHUNK_EMBEDDING:

        obj_save(osp.join(save_path, f'{ft_llm}_eos_token_embeddings.npy'), eos_token_embeddings)
        obj_save(osp.join(save_path, f'{ft_llm}_mean_embeddings.npy'), eos_token_embeddings)
        obj_save(osp.join(save_path, f'{ft_llm}_weighted_mean_embeddings.npy'), eos_token_embeddings)
        obj_save(osp.join(save_path, f'{ft_llm}_embedding_labels.npy'), valid_targets)
    else:
        save_part_embedding(new_dir(save_path, f'{ft_llm}/eos_token'), f'{ft_llm}_eos_token_embeddings', position='end',
                                        embeddings=eos_token_embeddings,
                                        labels=valid_targets,
                                        )
        save_part_embedding(new_dir(save_path, f'{ft_llm}/mean'), f'{ft_llm}_mean_embeddings', position='end',
                                        embeddings=mean_embeddings,
                                        labels=valid_targets
                                        )
        save_part_embedding(new_dir(save_path, f'{ft_llm}/weighted_mean'), f'{ft_llm}_weighted_mean_embeddings', position='end',
                                        embeddings=weighted_mean_embeddings,
                                        labels=valid_targets
                                        )


    return num_error, error_pos
    



def mistral_text_embedding(model, pooling):

    pass

################################################################################
# Command line arguments
################################################################################
@click.command()
@click.option('--model', type=str, default=None)
@click.option('--dataset', type=str, default=click.Choice(['20newsgroups', '']))
@click.option('--pooling', type=str, default=click.Choice(['mean', 'max', 'tfidf']))

def main(model, dataset, pooling):

    # hf_login(token=HUGGINGFACE['token'], mirror=HUGGINGFACE['mirror'])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # l2v = LLM2Vec.from_pretrained(
    # "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
    # peft_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
    # device_map=device,
    # torch_dtype=torch.bfloat16,
    # cache_dir='/mntcephfs/lab_data/xiaofeng/data/Text_AD/LLMs/'
    
    # )
    
    # l2v = LLM2Vec.from_pretrained(
    # "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
    # peft_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-unsup-simcse",
    # device_map=device,
    # torch_dtype=torch.bfloat16,
    # cache_dir='/mntcephfs/lab_data/xiaofeng/data/Text_AD/LLMs/'
    # )
    # McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-supervised
    # l2v = LLM2Vec.from_pretrained(
    # "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
    # peft_model_name_or_path="McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-unsup-simcse",
    # device_map=device,
    # torch_dtype=torch.bfloat16,
    # cache_dir='/mntcephfs/lab_data/xiaofeng/data/Text_AD/LLMs/'
    # )
    
    l2v = LLM2Vec.from_pretrained(
    "McGill-NLP/LLM2Vec-Llama-2-7b-chat-hf-mntp",
    peft_model_name_or_path="McGill-NLP/LLM2Vec-Llama-2-7b-chat-hf-mntp-supervised",
    device_map=device,
    torch_dtype=torch.bfloat16,
    cache_dir='/mntcephfs/lab_data/xiaofeng/data/Text_AD/LLMs/'
    )
    

    # # Encoding queries using instructions
    # instruction = (
    #     "Given a web search query, retrieve relevant passages that answer the query:"
    # )
    # queries = [
    #     [instruction, "how much protein should a female eat"],
    #     [instruction, "summit define"],
    # ]
    # Text = [
    #     'Hello Large Lanugage model are secretly powerful Text Encoder',
    #     # 'Hello Text ADBench'
    # ]
    # q_reps = l2v.encode(Text)
    # Encoding queries using instructions
    instruction = (
        "Given a web search query, retrieve relevant passages that answer the query:"
    )
    queries = [
        [instruction, "how much protein should a female eat"],
        [instruction, "summit define"],
    ]
    q_reps = l2v.encode(queries)
    print(f'Embedding: {q_reps.shape}')



if __name__ == '__main__':

    main()

    pass