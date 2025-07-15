__author__ = 'XF'
__date__ = '2024/07/17'

'''
The API of OpenAI LLMs.
'''
import time
import numpy as np
from openai import OpenAI
from embedding.configs_llms import OPENAI_EMBEDDING


def get_text_embedding(texts, model='text-embedding-3-small'):

    assert texts is not None and model in OPENAI_EMBEDDING['models']

    if isinstance(texts, str):
        texts = [texts]
    # print(f'Text Embedding ==========================')
    # print(f'model: [{model}]')
    # print(f'Number of Text: {len(texts)}')
    start_time = time.time()
    client = OpenAI(api_key=OPENAI_EMBEDDING['api_key'], base_url=OPENAI_EMBEDDING['request_url'])
    response = client.embeddings.create(input=texts, model=model, encoding_format='float')
    response_time = time.time() - start_time
    text_embeddings = np.stack([response.data[i].embedding for i in range(len(texts))])
    print(f'Embedding: {text_embeddings.shape}')
    print(f'Response time: {response_time} s')
    return text_embeddings