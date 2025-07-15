__author__ = 'XF'
__date__ = '2024/07/17'

'''
The configurations of API of LLMs.
'''


# basic configuration for LLMS used to embedding text.
OPENAI_EMBEDDING = {
    # 'request_url': 'https://api.apiyi.com/v1/', # esay to happen [request timed out]
    'request_url': 'http://8.218.238.241:17935/v1',
    # 'request_url': 'https://vip.aipyi.com/v1', # [connection error]
    'api_key': 'fill your API Key here',
    'models': ['text-embedding-3-small', 'text-embedding-3-large', 'text-embedding-ada-002']

}
# dimension text-embedding-3-large: 3072
# dimension text-embedding-3-small: 1536
# dimension text-embedding-ada-002: 1536


# basic configuration for LLaMA
META_EMBEDDING = {

    'models': ['Llama3-8b', 'Llama2-7b', 'Llama-1.3b', 'Mistral-7b'],
    'cache_dir': '/mnt/data/xiaofeng/data/Text_AD/language_models/LLMs/',
    'pooling_mode': ['eos_token', 'mean', 'weighted_mean']
}

META_BASE_LLMs = {
    'Llama3-8b': 'McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp',
    'Llama2-7b': 'McGill-NLP/LLM2Vec-Llama-2-7b-chat-hf-mntp',
    'Llama-1.3b': '',
    'Mistral-7b': 'McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp'
}

LLMs2VEC_FT_LLMs = {
    # Llama3-8b
    'Llama3-8b-mntp': 'McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp',
    'Llama3-8b-mntp-unsup-simcse': 'McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-unsup-simcse',
    'Llama3-8b-mntp-supervised': 'McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised',
    
    # Llama2-7b
    'Llama2-7b-mntp': 'McGill-NLP/LLM2Vec-Llama-2-7b-chat-hf-mntp',
    'Llama2-7b-mntp-unsup-simcse': 'McGill-NLP/LLM2Vec-Llama-2-7b-chat-hf-mntp-unsup-simcse',
    'Llama2-7b-mntp-supervised': 'McGill-NLP/LLM2Vec-Llama-2-7b-chat-hf-mntp-supervised',

    #Mistral-7b
    'Mistral-7b-mntp': 'McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp',
    'Mistral-7b-mntp-unsup-simcse': 'McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-unsup-simcse',
    'Mistral-7b-mntp-supervised': 'McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-supervised',
}

LM_CACHE_DIR = {
    'bert': '/mnt/data/xiaofeng/data/Text_AD/language_models/bert',
    'sbert': '/mnt/data/xiaofeng/data/Text_AD/language_models/sbert',
    'glove_6b': '/mnt/data/xiaofeng/data/Text_AD/language_models/Glove_6B'
}

HUGGINGFACE = {
    'token': 'fill your Huggingface Token here',
    'mirror': 'https://hf-mirror.com',
}

# embedding save path
SAVE_PATH = '/mnt/data/xiaofeng/data/Text_AD/embedding'

# maximum embedding size for prevent out of memory
CHUNK_EMBEDDING = 50000