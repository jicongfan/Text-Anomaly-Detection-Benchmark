# Example dataset: sms_spam


# Llama3 =======================================
# Llama3-8b-mntp
python main.py --dataset sms_spam --model_from meta --model Llama3-8b --ft_llm mntp --batch_size 5000 --max_size 28

# Llama3-8b-mntp-unsup-simcse
python main.py --dataset sms_spam --model_from meta --model Llama3-8b --ft_llm mntp-unsup-simcse --batch_size 5000 --max_size 28

# Llama3-8b-mntp-supervised
python main.py --dataset sms_spam --model_from meta --model Llama3-8b --ft_llm mntp-supervised --batch_size 5000 --max_size 28



# Llama2 ======================================
# Llama2-7b-mntp
python main.py --dataset sms_spam --model_from meta --model Llama2-7b --ft_llm mntp --batch_size 5000 --max_size 28

# Llama2-7b-mntp-unsup-simcse
python main.py --dataset sms_spam --model_from meta --model Llama2-7b --ft_llm mntp-unsup-simcse --batch_size 5000 --max_size 28

# Llama2-7b-mntp-supervised
python main.py --dataset sms_spam --model_from meta --model Llama2-7b --ft_llm mntp-supervised --batch_size 5000 --max_size 28



# Mistral ======================================
# Mistral-7b-mntp
python main.py --dataset sms_spam --model_from meta --model Mistral-7b --ft_llm mntp --batch_size 5000 --max_size 28


# Mistral-7b-mntp-unsup-simcse
python main.py --dataset sms_spam --model_from meta --model Mistral-7b --ft_llm mntp-unsup-simcse --batch_size 5000 --max_size 28


# Mistral-7b-mntp-supervised
python main.py --dataset sms_spam --model_from meta --model Mistral-7b --ft_llm mntp-supervised --batch_size 5000 --max_size 28


# OpenAI
python main.py --dataset sms_spam --model_from openai --model text-embedding-3-small --batch_size 1000
python main.py --dataset sms_spam --model_from openai --model text-embedding-ada-002 --batch_size 1000
python main.py --dataset sms_spam --model_from openai --model text-embedding-3-large --batch_size 1000
