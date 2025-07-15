# Example dataset: sms_spam

# Shallow Machine Learning-based AD Algorithms
python main.py --dataset sms_spam --ad ocsvm --repeat 1
python main.py --dataset sms_spam --ad iforest --repeat 5
python main.py --dataset sms_spam --ad lof --repeat 1
python main.py --dataset sms_spam --ad pca --repeat 1
python main.py --dataset sms_spam --ad knn --repeat 1
python main.py --dataset sms_spam --ad kde --repeat 1
python main.py --dataset sms_spam --ad ecod --repeat 1


# Deep Learning-based AD Methods
python main.py --dataset sms_spam --ad ae --repeat 5
python main.py --dataset sms_spam --ad dsvdd --repeat 5
python main.py --dataset sms_spam --ad dpad --repeat 5