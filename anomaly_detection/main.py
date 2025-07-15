__author__ = 'XF'
__date__ = '2024/09/09'

'''
Anomaly detection.
'''

import time
import torch
from pyod.models.ocsvm import OCSVM
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.knn import KNN
from pyod.models.pca import PCA
from pyod.models.ecod import ECOD
from pyod.models.kde import KDE
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.deep_svdd import DeepSVDD
from pyod.models.anogan import AnoGAN
from pyod.models.vae import VAE
from .dpad import DPAD
from .IGAN import InverseGAN

from configs import AD_ALGORITHMS, ARGUMENTS_DIR, osp, CONTAMINATION, CLASSIC_ML_ALGORITHMS
from tools import json_load

# RGP, IGAN, DPAD


from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, f1_score, accuracy_score, recall_score

from anomaly_detection.utils import get_data


class AnomalyDetection(object):

    def __init__(self, ad, dataset, device=None, repeat=1, print=None, **kw):
        
        self.ad = ad
        self.dataset = dataset
        self.repeat = repeat
        self.kw = kw
        self.print = print
        if device is not None:
            self.device = device
        else:
            device  = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.results = {
            'auroc': [],
            'auprc': [],
            'f1': [],
            'acc': [],
            'fpr': [],
            'fnr':[],
            'train_time': [],
            'test_time': []
        }

        self.train_data = None
        self.train_label = None
        self.test_data = None
        self.test_label = None
        self.n_feaures = None
        self.prepare()

        self.print(f'AD Method: {ad} =================================')
        for _ in range(1, self.repeat + 1):
            self.print(f'============= [{_}-th] repeat ============')
            self.forward()

    def prepare(self):
        
        self.train_data, self.train_label, self.test_data, self.test_label, self.n_feaures = get_data(self.dataset, base_model=self.kw['model'], ft_model=self.kw['ft_model'], pooling=self.kw['pooling'], normal_class=self.kw['normal_class'])

    def evaluation(self, predict_score, predict_label):

        auroc = roc_auc_score(self.test_label, predict_score)
        precision, recall, _ = precision_recall_curve(self.test_label, predict_score)
        auprc = auc(recall, precision)
        f1 = f1_score(self.test_label, predict_label)
        acc = accuracy_score(self.test_label, predict_label)
        self.results['auroc'].append(auroc)
        self.results['auprc'].append(auprc)
        self.results['f1'].append(f1)
        self.results['acc'].append(acc)
        self.results['fnr'].append(false_negative_rate(self.test_label, predict_label))
        self.results['fpr'].append(false_positive_rate(self.test_label, predict_label))

    
    def forward(self):
 
        model = self.init_model(self.ad)

        start = time.time()
        # training 
        model.fit(self.train_data)
        end = time.time()
        self.results['train_time'].append(end - start)
        self.print(f'Train time: [{self.results["train_time"][-1]}]')

        # testing
        predict_score = model.decision_function(self.test_data)
        self.results['test_time'].append(time.time() - end)
        self.print(f'Test time: [{self.results["test_time"][-1]}]')
        predict_label = []
        for score in predict_score:
            if score >= model.threshold_:
                predict_label.append(1)
            else:
                predict_label.append(0)

        # evaluation
        self.evaluation(predict_score, predict_label)


    def init_model(self, ad):

        # load arguments
        if self.kw['hps'] is not None:
            args = json_load(osp.join(ARGUMENTS_DIR, f'{self.kw["hps"]}.json'))
        else:
            if ad in ['ocsvm', 'iforest', 'knn', 'pca', 'lof', 'kde', 'ecod']:
                args = json_load(osp.join(ARGUMENTS_DIR, f'{ad}.json'))
            elif ad in ['ae', 'dsvdd', 'dpad']:
                args = json_load(osp.join(ARGUMENTS_DIR, f'{ad}_{self.kw["arg_model"]}.json'))
        args.setdefault('contamination', CONTAMINATION)
        # classic machine learning AD algorithm
        if ad == 'ocsvm':
            model = OCSVM
        elif ad == 'iforest':
            model = IForest
        elif ad == 'knn':
            model = KNN
        elif ad == 'pca':
            model = PCA
        elif ad == 'lof':
            model = LOF
        elif ad == 'ecod':
            model = ECOD
        elif ad == 'kde':
            model = KDE
            args.setdefault('kernel', self.kw['kernel'])
        elif ad == 'ae':
            model = AutoEncoder
        elif ad == 'dsvdd':
            model = DeepSVDD
            args.setdefault('n_features', self.n_feaures)
        elif ad == 'dpad':
            model = DPAD
            args.setdefault('n_features', self.n_feaures)
        else:
            raise Exception(f'Unknown anomaly detection method [{ad}]!')

        return model(**args)


def false_positive_rate(true_label, pred_label):

    # FPR = FP / (TN + FP)
    invert_true_label = [1 if e == 0 else 0 for e in true_label]
    invert_pred_label = [1 if e == 0 else 0 for e in pred_label]

    recall = recall_score(invert_true_label, invert_pred_label)

    return 1 - recall


def false_negative_rate(true_label, pred_label):

    recall = recall_score(true_label, pred_label)

    return 1 - recall

