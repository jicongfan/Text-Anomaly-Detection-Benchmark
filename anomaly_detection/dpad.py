import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
from torch.utils.data import Dataset
from numpy import percentile
from pyod.utils.torch_utility import get_activation_by_name


class CustomDataset(Dataset):
    def __init__(self,
                 X,
                 y):
        self.data = X
        self.targets = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return torch.from_numpy(self.data[idx]), (self.targets[idx])


class FourLayer(nn.Module):
    def __init__(self,
                 n_features=2,
                 num_classes=1,
                 hidden_neurons=None,
                 hidden_activation='relu'
                 ):
        super(FourLayer, self).__init__()
        self.n_features = n_features
        self.num_classes = num_classes
        self.hidden_neurons = hidden_neurons
        self.hidden_activation = hidden_activation

        activ = nn.ReLU(True)

        # for text anomaly detection
        # activ = nn.Tanh()
        # ==========================

        # self.feature_extractor = nn.Sequential(OrderedDict([
        #     ('fc1', nn.Linear(self.input_dim, self.num_hidden_nodes[0])),
        #     ('relu1', activ),
        #     ('fc2', nn.Linear(self.num_hidden_nodes[0], self.num_hidden_nodes[1])),
        #     ('relu2', activ),
        # ]))
        self.feature_extractor = nn.Sequential()
        self.feature_extractor.add_module('input_layer',
                          nn.Linear(self.n_features, self.hidden_neurons[0]))
        self.feature_extractor.add_module('hidden_activation_e0',
                          get_activation_by_name(self.hidden_activation))
        for i in range(1, len(self.hidden_neurons) - 1):
            self.feature_extractor.add_module(f'hidden_layer_e{i}',
                              nn.Linear(self.hidden_neurons[i - 1], self.hidden_neurons[i]))
            self.feature_extractor.add_module(f'hidden_activation_e{i}',
                              get_activation_by_name(self.hidden_activation))
        
        self.feature_extractor.add_module(f'net_output', nn.Linear(self.hidden_neurons[-2],
                                                   self.hidden_neurons[-1]))
        # self.feature_extractor.add_module(f'hidden_activation_e{len(self.hidden_neurons)}',
        #                   get_activation_by_name(self.hidden_activation))


        self.size_final = self.hidden_neurons[-1]

        self.classifier = nn.Sequential(OrderedDict([
            ('fc3', nn.Linear(self.hidden_neurons[-1], int(self.hidden_neurons[-1] / 2))),
            ('relu3', activ),
            ('fc4', nn.Linear(int(self.hidden_neurons[-1] / 2), self.num_classes))]))

        # self.lamda = nn.Parameter(0 * torch.ones([1, 1]))
        # self.inp_lamda = nn.Parameter(0 * torch.ones([1, 1]))

    def forward(self, input):
        # features = self.feature_extractor(input)
        # logits = self.classifier(features.view(-1, self.size_final))
        logits = self.feature_extractor(input)
        return logits

    def half_forward_start(self, input):
        return self.feature_extractor(input)

    def half_forward_end(self, input):
        return self.classifier(input.view(-1, self.size_final))


class DPAD:
    def __init__(self, gamma, lamb, k, hidden_neurons, num_classes=128,
               bs=8192, n_epochs=200, learning_rate=1e-3, adam=1, device=None, 
               contamination=0.1, n_features=None, hidden_activation='relu'):
        self.gamma, self.lamb, self.k, self.hidden_dims = \
             gamma, lamb, k, hidden_neurons,
        self.num_classes = num_classes

        self.bs = bs
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.adam = adam
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.contamination = contamination

        # train_dataset = CustomDataset(train_x, np.zeros(train_x.shape[0]))
        # test_dataset = CustomDataset(test_x, test_y)

        # self.train_loader = DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True,
        #                           num_workers=0)
        # self.test_loader = DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False,
        #                          num_workers=0)
        # n_dim = train_x.shape[-1]

        self.net = FourLayer(n_features=n_features, hidden_neurons=hidden_neurons)

        self.train_c = None

    def fit(self, X, y=None):

        train_dataset = CustomDataset(X, np.zeros(X.shape[0]))
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=self.bs, shuffle=True,
                                  num_workers=0)
        model = self.net.to(self.device)
        # model.device = device

        weight_decay = 0

        if self.adam == 1:
            optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate, weight_decay=weight_decay, amsgrad=1)
        if self.adam == 0:
            optimizer = torch.optim.SGD(self.net.parameters(), lr=self.learning_rate, weight_decay=weight_decay, momentum=0.9)

        best_loss = 1000000

        r = self.gamma
        punish = self.lamb


        for epoch in range(self.n_epochs):
            # ---------- Training ----------
            # Make sure the model is in train mode before training.
            # model.train()
            # These are used to record information in training.
            # train_loss = []
            epoch_loss  = 0
            all_dist = []
            exp_dist = []
            # Iterate the training set by batches.
            for batch in self.train_loader:
                model = model.to(self.device)
                model.train()
                loss = 0.0
                # A batch consists of image data and corresponding labels.
                imgs, _ = batch
                imgs = imgs.float().to(self.device)
                optimizer.zero_grad()
                outputs = model(imgs)
                rs = 0
                dists = torch.cdist(outputs.view(outputs.shape[0], -1), outputs.view(outputs.shape[0], -1), p=2)
                dists = torch.pow(dists, 2)
                real_dists = torch.sum(dists, dim=1)
                real_dists = torch.sum(real_dists, dim=0)
                exp = torch.exp(dists * (-r)).detach()
                # t_dis=dists=torch.pow((1+dists/r), -(r+1)/2)
                dists = exp * dists
                # dists=torch.pow((1+dists/r), -(r+1)/2)*dists
                dists = torch.sum(dists, dim=1)
                # print(dists)
                # break
                dists = torch.sum(dists, dim=0)
                loss = dists / self.bs
                for _, param in model.named_parameters():
                    rs += abs(torch.norm(param, p=2) - 1)
                rs = rs * punish
                loss = loss + rs
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                # train_loss.append(loss)
            print(f"Epoch {epoch + 1}/{self.n_epochs}, Loss: {epoch_loss}")

        self.get_train_c()
        self.decision_scores_ = self.decision_function(X)
        self.threshold_ = percentile(self.decision_scores_,
                                    100 * (1 - self.contamination))
        # auroc, aupr, f1, train_score_list = testing_while_train(model=model, batchsize=256, nn=nn, percent=percent,
        #                                                         train_loader=train_loader, test_loader=test_loader)
        # # if auroc > best_auroc:
        # #     best_auroc = auroc
        # #     best_auc_socres = train_score_list
        # # if aupr > best_pr:
        # #     best_pr = aupr
        # # if f1 > best_f1:
        # #     best_f1 = f1
        #
        # # if epoch==n_epochs-1:
        # #     return 0
        # return auroc, aupr, f1, train_score_list

    def get_train_c(self):
        self.net.eval()
        c = []
        for x,y in self.train_loader:
            x, y = x.float().to(self.device), y.to(self.device)
            with torch.no_grad():
                #pred = model.half_forward_start(x)
                pred = self.net(x)
                c += pred.detach()
        c = torch.stack(tuple(c))
        self.train_c = c

    def decision_function(self, test_x):
        # bz = batchsize
        # knn = nn
        nn = self.k
        model = self.net
        # device = 'cuda:0'
        # model = model.to(device)
        # model.device = device
        test_set = torch.utils.data.TensorDataset(torch.Tensor(test_x), torch.zeros(test_x.shape[0]))
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1024, shuffle=False, num_workers=0)

        model.eval()
        test_dists_all, targets = [], []
        for batch in test_loader:
            x, _ = batch
            x = x.float().to(self.device)

            with torch.no_grad():
                pred = model(x)
                pred = pred.detach()

            # test_c = torch.stack(tuple(preds))
            test_c = pred
            # print(c.shape)
            # print(test_c.shape)
            test_dists = torch.cdist(test_c.view(test_c.shape[0], -1), self.train_c.view(self.train_c.shape[0], -1), p=2)
            test_dist_sorted, indices = torch.topk(test_dists, k=nn, dim=1, largest=False)
            test_dists = torch.sum(test_dist_sorted, dim=1) / nn
            test_dists_all.append(test_dists)

        test_dists_all = torch.cat(tuple(test_dists_all)).cpu().numpy()
        return test_dists_all

        # targets = torch.stack(targets)
        # roc_auc = roc_auc_score(targets, test_dists_all)
        # precision, recall, _ = precision_recall_curve(targets, test_dists_all)
        # auc_pr = average_precision_score(targets, test_dists_all)
        # # recall of abnormal data ================================
        #
        # # calculate f1
        # f1 = f1_calculator(targets, test_dists_all)
        # # threshold = 0.5
        # # pred_lab = np.where(test_dists_all > threshold, 1, 0)
        # # recall_num = 0
        # # for pl, l in zip(pred_lab, targets):
        # #     if pl == 1 and l == 1:
        # #         recall_num += 1
        # # recall_rate = 2 * recall_num / len(targets)
        # # print(fpr)
        # #roc_auc = auc(fpr, tpr)
        # return roc_auc, auc_pr, f1, train_score_list
