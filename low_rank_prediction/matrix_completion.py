import click
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_percentage_error as mape

from tools import json_dump, generate_filename, osp

from low_rank_prediction.performace_matrices import auroc_20newsgroups, auroc_imdb, auroc_enron, auroc_reuters21578
from low_rank_prediction.performace_matrices import auroc_dbpedia_0, auroc_dbpedia_1, auroc_dbpedia_2, auroc_dbpedia_3, auroc_dbpedia_4
from low_rank_prediction.performace_matrices import auroc_sms_spam, auroc_sst2, auroc_wos


class MatrixCompletion(nn.Module):
    def __init__(self, M_obs, rank, init='random'):
        super().__init__()
        if init == 'random':
            m, n = M_obs.shape
            U = torch.randn(m, rank, dtype=torch.float32)
            V = torch.randn(n, rank, dtype=torch.float32)
        elif init == 'svd':
            U, S, V = np.linalg.svd(M_obs, full_matrices=False)
            U = U[:, :rank] @ np.diag(np.sqrt(S[:rank]))
            V = V[:, :rank] @ np.diag(np.sqrt(S[:rank]))
            U = torch.tensor(U, dtype=torch.float32)
            V = torch.tensor(V, dtype=torch.float32)
        self.U = nn.Parameter(U)
        self.V = nn.Parameter(V)

        
    def forward(self):
        return self.U @ self.V.t()


def train_model(M_true, missing_rate, rank, lr=0.001, epochs=10000, momentum=0.9, init='random', repeat=1, lamb=0.1):
    """
    Args:
        M_true: full matrix (m x n) 
        missing rate: 
        rank: Target rank for completion
    """
    
    Ms_hat = []
    masks = []
    for _ in range(1, repeat + 1):
        print(f'# ============ [{_}-th] repeat =========== #')
        mask = get_missing_data(M_true, missing_rate)
        M_obs = M_true * mask
        masks.append(mask)
        model = MatrixCompletion(M_obs, rank, init=init)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        # optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        
        M_obs_t = torch.tensor(M_obs, dtype=torch.float32)
        mask_t = torch.tensor(mask, dtype=torch.float32)
        for epoch in range(epochs):
            optimizer.zero_grad()
            M_pred = model()
            loss = criterion(M_pred * mask_t, M_obs_t) + lamb * (torch.norm(model.U) + torch.norm(model.V))
            loss.backward()
            optimizer.step()
            
            # if epoch % 100 == 0:
            #     print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        Ms_hat.append(model().detach().numpy())
    
    return Ms_hat, masks



def get_missing_data(data, missing_rate, mechanism='mcar'):

    while True:
        if mechanism == 'mcar':
            mask = 1 * (np.random.rand(*data.shape) > missing_rate) # True for missing values, false for others
        
        true_missing_rate = 1 - (np.count_nonzero(mask) / mask.size)
        if abs(missing_rate - true_missing_rate) < 0.01:
            break
    print(f'True missing rate: [{true_missing_rate:.4f}]')
    return 1 * mask


################################################################################
# Command line arguments
################################################################################
@click.command()
@click.option('--missing_rate', type=float, default=0.5)
@click.option('--rank', type=int, default=1)
@click.option('--lr', type=float, default=0.001)
@click.option('--epochs', type=int, default=5000)
@click.option('--init', type=str, default='random')
@click.option('--repeat', type=int, default=1)
@click.option('--lamb', type=float, default=0.1)

def main(missing_rate, rank, lr, epochs, init, repeat, lamb):

    all_p_matrices = [
        auroc_20newsgroups, auroc_imdb, auroc_enron, auroc_reuters21578,
        auroc_dbpedia_0, auroc_dbpedia_1, auroc_dbpedia_2, auroc_dbpedia_3, auroc_dbpedia_4,
        auroc_sms_spam, auroc_sst2, auroc_wos
    ]
    all_names = [
        '20newsgroups', 'imdb', 'enron', 'reuters21578',
        'dbpedia_0', 'dbpedia_1', 'dbpedia_2', 'dbpedia_3', 'dbpedia_4',
        'sms_spam', 'sst2', 'wos'
    ]

    print(f'missing rate: [{missing_rate}]')
    print(f'rank: [{rank}]')
    print(f'init: [{init}]')
    print(f'learning rate: [{lr}]')
    print(f'epoch: [{epochs}]')
    print(f'lambda: [{lamb}]')
    print(f'repeat: [{repeat}]')


    save_path = osp.join('./results', generate_filename('.json', *[str(rank), str(missing_rate), init, str(epochs), str(repeat), f'lambda-{str(lamb)}'], timestamp=True))
    for p_matrix, name in zip(all_p_matrices, all_names):

        print(f'low-rank matrix completion for [{name}] ========================')
        M_true = np.array(p_matrix)


        Ms_hat, masks = train_model(M_true, missing_rate=missing_rate, rank=rank, lr=lr, epochs=epochs, init=init, repeat=repeat, lamb=lamb)
        # evaluation =============================================================
        norm_errors = []
        mapes = []
        for M_hat, mask in zip(Ms_hat, masks):
            norm_errors.append(np.linalg.norm((M_true - M_hat) * (1 - mask)) / np.linalg.norm(M_true * (1 - mask)))
            mapes.append(mape(M_true * (1 - mask), M_hat * (1 - mask)))
        print(f'M_true: {M_true}')
        print(f'mask: {masks}')
        print(f'ms_hat: {Ms_hat}')
        print(f"Norm Error: {np.mean(norm_errors):.4f}")
        print(f"MAPE Error: {np.mean(mapes):.4f}")

        json_dump(save_path, {name: '====================================='})
        json_dump(save_path, {
                            'norm_mean': np.mean(norm_errors),
                            'mape_mean': np.mean(mapes),
                            'norm_std': np.std(norm_errors),
                            'mape_std': np.std(mapes)
        })

    print(f"Results saved to {save_path}")

if __name__ == "__main__":


    main()