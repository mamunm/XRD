import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from typing import Union
from pathlib import Path
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

class XRDDataset(Dataset):
    """custom class for XRD dataset"""
    def __init__(self, 
                 data: str = 'train',
                 objective: str = 'binary'):
        """
        Args:
            data (string): train or test or valid
            objective (string): binary or ternary or multiclass
        """
        data_path: Path = Path(__file__).parents[2] / 'dataset' 
        self.data_path = data_path / data 
        self.df = pd.read_csv(data_path / f'df_{data}.csv')
        self.df.drop('Unnamed: 0', axis=1, inplace=True)
        self.labels = self.df[['PO', 'index']]
        if objective == 'binary':
            self.categories = {0.1: 0, 0.575: 0, 1.050: 0, 1.525: 1, 2.000: 1}
        elif objective == 'ternary':
            self.categories = {0.1: 0, 0.575: 0, 1.050: 1, 1.525: 2, 2.000: 2}
        else:
            self.categories = {0.1: 0, 0.575: 1, 1.050: 2, 1.525: 3, 2.000: 4}
        self.labels['label'] = self.labels['PO'].map(self.categories)
        self.df.drop(columns=['PO'], inplace=True)

        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        didx = self.df[self.df['index'] == idx]['idx'].values[0]
        label = self.labels[self.labels['index'] == idx]['label'].values[0]
        data = np.load(self.data_path / "processed" / f'Bi2Te3-{didx:08d}.npy')
        ttheta = data[:, 0]
        data = data[:, 1]
        
        return data, label, ttheta
    
def get_data_loader(data: str, batch_size: int = 32, 
                    shuffle: bool = True, num_workers: int = 0,
                    objective: str = 'binary') -> DataLoader:
    """
    Args:
        data (string): train or test or valid
        properties (string): property to be used for training
        batch_size (int): batch size
        shuffle (bool): whether to shuffle the dataset
        num_workers (int): number of workers
        objective (string): binary or ternery or multiclass
    Returns:
        data_loader (torch.utils.data.DataLoader): data loader
    """
    dataset = XRDDataset(data=data, objective=objective)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, 
                             shuffle=shuffle, num_workers=num_workers,
                             drop_last=True)
    return data_loader
    
if __name__ == "__main__":
    # xrd = XRDDataset(data='train', 
    #                  normalize=True, 
    #                  remove_background=False, 
    #                  zero_padding=True)
    # temp = xrd[2]
    # print(len(temp[0]))
    # import seaborn as sns
    # from matplotlib import pyplot as plt
    # sns.lineplot(temp[2], temp[0])
    # plt.show()
    train = get_data_loader(data='train', batch_size=32, shuffle=True)
    print(train.__len__())
    
    
