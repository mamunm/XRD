import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from typing import Union
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

class XRDDataset(Dataset):
    """custom class for XRD dataset"""
    def __init__(self, 
                 data: pd.DataFrame):
        """
        Args:
            data (pd.DataFrame): dataframe containing the data
        """
        self.df = data
        self.data_path = Path(__file__).parents[2] / "dataset" / "processed_data"
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        didx = self.df[self.df['index'] == idx]['idx'].values[0]
        label = self.df[self.df['index'] == idx]['label'].values[0]
        data = np.load(self.data_path / "processed" / f'Bi2Te3-{didx:08d}.npy')
        ttheta = data[:, 0]
        data = data[:, 1]
        return data, label, ttheta
    
def get_data_loader(data_ratio: list = [0.75, 0.15, 0.10], 
                    batch_size: int = 32, shuffle: bool = True, 
                    num_workers: int = 0, objective: str = 'binary',
                    include_mid_point: bool = True, 
                    save_path: Union[str, Path] = None,
                    random_state: int = 42) -> DataLoader:
    """
    Args:
        data (string): train or test or valid
        properties (string): property to be used for training
        batch_size (int): batch size
        shuffle (bool): whether to shuffle the dataset
        num_workers (int): number of workers
        objective (string): binary or ternery or multiclass
        include_mid_point (bool): whether to include mid point
        save_path (string): path to save the dataset information
        random_state (int): random state for train_test_split
    Returns:
        data_loader (torch.utils.data.DataLoader): data loader
    """
    df = pd.read_csv(Path(__file__).parents[2] / 'dataset' / 'processed_data' / 'df_processed.csv')
    if objective == 'binary':
        categories = {k: 0 if k <= 1.05 else 1 for k in df['PO'].unique()}
    elif objective == 'ternary':
        categories = {k: 0 if k <= 0.74 else 1 if k <= 1.37 else 2 
                      for k in df['PO'].unique()}
    else:
        categories = {k: 0 if k <= 0.5 else 1 if k <= 0.9 else 2 
                      if k <= 1.1 else 3 if k <= 1.5 else 4 
                      for k in df['PO'].unique()}
    df['label'] = df['PO'].map(categories)
    if not include_mid_point:
        df = df.loc[df.PO != 1.05, :]
    df_train, df_test = train_test_split(df, test_size=data_ratio[1], 
        stratify=df['label'], random_state=random_state)
    df_train, df_valid = train_test_split(df_train, 
        test_size=data_ratio[2]/(1-data_ratio[1]), stratify=df_train['label'],
        random_state=random_state)
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    df_valid.reset_index(drop=True, inplace=True)
    df_train['index'] = df_train.index
    df_test['index'] = df_test.index
    df_valid['index'] = df_valid.index
    train_dataset = XRDDataset(data=df_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, 
                              shuffle=shuffle, num_workers=num_workers)
    valid_dataset = XRDDataset(data=df_valid)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, 
                              shuffle=shuffle, num_workers=num_workers)
    test_dataset = XRDDataset(data=df_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, 
                              shuffle=shuffle, num_workers=num_workers)
    if save_path:
        df_train.to_csv(save_path / 'train.csv', index=False)
        df_test.to_csv(save_path / 'test.csv', index=False)
        df_valid.to_csv(save_path / 'valid.csv', index=False)
    return train_loader, valid_loader, test_loader

    
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
    train, valid, test = get_data_loader(batch_size=512, shuffle=True)
    
    for a, b, c in train:
        print(a.shape, b.shape, c.shape)
        break
    
    
