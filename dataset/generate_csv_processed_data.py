from numpy.ma.core import default_fill_value
import pandas as pd 
from pathlib import Path

df_train = pd.read_csv(Path(__file__).resolve().parent / 'df_train.csv')
df_test = pd.read_csv(Path(__file__).resolve().parent / 'df_test.csv')
df_valid = pd.read_csv(Path(__file__).resolve().parent / 'df_valid.csv')
df_20K = pd.read_csv(Path(__file__).resolve().parent / 'input_20K' / 'index_dict.csv')

for df in [df_train, df_test, df_valid]:
    df.drop(['Unnamed: 0'], axis=1, inplace=True)
    df.drop(['index'], axis=1, inplace=True)

idx_omit = [255334156, 886397269]
df_20K = df_20K.loc[df_20K['idx'].isin(idx_omit) == False, :]
df_20K['DEL'] = 0.02

df = pd.concat([df_train, df_test, df_valid, df_20K])

df.reset_index(drop=True, inplace=True)

df.to_csv(Path(__file__).resolve().parent / 'processed_data' / 'df_processed.csv', index=False)
