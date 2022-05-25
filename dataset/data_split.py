from shutil import copy2
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import sys

df = pd.read_csv(Path(__file__).resolve().parent / 'input_10K' / 'index_dict.csv')
df_train, df_test = train_test_split(df, 
                                     test_size=0.2, 
                                     random_state=42, 
                                     stratify=df['PO'])
df_train, df_valid = train_test_split(df_train, 
                                      test_size=0.125, 
                                      random_state=42,
                                      stratify=df_train['PO'])

df_train.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)
df_valid.reset_index(drop=True, inplace=True)
df_train['index'] = df_train.index
df_test['index'] = df_test.index
df_valid['index'] = df_valid.index
df_train.to_csv('dataset/df_train.csv')
df_test.to_csv('dataset/df_test.csv')
df_valid.to_csv('dataset/df_valid.csv')

for dest, split in zip(['train', 'test', 'valid'], [df_train, df_test, df_valid]):
    for f in split.idx:
        print(f'moving Bi2Te3-{f:08d}.xy to {dest}')
        src = Path(__file__).resolve().parent / 'input_10K' / f"Bi2Te3-{f:08d}.xy"
        dst = Path(__file__).resolve().parent / dest / f"Bi2Te3-{f:08d}.xy"
        copy2(src, dst)