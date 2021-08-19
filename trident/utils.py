import os, sys
import pandas as pd, numpy as np, scipy as sp
from . import CLASS_MAP
from .config import Config

# somewhat like a generator
def dataset(df, directory, mode='test'):
    df = df.reset_index(drop=True)

    X = np.empty((df.shape[0], Config.FEATURE_SIZE))
    y = df['category'].values
    for i, idx in df.iterrows():
        if mode == 'train':
            frame = read_data(os.path.join(directory, idx['path']), start=2)
        elif mode == 'test':
            frame = read_data(os.path.join(directory, idx['path']), start=1)
        data = preprocess_input(frame, f_size=Config.FEATURE_SIZE) ####### breaking change, do not directly execute
        # print(data.shape)
        X[i, ] = data

    return X, y

# read txt files into dataframes
# because of encoding issues and stuff
def read_data(path, start=1):
    # data is not seperated correctly
    df = pd.read_csv(path, sep='\s+\t', engine='python', index_col=None, skiprows=start)
    # interpolate when value is zero (sensor failure)
    df.replace(0.0, np.nan, inplace=True)
    df = df.interpolate(method='linear', axis=0)

    return df

def generate_file_index(path):
    paths = []
    categories = []
    lbl = -1
    for root, _, files in os.walk(path):
        for file in files:
            # print(root.split('/')[-1], file)
            foldername = root.split('/')[-1]
            # the processing
            if foldername == 'G11':
                lbl = 0
            elif foldername == 'G15':
                lbl = 1
            elif foldername == 'G17':
                lbl = 2
            elif foldername == 'G19':
                lbl = 3
            elif foldername == 'G32':
                lbl = 4
            elif foldername == 'G34':
                lbl = 5
            elif foldername == 'G48':
                lbl = 6
            elif foldername == 'G49':
                lbl = 7

            # skip index.csv file to be placed in index.csv file XD
            if 'index' in file:
                continue
            categories.append(lbl)
            if foldername == path.split('/')[-1]:
                paths.append(file)
            else:
                paths.append(os.path.join(foldername, file))
        

    full_df = pd.DataFrame(np.column_stack((paths, categories)), columns=['path', 'category'])
    # print(full_df)
    full_df.to_csv(os.path.join(path, 'index.csv'), index=False)

def preprocess_input(df, f_size=20):
    ''' resample data to same length '''
    interp_func = sp.interpolate.interp1d(df.index, df.values,
                                        kind='linear',
                                        axis=0,
                                        fill_value='extrapolate')
    
    # reset all length to 450
    new_len = np.arange(0, len(df), len(df)/450)
    rescaled = interp_func(new_len)
    # print(rescaled.shape)

    ''' calculate discrete cosine transform '''
    # flatten to (8*450, 1)
    rescaled = np.reshape(rescaled, (1, np.prod(rescaled.shape)))
    # print(rescaled)
    dct_map = sp.fftpack.dct(rescaled, axis=1)
    # print(dct_map.shape)

    return dct_map[0, 1:f_size+1]
