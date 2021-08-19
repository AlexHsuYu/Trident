''' try continuous wavelet transform (CWT) '''
import os, sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 16
plt.rcParams['savefig.directory'] = os.path.dirname(__file__)
from scipy import signal
from utils import read_data, Tsunami

from __init__ import BASE_PATH, PROJECT_ROOT

''' NOTE: this file is for experimentations '''

def awgn(x, snr):
    x_power = x ** 2
    x_avg_power = np.mean(x_power)
    x_avg_db = 10 * np.log10(x_avg_power)
    n_avg_db = x_avg_db - snr
    n_avg_power = 10 ** (n_avg_db/10)
    wgn = np.random.normal(0.0, np.sqrt(n_avg_power), len(x))

    return x + wgn

# all in one csv generation func
def _all():
    master_path = os.path.join(BASE_PATH, 'interp1d_data/index.csv')
    master_df = pd.read_csv(master_path)
    full_df = pd.DataFrame()
    # full_df = pd.DataFrame(columns=range(8))
    for i, idx in master_df.iterrows():
        file_path = os.path.join(BASE_PATH, 'interp1d_data', idx['path'])
        df = pd.read_csv(file_path)
        df['label'] = np.full(df.shape[0], idx['category'])
        full_df = full_df.append(df)
        # full_df = pd.concat([full_df, df], axis=0)

    full_df.to_csv(os.path.join(BASE_PATH, 'interp1d_data/all_in_one.csv'), index=False)

if __name__ == "__main__":
    ''' load data '''
    f_index = pd.read_csv(os.path.join(PROJECT_ROOT, 'assets/cnn_test.csv'))

    tg = Tsunami(
        os.path.join(BASE_PATH, 'interp1d_data'),
        f_index,
        batch_size=8
    )
    it = iter(tg)
    x, y = next(it)
    print(y)
    
