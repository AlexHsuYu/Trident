import pandas as pd 
import os
from scipy.fftpack import dct
import numpy as np
F_PATH = '/home/lab/Documents/python/Data_contest2019_V2/tayas/datasets/interp1d_data/index.csv'
           
dfs = pd.read_csv(F_PATH)


al= pd.DataFrame({})

for k, col in dfs.iterrows():    
    df = pd.read_csv(os.path.join('/home/lab/Documents/python/Data_contest2019_V2/tayas/datasets/interp1d_data/',str(dfs.loc[k,'path'])))     
    dst = []
    a0 = pd.DataFrame({})
    if df.shape[1] < 8:
        for i in np.arange(8 - df.shape[1]):
            df[ str(df.shape[1] + i) ] = df[str(i)]            
    matrix = df.values
    rows=450
    columns=8
    solution=[[] for i in range(rows+columns-1)]
    for i in range(rows): 
        for j in range(columns): 
            sum=i+j 
            if(sum%2 ==0):            
                solution[sum].insert(0,matrix[i][j]) 
            else:            
                solution[sum].append(matrix[i][j])        
    count = 0 
    for i in solution: 
        for j in i:
            count = count+1 
            dst.append(j)
              
    dst = dct(dst,type=2, n=None, axis=-1, norm=None, overwrite_x=False)
    a0 = pd.DataFrame(dst)
    a0 = a0[1:250].T
    # print(a0)
    a0['label'] = dfs.loc[k, 'category']    
    al = pd.concat([al,a0],axis=0)
print(len(al))
al.to_csv('/home/lab/Documents/python/Data_contest2019_V2/tayas/datasets/interp1d_data/all_in_one_V2.csv',index=False)
  
'''
a columns to dct
'''
# import pandas as pd 
# import os
# from scipy.fftpack import dct
# import numpy as np
# F_PATH = '/home/lab/Documents/python/Data_contest2019_V2/tayas/datasets/interp1d_data/index.csv'
           
# dfs = pd.read_csv(F_PATH)


# al= pd.DataFrame({})
# for i, col in dfs.iterrows():
    
#     a = pd.read_csv(os.path.join('/home/lab/Documents/python/Data_contest2019_V2/tayas/datasets/interp1d_data/',str(dfs.loc[i,'path'])  ))     # print(dfs.shape[1])
    
#     if a.shape[1] < 8:
#         for i in np.arange(8 - a.shape[1]):
#             a[ str(a.shape[1] + i) ] = dfs[str(i)]
            
#     a.columns = range(len(a.columns))    
#     a0 = pd.DataFrame({})           
#     for r,col in enumerate(a.columns):             
#         dct1 = dct(a[r],type=2, n=None, axis=-1, norm=None, overwrite_x=False)
#         dct1 = pd.DataFrame(dct1)
#         a0 = pd.concat([a0,dct1[1:]],axis=1)   
    
#     a = pd.DataFrame(a0)
#     a.columns = range(len(a0.columns))
#     a['label'] = dfs.loc[i, 'category']
    
#     print(a)
    
#     a = pd.DataFrame(a)
    
#     al = pd.concat([al,a0],axis=0)
# print(len(al))
    
# al.to_csv('/home/lab/Documents/python/Data_contest2019_V2/tayas/datasets/interp1d_data/all_in_one_V2.csv')
      