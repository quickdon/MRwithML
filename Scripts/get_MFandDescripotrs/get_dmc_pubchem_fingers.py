import pandas as pd
import numpy as np

# 加载smiles和Etot数据
data = np.load('path/DMC.npz', allow_pickle=True)
smiles_list = data['graphs']
Etot = data['Etot']

# 加载生成的指纹
fingerprints_df = pd.read_csv('path/pubchem_fingerprints.csv')

# 提取指纹数据
fingerprints = fingerprints_df.drop(columns=['Name']).values

# 保存到新的npz文件中
np.savez('path/DMC_pubchem_fingerprints.npz', fingerprints=fingerprints, smiles=smiles_list, Etot=Etot)