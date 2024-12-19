import pandas as pd
import numpy as np

# 加载smiles和Etot数据
data = np.load('path/DFT_uniques.npz', allow_pickle=True)
smiles_list = data['graphs']
Etot = data['Etot']
Exc = data['Exc']
Eee = data['Eee']
Cp = data['Cp']
Eatom = data['Eatomization']
atoms_list = data['atoms']

# 加载生成的指纹
fingerprints_df = pd.read_csv('path/DFT_pubchem_fingerprints.csv')

# 提取指纹数据
fingerprints = fingerprints_df.drop(columns=['Name']).values

total_atoms = [sum(atoms) for atoms in atoms_list]
average_Etot = Etot / total_atoms

np.savez('path/DFT_pubchem_fingerprints_eval_etot.npz', fingerprints=fingerprints, smiles=smiles_list, Etot=Etot, average_Etot=average_Etot, total_atoms=total_atoms)
'''
# 保存到新的npz文件中
np.savez('path/DFT_pubchem_fingerprints.npz', fingerprints=fingerprints, smiles=smiles_list, Etot=Etot, 
         Exc=Exc, Eee=Eee, Cp=Cp, Eatom=Eatom)
'''