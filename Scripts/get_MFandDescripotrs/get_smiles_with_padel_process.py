import numpy as np

# 加载smiles和Etot数据
# data = np.load('path/DMC.npz', allow_pickle=True)
data = np.load('path/DFT_uniques.npz', allow_pickle=True)
smiles_list = data['graphs']

# 保存smiles到文件
# with open('path/DMC_molecules.smi', 'w') as f:
with open('path/DFT_molecules.smi', 'w') as f:
    for smiles in smiles_list:
        f.write(f"{smiles}\n")
        
# java -jar PaDEL-Descriptor/PaDEL-Descriptor.jar -fingerprints -descriptortypes PaDEL-Descriptor/descriptors.xml -dir VQM24/DFT_molecules.smi -file DFT_fingerprints.csv -retainorder
