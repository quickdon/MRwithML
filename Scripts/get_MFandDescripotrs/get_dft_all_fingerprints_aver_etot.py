import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem

# 加载数据
data = np.load('VQM24/original/DFT_uniques.npz', allow_pickle=True)
smiles_list = data['graphs']
Etot = data['Etot']
atoms_list = data['atoms']  # 原子序数列表

# 函数：计算MACCS指纹
def get_maccs_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fp = MACCSkeys.GenMACCSKeys(mol)
        return list(map(int, fp.ToBitString()))
    return None

# 函数：计算ECFP指纹
def get_ecfp_fingerprint(smiles, radius=3, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fp = AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits).GetFingerprint(mol)
        return list(map(int, fp.ToBitString()))
    return None

# 函数：计算FCFP指纹
def get_fcfp_fingerprint(smiles, radius=3, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        invgen = AllChem.GetMorganFeatureAtomInvGen()
        fp = AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits, atomInvariantsGenerator=invgen).GetFingerprint(mol)
        return list(map(int, fp.ToBitString()))
    return None

# 函数：计算RDKit指纹
def get_rdkit_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fp = AllChem.GetRDKitFPGenerator().GetFingerprint(mol)
        return list(map(int, fp.ToBitString()))
    return None

# 函数：计算AtomPair指纹
def get_atompair_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fp = AllChem.GetAtomPairGenerator(countSimulation=False).GetFingerprint(mol)
        return list(map(int, fp.ToBitString()))
    return None

# 函数：计算TopologicalTorsion指纹
def get_torsion_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fp = AllChem.GetTopologicalTorsionGenerator(countSimulation=False).GetFingerprint(mol)
        return list(map(int, fp.ToBitString()))
    return None

# 函数：计算Pattern指纹
def get_pattern_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fp = Chem.PatternFingerprint(mol)
        return list(map(int, fp.ToBitString()))
    return None

# 函数：计算Layered指纹
def get_layered_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fp = Chem.LayeredFingerprint(mol)
        return list(map(int, fp.ToBitString()))
    return None

# 计算指纹
maccs_fingerprints = [get_maccs_fingerprint(smiles) for smiles in smiles_list]
ecfp_fingerprints = [get_ecfp_fingerprint(smiles) for smiles in smiles_list]
fcfp_fingerprints = [get_fcfp_fingerprint(smiles) for smiles in smiles_list]
rdkit_fingerprints = [get_rdkit_fingerprint(smiles) for smiles in smiles_list]
atompair_fingerprints = [get_atompair_fingerprint(smiles) for smiles in smiles_list]
torsion_fingerprints = [get_torsion_fingerprint(smiles) for smiles in smiles_list]
pattern_fingerprints = [get_pattern_fingerprint(smiles) for smiles in smiles_list]
layered_fingerprints = [get_layered_fingerprint(smiles) for smiles in smiles_list]

# 过滤掉无法处理的分子
valid_indices = [i for i in range(len(maccs_fingerprints)) if maccs_fingerprints[i] is not None and ecfp_fingerprints[i] is not None 
                 and fcfp_fingerprints[i] is not None and rdkit_fingerprints[i] is not None and atompair_fingerprints[i] is not None 
                 and torsion_fingerprints[i] is not None and pattern_fingerprints[i] is not None and layered_fingerprints[i] is not None]
maccs_fingerprints = [maccs_fingerprints[i] for i in valid_indices]
ecfp_fingerprints = [ecfp_fingerprints[i] for i in valid_indices]
fcfp_fingerprints = [fcfp_fingerprints[i] for i in valid_indices]
rdkit_fingerprints = [rdkit_fingerprints[i] for i in valid_indices]
atompair_fingerprints = [atompair_fingerprints[i] for i in valid_indices]
torsion_fingerprints = [torsion_fingerprints[i] for i in valid_indices]
pattern_fingerprints = [pattern_fingerprints[i] for i in valid_indices]
layered_fingerprints = [layered_fingerprints[i] for i in valid_indices]
smiles_list = [smiles_list[i] for i in valid_indices]
Etot = Etot[valid_indices]
atoms_list = [atoms_list[i] for i in valid_indices]

# 计算总原子数和平均能量
total_atoms = [sum(atoms) for atoms in atoms_list]
average_Etot = Etot / total_atoms

# 转换指纹为numpy数组
maccs_fingerprints = np.array(maccs_fingerprints)
ecfp_fingerprints = np.array(ecfp_fingerprints)
fcfp_fingerprints = np.array(fcfp_fingerprints)
rdkit_fingerprints = np.array(rdkit_fingerprints)
atompair_fingerprints = np.array(atompair_fingerprints)
torsion_fingerprints = np.array(torsion_fingerprints)
pattern_fingerprints = np.array(pattern_fingerprints)
layered_fingerprints = np.array(layered_fingerprints)

# 保存到新的npz文件中
np.savez('VQM24/DFT_maccs_fingerprints_aver_etot.npz', fingerprints=maccs_fingerprints, smiles=smiles_list, Etot=Etot, average_Etot=average_Etot, total_atoms=total_atoms)
np.savez('VQM24/DFT_ecfp_fingerprints_aver_etot.npz', fingerprints=ecfp_fingerprints, smiles=smiles_list, Etot=Etot, average_Etot=average_Etot, total_atoms=total_atoms)
np.savez('VQM24/DFT_fcfp_fingerprints_aver_etot.npz', fingerprints=fcfp_fingerprints, smiles=smiles_list, Etot=Etot, average_Etot=average_Etot, total_atoms=total_atoms)
np.savez('VQM24/DFT_rdkit_fingerprints_aver_etot.npz', fingerprints=rdkit_fingerprints, smiles=smiles_list, Etot=Etot, average_Etot=average_Etot, total_atoms=total_atoms)
np.savez('VQM24/DFT_atompair_fingerprints_aver_etot.npz', fingerprints=atompair_fingerprints, smiles=smiles_list, Etot=Etot, average_Etot=average_Etot, total_atoms=total_atoms)
np.savez('VQM24/DFT_torsion_fingerprints_aver_etot.npz', fingerprints=torsion_fingerprints, smiles=smiles_list, Etot=Etot, average_Etot=average_Etot, total_atoms=total_atoms)
np.savez('VQM24/DFT_pattern_fingerprints_aver_etot.npz', fingerprints=pattern_fingerprints, smiles=smiles_list, Etot=Etot, average_Etot=average_Etot, total_atoms=total_atoms)
np.savez('VQM24/DFT_layered_fingerprints_aver_etot.npz', fingerprints=layered_fingerprints, smiles=smiles_list, Etot=Etot, average_Etot=average_Etot, total_atoms=total_atoms)

print("数据已保存为新的npz文件。")
