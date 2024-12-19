import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Descriptors3D

def reorder_atoms(mol, target_atoms):
    """重新排列分子的原子顺序以匹配目标原子序数"""
    atom_order = []
    for target_atom in target_atoms:
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == target_atom and atom.GetIdx() not in atom_order:
                atom_order.append(atom.GetIdx())
                break
    return atom_order

def create_molecule_from_smiles_and_coordinates(smiles, atoms, coordinates):
    """从SMILES字符串和坐标信息创建包含显式氢原子的RDKit分子对象"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("无法从SMILES字符串生成分子对象")
        
        # 添加显式氢原子
        mol = Chem.AddHs(mol)
        
        # 检查坐标数是否与原子数匹配
        if len(coordinates) != mol.GetNumAtoms():
            raise ValueError("坐标数与原子数不匹配")
        
        # 检查生成的分子原子序数是否与文件中的原子序数匹配
        mol_atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        if mol_atoms != list(atoms):
            atom_order = reorder_atoms(mol, atoms)
            mol = Chem.RenumberAtoms(mol, atom_order)
            mol_atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
            if mol_atoms != list(atoms):
                raise ValueError("SMILES生成的分子原子序数与文件中的原子序数不匹配，即使重新排列")
        
        # 添加坐标
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i, coord in enumerate(coordinates):
            conf.SetAtomPosition(i, coord)
        
        mol.AddConformer(conf)
        return mol
    except Exception as e:
        print(f"Error processing molecule with SMILES '{smiles}': {e}")
        return None

def calculate_descriptors(mol):
    """计算所有描述符"""
    descriptors_2d = Descriptors.CalcMolDescriptors(mol)
    descriptors_3d = Descriptors3D.CalcMolDescriptors3D(mol)
    return {**descriptors_2d, **descriptors_3d}
    # return descriptors_2d
    # return descriptors_3d

def hash_combine(seed, v):
    """模拟Boost库中的hash_combine函数"""
    return seed ^ (hash(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2))

def generate_fingerprint(descriptors, fp_size=2048):
    """生成描述符的指纹"""
    hashed_fingerprint = np.zeros(fp_size, dtype=int)
    
    for layer, (key, value) in enumerate(descriptors.items()):
        seed = layer
        seed = hash_combine(seed, value)
        
        index = seed % fp_size
        hashed_fingerprint[index] = 1
    
    return hashed_fingerprint

# 从npz文件中读取数据
data = np.load('path/DMC.npz', allow_pickle=True)
atoms_list = data['atoms']
coordinates_list = data['coordinates']
smiles_list = data['graphs']
Etot_list = data['Etot']

# 存储生成的指纹和对应的Etot值
fingerprints = []
Etot_output = []

for i in range(len(smiles_list)):
    atoms = atoms_list[i]
    coordinates = coordinates_list[i]
    smiles = smiles_list[i]
    Etot = Etot_list[i]

    mol = create_molecule_from_smiles_and_coordinates(smiles, atoms, coordinates)
    if mol is not None:
        # 计算描述符
        descriptors = calculate_descriptors(mol)
        
        print(descriptors)
       
        # 生成指纹
        fingerprint = generate_fingerprint(descriptors, fp_size=2048)
        fingerprints.append(fingerprint)
        Etot_output.append(Etot)

# 将指纹转换为numpy数组
fingerprints_array = np.array(fingerprints)
Etot_array = np.array(Etot_output)

# 保存指纹和Etot到新的npz文件
np.savez('VQM24/DMC_descriptors_hashed_fingerprints.npz', fingerprints=fingerprints_array, Etot=Etot_array)
# np.savez('VQM24/DMC_descriptors_hashed_2d_fingerprints.npz', fingerprints=fingerprints_array, Etot=Etot_array)
# np.savez('VQM24/DMC_descriptors_hashed_3d_fingerprints.npz', fingerprints=fingerprints_array, Etot=Etot_array)
