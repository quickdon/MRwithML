import numpy as np

def hash_combine(seed, v):
    """模拟Boost库中的hash_combine函数"""
    return seed ^ (hash(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2))

def generate_hashed_fingerprint(fingerprints, fp_size=2048):
    """生成指纹的哈希指纹"""
    hashed_fingerprints = np.zeros((len(fingerprints), fp_size), dtype=int)
    
    for i, fingerprint in enumerate(fingerprints):
        feature_list = [j for j in range(len(fingerprint)) if fingerprint[j] == 1]
        for feature in feature_list:
            hash_val = hash_combine(0, feature)
            index = hash_val % fp_size
            hashed_fingerprints[i, index] = 1
    
    return hashed_fingerprints

def process_npz_file(input_file, output_file, fp_size=2048):
    """处理npz文件并生成新的哈希指纹npz文件"""
    data = np.load(input_file)
    fingerprints = data['fingerprints']
    Etot = data['Etot']

    hashed_fingerprints = generate_hashed_fingerprint(fingerprints, fp_size=fp_size)

    np.savez(output_file, fingerprints=hashed_fingerprints, Etot=Etot)

# 文件处理
process_npz_file('VQM24/DMC_maccs_fingerprints.npz', 'VQM24/DMC_maccs_hashed_fingerprints_2048.npz', fp_size=2048)
process_npz_file('VQM24/DMC_pubchem_fingerprints.npz', 'VQM24/DMC_pubchem_hashed_fingerprints_2048.npz', fp_size=2048)
