import numpy as np
import hashlib

def hash_feature(feature):
    """生成特征的哈希值，并映射到指定长度的指纹中"""
    hash_obj = hashlib.sha256(feature.encode('utf-8')).digest()
    bit_array = np.unpackbits(np.frombuffer(hash_obj, dtype=np.uint8))
    
    # 将256位的哈希值扩展或分配到指定长度
    # extended_bit_array = np.tile(bit_array, (length // len(bit_array)) + 1)[:length]
    
    return bit_array

def generate_hashed_fingerprint(fingerprints):
    """生成256位哈希指纹"""
    hashed_fingerprints = np.zeros((fingerprints.shape[0], 256), dtype=int)
    for i, fingerprint in enumerate(fingerprints):
        feature_list = [str(j) for j in range(len(fingerprint)) if fingerprint[j] == 1]
        for feature in feature_list:
            hashed_bits = hash_feature(feature)
            hashed_fingerprints[i] = np.bitwise_or(hashed_fingerprints[i], hashed_bits)
    return hashed_fingerprints

def process_npz_file(input_file, output_file):
    """处理npz文件并生成新的哈希指纹npz文件"""
    data = np.load(input_file)
    fingerprints = data['fingerprints']
    Etot = data['Etot']

    hashed_fingerprints = generate_hashed_fingerprint(fingerprints)

    np.savez(output_file, fingerprints=hashed_fingerprints, Etot=Etot)

# 文件处理

process_npz_file('VQM24/DMC_maccs_fingerprints.npz', 'VQM24/DMC_maccs_hashed_fingerprints_256.npz')

process_npz_file('VQM24/DMC_pubchem_fingerprints.npz', 'VQM24/DMC_pubchem_hashed_fingerprints_256.npz')

# 如果需要生成2048位的指纹，只需修改length参数
# process_npz_file('VQM24/DMC_maccs_fingerprints.npz', 'VQM24/DMC_maccs_hashed_fingerprints_2048_Sha.npz', length=2048)
# process_npz_file('VQM24/DMC_pubchem_fingerprints.npz', 'VQM24/DMC_pubchem_hashed_fingerprints_2048_sha.npz', length=2048)
