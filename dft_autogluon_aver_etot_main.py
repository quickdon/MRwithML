import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from autogluon.tabular import TabularPredictor

def load_data(npz_file, target='average_Etot'):
    data = np.load(npz_file, allow_pickle=True)
    X = data['fingerprints']
    y = data[target]
    total_atoms = data['total_atoms']
    return X, y, total_atoms

def plot_comparison(y_test, y_pred, mse, mae, r2, target='Etot', title='Prediction vs Actual', save_path='path/comparison.png'):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='k', label='Data Points')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', lw=2, label='Ideal Fit')
    plt.xlabel(f'Actual {target} (Ha)', fontsize=14)
    plt.ylabel(f'Predicted {target} (Ha)', fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.figtext(0.15, 0.8, f'MSE: {mse:.4f}', fontsize=12, ha='left')
    plt.figtext(0.15, 0.75, f'MAE: {mae:.4f}', fontsize=12, ha='left')
    plt.figtext(0.15, 0.7, f'R^2: {r2:.4f}', fontsize=12, ha='left')
    plt.savefig(save_path)
    plt.close()

def stratified_split(data, target_column, test_size=0.2, n_splits=5, random_state=42):
    X = data.drop(columns=[target_column, 'total_atoms'])
    y = data[target_column]
    total_atoms = data['total_atoms']

    # 使用qcut将目标变量分层
    y_bins = pd.qcut(y, q=10, duplicates='drop')

    stratified_split = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    
    for train_index, test_index in stratified_split.split(X, y_bins):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        total_atoms_test = total_atoms[test_index]
        
    train_data = pd.DataFrame(X_train)
    train_data[target_column] = y_train

    return train_data, X_test, y_test, total_atoms_test

def main(npz_file):
    X, y, total_atoms = load_data(npz_file, target='average_Etot')
    df = pd.DataFrame(X)
    df['average_Etot'] = y
    df['total_atoms'] = total_atoms

    finger_name = npz_file.replace('VQM24/DFT_', '').replace('_fingerprints_v2.npz', '')
    
    # 将数据分割为训练集和测试集
    train_data, X_test, y_test, total_atoms_test = stratified_split(df, target_column='average_Etot')

    # 使用AutoGluon进行模型训练
    predictor = TabularPredictor(label='average_Etot', eval_metric='mean_squared_error').fit(train_data)

    # 预测和评估模型
    y_pred = predictor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Average Energy - Best Model Results: MSE={mse}, MAE={mae}, R2={r2}")

    # 保存平均能量预测值与真实值对比图
    plot_comparison(y_test, y_pred, mse, mae, r2, target='average_Etot', title=f'Prediction vs Actual Average Etot (AutoGluon with {finger_name})', save_path=f'path/{finger_name}_average_Etot_autogluon_comparison.png')
    
    # 转换回总能量进行评价
    total_Etot_pred = y_pred * total_atoms_test
    total_Etot_test = y_test * total_atoms_test

    mse_total = mean_squared_error(total_Etot_test, total_Etot_pred)
    mae_total = mean_absolute_error(total_Etot_test, total_Etot_pred)
    r2_total = r2_score(total_Etot_test, total_Etot_pred)

    print(f"Total Energy - Best Model Results: MSE={mse_total}, MAE={mae_total}, R2={r2_total}")

    # 保存总能量预测值与真实值对比图
    plot_comparison(total_Etot_test, total_Etot_pred, mse_total, mae_total, r2_total, target='Etot', title=f'Prediction vs Actual Etot (AutoGluon with {finger_name})', save_path=f'path/{finger_name}_Etot_autogluon_comparison.png')
    
if __name__ == "__main__":
    npz_files = ['VQM24/DFT_rdkit_fingerprints_aver_etot.npz', 'VQM24/DFT_pubchem_fingerprints_aver_etot.npz', 'VQM24/DFT_atompair_fingerprints_aver_etot.npz', 
                 'VQM24/DFT_torsion_fingerprints_aver_etot.npz', 'VQM24/DFT_pattern_fingerprints_aver_etot.npz', 'VQM24/DFT_layered_fingerprints_aver_etot.npz',
                 'VQM24/DFT_maccs_fingerprints_aver_etot.npz', 'VQM24/DFT_ecfp_fingerprints_aver_etot.npz', 'VQM24/DFT_fcfp_fingerprints_aver_etot.npz']
    
    for npz_file in npz_files:
        main(npz_file)
