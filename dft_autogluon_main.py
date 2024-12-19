import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from autogluon.tabular import TabularPredictor

def load_data(npz_file, target='Etot'):
    data = np.load(npz_file, allow_pickle=True)
    X = data['fingerprints']
    y = data[target]
    return X, y

def plot_comparison(y_test, y_pred, mse, mae, r2, target='Etot', title='Prediction vs Actual Etot', save_path='path/comparison.png'):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='k', label='Data Points')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', lw=2, label='Ideal Fit')
    if target in ['Etot', 'Exc', 'Eee', 'Eatom']:
        plt.xlabel(f'Actual {target} (Ha)', fontsize=14)
        plt.ylabel(f'Predicted {target} (Ha)', fontsize=14)
    else:
        plt.xlabel('Actual Cp (cal/(mol*K))', fontsize=14)
        plt.ylabel('Predicted Cp (cal/(mol*K))', fontsize=14)
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
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # 使用qcut将目标变量分层
    y_bins = pd.qcut(y, q=10, duplicates='drop')

    stratified_split = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    
    for train_index, test_index in stratified_split.split(X, y_bins):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
    train_data = pd.DataFrame(X_train)
    train_data[target_column] = y_train

    return train_data, X_test, y_test

def main(npz_file, target='Etot'):
    X, y = load_data(npz_file, target=target)
    df = pd.DataFrame(X)
    df[target] = y

    finger_name = npz_file.replace('VQM24/DFT_', '').replace('_fingerprints.npz', '')
    
    # 将数据分割为训练集和测试集
    if target not in ['Cp', 'Eatom']:
        train_data, X_test, y_test = stratified_split(df, target_column=target)
    else:    
        train_data = df.sample(frac=0.8, random_state=42)
        test_data = df.drop(train_data.index)
        y_test = test_data[target]
        X_test = test_data.drop(columns=[target])

    # 使用AutoGluon进行模型训练
    predictor = TabularPredictor(label=target, eval_metric='mean_squared_error').fit(train_data)

    # 预测和评估模型
    y_pred = predictor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Best Model Results: MSE={mse}, MAE={mae}, R2={r2}")

    # 保存预测值与真实值对比图
    plot_comparison(y_test, y_pred, mse, mae, r2, target=target, title=f'Prediction vs Actual {target} (AutoGluon with {finger_name})', save_path=f'path/{finger_name}_{target}_autogluon_comparison.png')
    
if __name__ == "__main__":
    npz_files = ['VQM24/DFT_rdkit_fingerprints.npz', 'VQM24/DFT_pubchem_fingerprints.npz', 'VQM24/DFT_atompair_fingerprints.npz', 
                 'VQM24/DFT_torsion_fingerprints.npz', 'VQM24/DFT_pattern_fingerprints.npz', 'VQM24/DFT_layered_fingerprints.npz',
                 'VQM24/DFT_maccs_fingerprints.npz', 'VQM24/DFT_ecfp_fingerprints.npz', 'VQM24/DFT_fcfp_fingerprints.npz']
    targets = ['Etot', 'Exc', 'Eee', 'Cp', 'Eatom']
    
    for npz_file in npz_files:
        for target in targets:
            main(npz_file, target=target)
