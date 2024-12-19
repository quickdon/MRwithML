import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import LinearSVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D
import pickle
from joblib import Parallel, delayed

def load_data(npz_file, target='Etot'):
    """
    加载npz文件中的指纹和Etot数据
    """
    data = np.load(npz_file, allow_pickle=True)
    X = data['fingerprints']
    y = data[target]
    return X, y

def stratified_split(X, y, test_size=0.2, n_splits=5, random_state=42):
    # 将y离散化，以便进行分层采样
    y_bins = pd.qcut(y, q=5, duplicates='drop')

    strat_split = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    for train_index, test_index in strat_split.split(X, y_bins):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    return X_train, X_test, y_train, y_test

def build_cnn(input_shape):
    model = Sequential([
        Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_model(X, y, target='Etot', model_type='random_forest', test_size=0.2, random_state=42):
    """
    训练回归模型并返回训练好的模型和测试集
    """
    if target not in ['Etot', 'Exc', 'Eee']:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    else:
        X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=test_size, random_state=random_state)
    
    if model_type == 'random_forest':
        model = RandomForestRegressor(n_jobs=-1)
    elif model_type == 'linear_regression':
        model = LinearRegression()
    elif model_type == 'ridge':
        model = Ridge()
    elif model_type == 'linearsvm':
        model = LinearSVR()
    elif model_type == 'xgboost':
        model = XGBRegressor()
    elif model_type == 'adaboost':
        model = AdaBoostRegressor()
    elif model_type == 'cnn':
        model = build_cnn((X_train.shape[1], 1))
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
    else:
        raise ValueError(f"未知的模型类型: {model_type}")
    
    if model_type not in ['cnn']:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    else:
        y_pred = model.predict(X_test).flatten()
    
    return model, X_test, y_test, y_pred

def evaluate_model(y_test, y_pred):
    """
    计算并返回模型的评价指标
    """
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, mae, r2

def plot_comparison(y_test, y_pred, mse, mae, r2, target='Etot', title='Predicted vs Actual Etot', save_path='path/comparison.png'):
    """
    绘制预测值和真实值的比较图并保存
    """
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

def main(npz_file, target='Etot', model_type='random_forest'):
    X, y = load_data(npz_file, target=target)
    model, X_test, y_test, y_pred = train_model(X, y, target=target, model_type=model_type)
    mse, mae, r2 = evaluate_model(y_test, y_pred)
    
    finger_name = npz_file.replace('VQM24/DFT_', '').replace('_fingerprints.npz', '')
    
    print(f"Model: {model_type}")
    print(f"Fingerprint: {finger_name}")
    print(f"Target: {target}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R^2 Score: {r2}")
    
    plot_comparison(y_test, y_pred, mse, mae, r2, target=target, title=f'Predicted vs Actual {target} ({model_type} with {finger_name})', save_path=f'path/{finger_name}_{model_type}_{target}_comparison.png')
    
    # 保存模型
    model_filename = f'path/{finger_name}_{model_type}_{target}_model.pkl' if model_type not in ['cnn'] else f'path/{finger_name}_{model_type}_{target}_model.h5'
    if model_type not in ['cnn']:
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)
    else:
        model.save(model_filename)

# 运行主函数
if __name__ == "__main__":
    # 文件名，可以根据需要更改
    npz_files = ['VQM24/DFT_rdkit_fingerprints.npz', 'VQM24/DFT_pubchem_fingerprints.npz', 'VQM24/DFT_atompair_fingerprints.npz', 
                 'VQM24/DFT_torsion_fingerprints.npz', 'VQM24/DFT_pattern_fingerprints.npz', 'VQM24/DFT_layered_fingerprints.npz',
                 'VQM24/DFT_maccs_fingerprints.npz', 'VQM24/DFT_ecfp_fingerprints.npz', 'VQM24/DFT_fcfp_fingerprints.npz']
    models = ['random_forest', 'linear_regression', 'ridge', 'linearsvm', 'xgboost', 'adaboost', 'cnn']
    targets = ['Etot', 'Exc', 'Eee', 'Cp', 'Eatom']
    
    Parallel(n_jobs=5)(delayed(main)(npz_file, target=target, model_type=model_type)
                    for npz_file in npz_files
                    for model_type in models
                    for target in targets)
    
