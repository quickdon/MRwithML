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

def plot_comparison(y_test, y_pred, mse, mae, r2, title='Prediction vs Actual Etot', save_path='path/comparison.png'):
    plt.figure()
    plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='k', label='Data Points')
    plt.plot([min(y_test)-4, max(y_test)+4], [min(y_test)-4, max(y_test)+4], 'r--', lw=2, label='Ideal Fit')
    plt.xlabel('Actual C$_{p}$ (cal/(mol*K)))', fontsize=12)
    plt.ylabel('Predicted C$_{p}$ (cal/(mol*K))', fontsize=12)
    plt.xlim(min(y_test)-4, max(y_test)+4)
    plt.ylim(min(y_test)-4, max(y_test)+4)
    ti = np.arange(10, 70, 10)
    plt.xticks(ti, fontsize=10)
    plt.yticks(ti, fontsize=10)
    plt.title(title, fontsize=12)
    ax = plt.gca()
    ax.set_aspect(1)
    plt.legend()
    plt.tight_layout()
    plt.figtext(0.23, 0.75, f'MSE: {mse:.4f}', fontsize=12, ha='left')
    plt.figtext(0.23, 0.7, f'MAE: {mae:.4f}', fontsize=12, ha='left')
    plt.figtext(0.23, 0.65, f'R$^2$: {r2:.4f}', fontsize=12, ha='left')
    plt.savefig(save_path, dpi=300)
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

    return train_data, X_test, y_test, X_train, y_train

def main(npz_file, model_file , target='Etot'):
    X, y = load_data(npz_file, target=target)
    df = pd.DataFrame(X)
    df[target] = y
    
    # 将数据分割为训练集和测试集
    if target not in ['Cp', 'Eatom']:
        train_data, X_test, y_test, X_train, y_train = stratified_split(df, target_column=target)
    else:    
        train_data = df.sample(frac=0.8, random_state=42)
        test_data = df.drop(train_data.index)
        y_test = test_data[target]
        X_test = test_data.drop(columns=[target])
        y_train = train_data[target]
        X_train = train_data.drop(columns=[target])

    # 使用AutoGluon进行模型训练
    predictor = TabularPredictor(label=target).load(model_file)

    # 预测和评估模型
    y_pred = predictor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    y_pred0 = predictor.predict(X_train)
    mse0 = mean_squared_error(y_train, y_pred0)
    mae0 = mean_absolute_error(y_train, y_pred0)
    r20 = r2_score(y_train, y_pred0)

    # 保存预测值与真实值对比图
    plot_comparison(y_test, y_pred, mse, mae, r2, title='Prediction vs Actual C$_{p}$ (AutoGluon with Descriptors)', save_path='path/Descriptors_Cp_autogluon_comparison.png')
    plot_comparison(y_train, y_pred0, mse0, mae0, r20, title='Prediction vs Actual C$_{p}$ (AutoGluon with Descriptors)', save_path='path/Descriptors_Cp_autogluon_comparison_train.png')
    
if __name__ == "__main__":
    npz_file = 'VQM24/DFT_descriptors_fingerprints.npz'
    model_file = 'AutogluonModels/Descriptors_Cp'
    targets = ['Etot', 'Exc', 'Eee', 'Cp', 'Eatom']
    target = 'Cp'

    main(npz_file, model_file, target=target)