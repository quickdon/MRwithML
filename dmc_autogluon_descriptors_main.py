import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from autogluon.tabular import TabularPredictor

def load_data(npz_file):
    data = np.load(npz_file, allow_pickle=True)
    X = data['fingerprints']
    y = data['Etot']
    return X, y

def plot_comparison(y_test, y_pred, mse, mae, r2, title='Prediction vs Actual Etot', save_path='path/comparison.png'):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='k', label='Data Points')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', lw=2, label='Ideal Fit')
    plt.xlabel('Actual Etot (Ha)', fontsize=14)
    plt.ylabel('Predicted Etot (Ha)', fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.figtext(0.15, 0.8, f'MSE: {mse:.4f}', fontsize=12, ha='left')
    plt.figtext(0.15, 0.75, f'MAE: {mae:.4f}', fontsize=12, ha='left')
    plt.figtext(0.15, 0.7, f'R^2: {r2:.4f}', fontsize=12, ha='left')
    plt.savefig(save_path)
    plt.close()

def main(npz_file):
    X, y = load_data(npz_file)
    df = pd.DataFrame(X)
    df['Etot'] = y

    finger_name = npz_file.replace('VQM24/DMC_', '').replace('_fingerprints', '').replace('.npz', '')
    
    # 将数据分割为训练集和测试集
    train_data = df.sample(frac=0.8, random_state=42)
    test_data = df.drop(train_data.index)
    y_test = test_data['Etot']
    X_test = test_data.drop(columns=['Etot'])

    # 使用AutoGluon进行模型训练
    predictor = TabularPredictor(label='Etot', eval_metric='mean_squared_error').fit(train_data)

    # 预测和评估模型
    y_pred = predictor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Best Model Results: MSE={mse}, MAE={mae}, R2={r2}")

    # 保存预测值与真实值对比图
    plot_comparison(y_test, y_pred, mse, mae, r2, title=f'Prediction vs Actual Etot (AutoGluon with {finger_name})', save_path=f'path/{finger_name}_autogluon_comparison.png')

if __name__ == "__main__":
    npz_files = ['VQM24/DMC_descriptors_hashed_fingerprints.npz', 'VQM24/DMC_descriptors_hashed_2d_fingerprints.npz', 'VQM24/DMC_descriptors_fingerprints.npz', 'VQM24/DMC_descriptors_2d_fingerprints.npz']
    
    for npz_file in npz_files:
        main(npz_file)
