import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt

def load_data(npz_file):
    data = np.load(npz_file, allow_pickle=True)
    X = data['fingerprints']
    y = data['Etot']
    return X, y

def plot_comparison(y_test, y_pred, mse, mae, r2, title='Prediction vs Actual Etot', save_path='path/comparison.png'):
    plt.figure()
    plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='k', label='Data Points')
    plt.plot([min(y_test)-5, max(y_test)+5], [min(y_test)-5, max(y_test)+5], 'r--', lw=2, label='Ideal Fit')
    plt.xlabel('Actual E$_{tot}$ (Ha)', fontsize=12)
    plt.ylabel('Predicted E$_{tot}$ (Ha)', fontsize=12)
    plt.xlim(min(y_test)-5, max(y_test)+5)
    plt.ylim(min(y_test)-5, max(y_test)+5)
    ti = np.arange(-80, 0, 10)
    plt.xticks(ti, fontsize=10)
    plt.yticks(ti, fontsize=10)
    plt.title(title, fontsize=12)
    ax = plt.gca()
    ax.set_aspect(1)
    plt.legend()
    plt.tight_layout()
    plt.figtext(0.22, 0.75, f'MSE: {mse:.4f}', fontsize=12, ha='left')
    plt.figtext(0.22, 0.7, f'MAE: {mae:.4f}', fontsize=12, ha='left')
    plt.figtext(0.22, 0.65, f'R$^2$: {r2:.4f}', fontsize=12, ha='left')
    plt.savefig(save_path, dpi=300)
    plt.close()
    
def main(npz_file, model_file):
    X, y = load_data(npz_file)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBRegressor()
    model.load_model(model_file)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    y_pred0 = model.predict(X_train)
    mse0 = mean_squared_error(y_train, y_pred0)
    mae0 = mean_absolute_error(y_train, y_pred0)
    r20 = r2_score(y_train, y_pred0)
    
    plot_comparison(y_test, y_pred, mse, mae, r2, title='Prediction vs Actual E$_{tot}$ (XGBoost with Hashed Descriptors)', save_path=f'path/XGBoost_comparison_Descriptors_hashed.png')
    plot_comparison(y_train, y_pred0, mse0, mae0, r20, title='Prediction vs Actual E$_{tot}$ (XGBoost with Hashed Descriptors)', save_path=f'path/XGBoost_comparison_Descriptors_hashed_train.png')
    
if __name__ == "__main__":
    npz_file = 'VQM24/DMC_descriptors_hashed_fingerprints.npz'
    # npz_file = 'VQM24/DMC_descriptors_fingerprints.npz'
    model_file = 'path/best_xgboost_model_descriptors_hashed_fingerprints.model'
    # model_file = 'path/best_xgboost_model_descriptors_fingerprints.model'
    
    main(npz_file, model_file)
