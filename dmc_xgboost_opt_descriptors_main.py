import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostRegressor, Pool
import xgboost as xgb
import optuna
import matplotlib.pyplot as plt
import json

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

def objective_catboost(trial, X_train, X_valid, y_train, y_valid):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
        'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-3, 10),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_strength': trial.suggest_float('random_strength', 0.0, 1.0),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'loss_function': 'RMSE',
        'task_type': 'GPU',
        'devices': '0:1'
    }
    
    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train, eval_set=(X_valid, y_valid), verbose=0, early_stopping_rounds=100)
    
    y_pred = model.predict(X_valid)
    mse = mean_squared_error(y_valid, y_pred)
    return mse

def objective_xgboost(trial, X_train, X_valid, y_train, y_valid):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10),
        'device': 'cuda',
        'eval_metric': 'rmse',
        'early_stopping_rounds': 100
    }
    
    model = xgb.XGBRegressor(**params)
    eval_set = [(X_valid, y_valid)]
    model.fit(X_train, y_train, eval_set=eval_set, verbose=0)
    
    y_pred = model.predict(X_valid)
    mse = mean_squared_error(y_valid, y_pred)
    return mse

def process_file(npz_file, model_type='xgboost'):
    X, y = load_data(npz_file)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

    study = optuna.create_study(direction='minimize')
    
    if model_type == 'catboost':
        study.optimize(lambda trial: objective_catboost(trial, X_train, X_valid, y_train, y_valid), n_trials=50)
    elif model_type == 'xgboost':
        study.optimize(lambda trial: objective_xgboost(trial, X_train, X_valid, y_train, y_valid), n_trials=50)

    best_params = study.best_params

    if model_type == 'catboost':
        model = CatBoostRegressor(**best_params)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=100, early_stopping_rounds=100)
    elif model_type == 'xgboost':
        model = xgb.XGBRegressor(**best_params)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=100)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Best Model Results for {npz_file} using {model_type}: MSE={mse}, MAE={mae}, R2={r2}")
    print(f"Best Hyperparameters for {npz_file} using {model_type}: {best_params}")

    finger_name = npz_file.replace('VQM24/DMC_', '').replace('.npz', '')

    # 保存最佳模型
    model_filename = f'path/best_{model_type}_model_{finger_name}.model'
    model.save_model(model_filename)

    # 保存最佳参数
    params_filename = f'path/best_{model_type}_params_{finger_name}.json'
    with open(params_filename, 'w') as f:
        json.dump(best_params, f)

    # 保存预测值与真实值对比图
    plot_comparison(y_test, y_pred, mse, mae, r2, title=f'{model_type.capitalize()} Prediction vs Actual ({finger_name})', save_path=f'path/{model_type}_comparison_{finger_name}.png')


if __name__ == "__main__":

    npz_files = ['VQM24/DMC_descriptors_hashed_fingerprints.npz', 'VQM24/DMC_descriptors_hashed_2d_fingerprints.npz', 'VQM24/DMC_descriptors_2d_fingerprints.npz', 'VQM24/DMC_descriptors_fingerprints.npz']
    for npz_file in npz_files:
        process_file(npz_file, model_type='xgboost')
 
