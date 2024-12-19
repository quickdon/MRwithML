import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import optuna
import matplotlib.pyplot as plt
import json

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
        'eval_metric': 'rmse',
        'early_stopping_rounds': 100
    }
    
    model = xgb.XGBRegressor(**params)
    eval_set = [(X_valid, y_valid)]
    model.fit(X_train, y_train, eval_set=eval_set, verbose=0)
    
    y_pred = model.predict(X_valid)
    mse = mean_squared_error(y_valid, y_pred)
    return mse

def stratified_split(data, target_column, test_size=0.2, n_splits=5, random_state=42):
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # 使用qcut将目标变量分层
    y_bins = pd.qcut(y, q=10, duplicates='drop')

    stratified_split = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    
    for train_index, test_index in stratified_split.split(X, y_bins):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    return X_train, y_train, X_test, y_test


def objective_lightgbm(trial, X_train, X_valid, y_train, y_valid):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 31, 255),
        'max_depth': trial.suggest_int('max_depth', -1, 15),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10),
        'objective': 'regression',
        'metric': 'rmse'
    }
    
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
    
    y_pred = model.predict(X_valid)
    mse = mean_squared_error(y_valid, y_pred)
    return mse

def process_file(npz_file, model_type='xgboost', target='Etot'):
    X, y = load_data(npz_file, target=target)
       
    if target not in ['Cp', 'Eatom']:
        df = pd.DataFrame(X)
        df[target] = y
        X_train, y_train, X_test, y_test = stratified_split(df, target_column=target)
        df_train = pd.DataFrame(X_train)
        df_train[target] = y_train
        X_train, y_train, X_valid, y_valid = stratified_split(df_train, target_column=target, test_size=0.25)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)    
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

    study = optuna.create_study(direction='minimize')
    
    if model_type == 'xgboost':
        study.optimize(lambda trial: objective_xgboost(trial, X_train, X_valid, y_train, y_valid), n_trials=50)
    elif model_type == 'lightgbm':
        study.optimize(lambda trial: objective_lightgbm(trial, X_train, X_valid, y_train, y_valid), n_trials=50)

    best_params = study.best_params

    if model_type == 'xgboost':
        model = xgb.XGBRegressor(**best_params)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=100)
    elif model_type == 'lightgbm':
        model = lgb.LGBMRegressor(**best_params)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Best Model Results for {npz_file} using {model_type} for {target}: MSE={mse}, MAE={mae}, R2={r2}")
    print(f"Best Hyperparameters for {npz_file} using {model_type} for {target}: {best_params}")

    finger_name = npz_file.replace('VQM24/DFT_', '').replace('.npz', '')

    # 保存最佳模型
    model_filename = f'path/best_{model_type}_model_{target}_{finger_name}.model'
    model.save_model(model_filename)

    # 保存最佳参数
    params_filename = f'path/best_{model_type}_params_{target}_{finger_name}.json'
    with open(params_filename, 'w') as f:
        json.dump(best_params, f)

    # 保存预测值与真实值对比图
    plot_comparison(y_test, y_pred, mse, mae, r2, target=target, title=f'{model_type.capitalize()} Prediction vs Actual {target} ({finger_name})', save_path=f'path/{model_type}_{target}_comparison_{finger_name}.png')


if __name__ == "__main__":
    npz_files = ['VQM24/DFT_descriptors_fingerprints.npz', 'VQM24/DFT_descriptors_2d_fingerprints.npz']
    targets1 = ['Etot', 'Exc', 'Eee']
    targets2 = ['Cp', 'Eatom']

    for npz_file in npz_files:
        for target in targets1:
            process_file(npz_file, model_type='xgboost', target=target)
            
    for npz_file in npz_files:
        for target in targets2:
            process_file(npz_file, model_type='xgboost', target=target)
            process_file(npz_file, model_type='lightgbm', target=target)
 