import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import optuna
import pickle
import json

def load_data(npz_file):
    data = np.load(npz_file, allow_pickle=True)
    X = data['fingerprints']
    y = data['Etot']
    return X, y

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, mae, r2, y_pred

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

def bayesian_optimization(model, param_space, X_train, y_train):
    bayes_search = BayesSearchCV(estimator=model, search_spaces=param_space, n_iter=50, cv=KFold(n_splits=5), n_jobs=-1)
    bayes_search.fit(X_train, y_train)
    return bayes_search.best_estimator_, bayes_search.best_params_


def objective(trial, X_train, X_valid, y_train, y_valid):
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
        'devices': '0:1'  # 指定使用的GPU设备
    }
    
    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train, eval_set=(X_valid, y_valid), verbose=0, early_stopping_rounds=100)
    
    y_pred = model.predict(X_valid)
    mse = mean_squared_error(y_valid, y_pred)
    return mse

def main(npz_file):
    X, y = load_data(npz_file)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    finger_name = npz_file.replace('VQM24/DMC_', '').replace('_fingerprints.npz', '')
    
    models = {
        'xgboost': (XGBRegressor(), {
            'n_estimators': Integer(10, 1000),
            'max_depth': Integer(1, 50),
            'max_leaves': Integer(2, 50),
            'learning_rate': Real(1e-3, 1, prior='log-uniform'),
            'subsample': Real(0.1, 1.0),
            'colsample_bytree': Real(0.1, 1.0)
        }),
        'adaboost': (AdaBoostRegressor(estimator=DecisionTreeRegressor()), {
            'n_estimators': Integer(10, 1000),
            'learning_rate': Real(1e-3, 1, prior='log-uniform'),
            'loss': Categorical(['linear', 'square', 'exponential']),
            'estimator__criterion':Categorical(['squared_error', 'friedman_mse', 'absolute_error']),
            'estimator__max_depth':Integer(1, 50),
            'estimator__min_samples_split':Integer(2, 20),
            'estimator__min_samples_leaf':Integer(1, 20)
        })
    }
    
    results = []
    
    for model_name, (model, param_space) in models.items():
        print(f"Optimizing {model_name}...")
        best_model, best_params = bayesian_optimization(model, param_space, X_train, y_train)
        mse, mae, r2, y_pred = evaluate_model(best_model, X_test, y_test)
        
        result = {
            'model': model_name,
            'best_params': best_params,
            'mse': mse,
            'mae': mae,
            'r2': r2
        }
        
        results.append(result)
        
        print(f"Results for {model_name}: MSE={mse}, MAE={mae}, R2={r2}")
        
        plot_comparison(y_test, y_pred, mse, mae, r2, title=f'Prediction vs Actual Etot ({model_name} with {finger_name})', save_path=f'path/{finger_name}_{model_name}_comparison.png')
        
        # 保存最佳模型和参数
        model_filename = f"path/{finger_name}_{model_name}_best_model.pkl"
        with open(model_filename, 'wb') as f:
            pickle.dump(best_model, f)
        
        params_filename = f"path/{finger_name}_{model_name}_best_params.json"
        with open(params_filename, 'w') as f:
            json.dump(best_params, f)
    
def main1(npz_file):
    X, y = load_data(npz_file)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, X_valid, y_train, y_valid), n_trials=50)

    best_params = study.best_params

    model = CatBoostRegressor(**best_params)
    model.fit(X_train, y_train, eval_set=(X_valid, y_valid), verbose=100, early_stopping_rounds=100)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Best Model Results for {npz_file}: MSE={mse}, MAE={mae}, R2={r2}")
    print(f"Best Hyperparameters for {npz_file}: {best_params}")
        
    # 保存最佳模型
    model_filename = f'path/best_catboost_model_{npz_file.split("/")[-1].split("_")[1]}.cbm'
    model.save_model(model_filename)
        
    # 保存最佳参数
    params_filename = f'path/best_catboost_params_{npz_file.split("/")[-1].split("_")[1]}.json'
    with open(params_filename, 'w') as f:
        json.dump(best_params, f)

    # 保存预测值与真实值对比图
    plot_comparison(y_test, y_pred, mse, mae, r2, title=f'CatBoost Prediction vs Actual Etot ({npz_file})', save_path=f'path/catboost_comparison_{npz_file.split("/")[-1].split("_")[1]}.png')

if __name__ == "__main__":

    npz_files = ['VQM24/DMC_pattern_fingerprints.npz', 'VQM24/DMC_layered_fingerprints.npz']
    for npz_file in npz_files:
        main(npz_file)
        # main1(npz_file)
