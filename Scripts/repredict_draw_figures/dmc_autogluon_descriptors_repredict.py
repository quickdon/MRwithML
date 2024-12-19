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
    df = pd.DataFrame(X)
    df['Etot'] = y
    
    train_data = df.sample(frac=0.8, random_state=42)
    test_data = df.drop(train_data.index)
    y_test = test_data['Etot']
    X_test = test_data.drop(columns=['Etot'])
    y_train = train_data['Etot']
    X_train = train_data.drop(columns=['Etot'])
    
    model = TabularPredictor(label='Etot').load(model_file)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    y_pred0 = model.predict(X_train)
    mse0 = mean_squared_error(y_train, y_pred0)
    mae0 = mean_absolute_error(y_train, y_pred0)
    r20 = r2_score(y_train, y_pred0)
    
    plot_comparison(y_test, y_pred, mse, mae, r2, title='Prediction vs Actual E$_{tot}$ (AutoGluon with Hashed Descriptors)', save_path=f'path/descriptors_hashed_autogluon_comparison.png')
    plot_comparison(y_train, y_pred0, mse0, mae0, r20, title='Prediction vs Actual E$_{tot}$ (AutoGluon with Hashed Descriptors)', save_path=f'path/descriptors_hashed_autogluon_comparison_train.png')
    
if __name__ == "__main__":
    npz_file = 'VQM24/DMC_descriptors_hashed_fingerprints.npz'
    # npz_file = 'VQM24/DMC_descriptors_fingerprints.npz'
    model_file = 'AutogluonModels/Descriptors_hashed'
    # model_file = 'AutogluonModels/Descriptors'
    
    main(npz_file, model_file)