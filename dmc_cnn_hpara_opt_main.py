import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout
from keras import Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import keras_tuner as kt
import tensorflow as tf
import json

def load_data(npz_file):
    data = np.load(npz_file, allow_pickle=True)
    X = data['fingerprints']
    y = data['Etot']
    return X, y

def set_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    try:
        for gpu in gpus:
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=22528)])
    except RuntimeError as e:
        print(e)

class MyHyperModel(kt.HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        model = Sequential()
        model.add(Input(shape=self.input_shape))
        
        for i in range(hp.Int("num_conv_layers", 1, 5)):
            model.add(Conv1D(filters=hp.Int(f'filters_{i}', min_value=32, max_value=128, step=32),
                             kernel_size=hp.Int(f'kernel_size_{i}', min_value=1, max_value=5, step=2),
                             activation='relu'))
            model.add(MaxPooling1D(pool_size=2))

        model.add(Flatten())
        
        if hp.Boolean(f'dropout_{i}'):
            model.add(Dropout(0.25))

        for i in range(hp.Int("num_dense_layers", 1, 3)):
            model.add(Dense(units=hp.Int(f'dense_units_{i}', min_value=500, max_value=5000, step=500),
                            activation='relu'))

        model.add(Dense(1))

        model.compile(optimizer=Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG', default=1e-3)),
                      loss='mse',
                      metrics=['mae'])
        return model

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

def train_on_gpu(npz_files, gpu_id):
    set_memory_growth()

    for npz_file in npz_files:
        finger_name = npz_file.replace('VQM24/DMC_', '').replace('_fingerprints.npz', '')
        
        X, y = load_data(npz_file)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        input_shape = (X.shape[1], 1)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        with tf.device(f'/GPU:{gpu_id}'):
            strategy = tf.distribute.OneDeviceStrategy(device=f'/GPU:{gpu_id}')

            with strategy.scope():
                hypermodel = MyHyperModel(input_shape=input_shape)

                tuner = kt.BayesianOptimization(
                    hypermodel,
                    objective='val_loss',
                    max_trials=25,
                    directory='cnn_tuner',
                    project_name=f'{finger_name}_cnn_optimization'
                )

                early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

                tuner.search(X_train, y_train, epochs=35, validation_split=0.2, callbacks=[early_stopping])

            best_model = tuner.get_best_models(num_models=1)[0]
            best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
            
            y_pred = best_model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"GPU {gpu_id} - Best Model Results for {finger_name}: MSE={mse}, MAE={mae}, R2={r2}")
            print(f"GPU {gpu_id} - Best Hyperparameters for {finger_name}: {best_hyperparameters.values}")
            
            model_filename = f'path/{finger_name}_best_cnn_model.h5'
            best_model.save(model_filename)
            
            params_filename = f'path/{finger_name}_best_cnn_params.json'
            with open(params_filename, 'w') as f:
                json.dump(best_hyperparameters.values, f)
                
            plot_comparison(y_test, y_pred, mse, mae, r2, title=f'HPO_CNN Prediction vs Actual Etot with {finger_name}', save_path=f'path/{finger_name}_hpo_cnn_comparison.png')
            

def main(npz_files):
    npz_files_gpu0 = npz_files[:len(npz_files)//2]
    npz_files_gpu1 = npz_files[len(npz_files)//2:]
    
    train_on_gpu(npz_files_gpu0, gpu_id=0)
    
    train_on_gpu(npz_files_gpu1, gpu_id=1)

if __name__ == "__main__":
    npz_files = ['VQM24/DMC_rdkit_fingerprints.npz', 'VQM24/DMC_pubchem_fingerprints.npz', 'VQM24/DMC_atompair_fingerprints.npz', 
                 'VQM24/DMC_pattern_fingerprints.npz', 'VQM24/DMC_layered_fingerprints.npz',
                 'VQM24/DMC_maccs_fingerprints.npz', 'VQM24/DMC_ecfp_fingerprints.npz']

    main(npz_files)
