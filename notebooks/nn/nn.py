import numpy as np
import os

os.environ["KERAS_BACKEND"] = "jax"

# Note that Keras should only be imported after the backend
# has been configured. The backend cannot be changed once the
# package is imported.
import keras
from keras import Sequential
from keras.layers import Dense
# keras.utils.set_random_seed(777)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from utils.waves import WaveGenerator

def generate_data(N, secs=1/32):
    # Randomly generate frequencies between C2 and C8
    c2 = 65.41 # n = 16
    c8 = 4186.01 # n = 88
    c7 = 2093.00 # n = 76

    # Generate N training samples
    freqs = np.random.uniform(c2, c8, size=N)
    noise_sds = np.random.exponential(0.02, size=N)
    amplitudes = np.random.uniform(0.4, 1, size=N)
    x_intercepts = np.random.uniform(0, 2*np.pi, size=N)
    X_lst = []
    wg = WaveGenerator()
    for i in range(N):
        sine = wg.gen_wave('sine', freqs[i], secs=secs, noise_sd=noise_sds[i], amplitude=amplitudes[i], x_intercept=x_intercepts[i])
        X_lst.append(sine)

    X = np.stack(X_lst)
    y = np.stack([freqs,amplitudes]).T
    split = 0.7
    X_train = X[:int(split*len(X_lst))]
    y_train = y[:int(split*len(X_lst))]
    X_test = X[int(split*len(X_lst)):]
    y_test = y[int(split*len(X_lst)):]
    
    return X, y, X_train, y_train, X_test, y_test

def train_nn(model, X, y, epochs=None, batch_size_frac=0.01):
    if epochs is None:
        epochs = 3 * X.shape[1]
    callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    history = model.fit(X, y, validation_split=0.2, batch_size=int(len(X)*batch_size_frac), epochs=epochs, callbacks=[callback])
    return history, model

def generate_frequency(key_num):
    return np.round(np.power(2, (key_num-49)/12) * 440, 2)

def eval_nn(model, secs=1/32, max_freq=None):
    freqs = [generate_frequency(n) for n in range(16, 89)] # actual note frequencies
    if max_freq is not None:
        freqs = [y for y in freqs if y < max_freq]
    X_test = []
    y_test = []
    noise_sd = np.random.exponential(0.02)
    x_intercept = np.random.uniform(0, 2*np.pi)
    amplitudes = np.arange(0, 1, 0.05).round(2)
    wg = WaveGenerator()
    for i, freq in enumerate(freqs):
        for amplitude in amplitudes:
            wave = wg.gen_wave('sine', freq, secs=secs, noise_sd=noise_sd, amplitude=amplitude, x_intercept=x_intercept)
            X_test.append(wave)
            y_test.append(freq)
    X_test = np.stack(X_test) # waves generated with actual note frequencies
    y_pred = [pred[0] for pred in model.predict(X_test)]
    # plot grid of losses
    plot_heatmap(np.array(y_test), np.array(y_pred), np.tile(amplitudes, len(freqs)))

def plot_heatmap(true_frequencies, predicted_values, amplitudes):
    """
    Plot a heatmap for true frequencies, predicted values, and amplitudes.

    Parameters:
    - true_frequencies: List of true frequencies
    - predicted_values: List of predicted values
    - amplitudes: List of amplitudes corresponding to each pair

    Returns:
    - None
    """
    absolute_difference = np.abs(np.array(predicted_values) - np.array(true_frequencies))
    percentage_difference = absolute_difference / np.array(true_frequencies) * 100

    data = {'True Frequency': true_frequencies,
            'Amplitude': amplitudes,
            'Percentage Difference': percentage_difference}

    df = pd.DataFrame(data)

    # Create a pivot table for the heatmap
    pivot_table = df.pivot_table(values='Percentage Difference', index='Amplitude', columns='True Frequency', aggfunc='mean')

    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, cmap='viridis', annot=False, fmt=".2f", cbar_kws={'label': 'Percentage Difference'})
    plt.title('Heatmap of Percentage Difference for True Frequencies and Amplitudes')
    plt.show()

def create_nn(input_shape):
    # Build network
    model = keras.Sequential()
    model.add(Dense(input_shape, input_shape=(input_shape,), kernel_initializer='normal', activation='relu'))
    model.add(Dense(input_shape, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_absolute_percentage_error', optimizer='adam')
    
    # Display a summary of the model architecture.
    model.summary()

    return model

def create_nn2(input_shape):
    # Build network
    model = keras.Sequential()
    model.add(Dense(100, input_shape=(input_shape,), kernel_initializer='normal', activation='relu'))
    model.add(Dense(100, kernel_initializer='normal', activation='relu'))
    model.add(Dense(200, kernel_initializer='normal', activation='relu'))
    model.add(Dense(300, kernel_initializer='normal', activation='relu'))
    model.add(Dense(500, kernel_initializer='normal', activation='relu'))
    model.add(Dense(2, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_absolute_percentage_error', optimizer='adam')
    
    # Display a summary of the model architecture.
    model.summary()

    return model