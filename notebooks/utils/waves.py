import numpy as np

# Generate sine wave
def gen_sine(freq, sr, secs, noise_sd=None, amplitude=0.5, x_intercept=0):
    num = int(sr*secs)
    testing_np_version = np.zeros((num, 2))
    testing_np_version[:, 0] = np.sin(2 * np.pi * np.arange(num) * freq / sr + x_intercept) * amplitude
    testing_np_version[:, 1] = np.sin(2 * np.pi * np.arange(num) * freq / sr + x_intercept) * amplitude
    
    if noise_sd is not None:
        # Add noise
        testing_np_version += np.random.normal(0, noise_sd, size=(num, 2))

    return testing_np_version