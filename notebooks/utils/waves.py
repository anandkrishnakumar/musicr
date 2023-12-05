import numpy as np

# Generate sine wave
def gen_sine(freq, sr, secs, noise_sd=False, amplitude=0.5):
    num = int(sr*secs)
    testing_np_version = np.zeros((num, 2))
    testing_np_version[:, 0] = np.sin(2 * np.pi * np.arange(num) * freq / sr) * amplitude
    testing_np_version[:, 1] = np.sin(2 * np.pi * np.arange(num) * freq / sr) * amplitude
    
    if noise_sd is not None:
        # Add noise
        testing_np_version[:, 0] += np.random.normal(0, noise_sd, size=num)
        testing_np_version[:, 1] += np.random.normal(0, noise_sd, size=num)

    return testing_np_version