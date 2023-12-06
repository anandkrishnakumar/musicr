import numpy as np

# Generate sine wave


def gen_sine(freq, sr, secs, noise_sd=0, amplitude=0.5, x_intercept=0):
    num = int(sr*secs)
    out = np.zeros((num, 2))
    out[:, 0] = np.sin(2 * np.pi * np.arange(num) *
                       freq / sr + x_intercept) * amplitude
    out[:, 1] = np.sin(2 * np.pi * np.arange(num) *
                       freq / sr + x_intercept) * amplitude

    # Add noise
    out += np.random.normal(0, noise_sd, size=(num, 2))

    return out


def gen_sequence(freqs, secs, wave_gen_fn=gen_sine, sr=48000, amplitude=1, x_intercept=0):
    out_lst = []
    for freq, sec in zip(freqs, secs):
        sine = wave_gen_fn(freq, sr, sec, amplitude=amplitude, x_intercept=x_intercept)
        out_lst.append(sine)
    return np.concatenate(out_lst)
