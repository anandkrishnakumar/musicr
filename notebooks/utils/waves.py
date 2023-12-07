import numpy as np

class WaveGenerator:
    """
    A class for generating sequences of waveforms.

    Attributes:
    - sr (int): The sample rate.
    - waveform_functions (dict): A dictionary of supported waveform functions.

    Methods:
    - gen_wave(waveform_name, freq, secs, noise_sd=0, amplitude=0.5, x_intercept=0):
        Generates a waveform using the specified waveform function.

    - gen_sequence(waveform_name, freqs, secs, amplitude=1, x_intercept=0):
        Generates a sequence of waveforms with different frequencies and durations.

    Example Usage:
    --------------
    # Create an instance of the WaveGenerator class
    wave_gen = WaveGenerator(sr=48000)

    # Generate a sequence of sine waves with different frequencies and durations
    freqs = [440, 880, 1320]
    durations = [1, 2, 3]
    generated_sequence = wave_gen.gen_sequence('sine', freqs, durations)

    # Print or use the generated sequence as needed
    print(generated_sequence)
    """

    def __init__(self, sr=48000):
        """
        Initialize the WaveGenerator.

        Parameters:
        - sr (int): The sample rate.
        """
        self.sr = sr
        self.waveform_functions = {
            'sine': self._sine_wave,
            'saw': self._saw_wave,
            # Add more waveform functions as needed
        }

    def _sine_wave(self, time, freq, amplitude, x_intercept):
        return np.sin(2 * np.pi * time * freq + x_intercept) * amplitude
    
    def _saw_wave(self, time, freq, amplitude, x_intercept):
        p = 1/freq
        return 2 * (time/p - np.floor(1/2 + time/p))

    def gen_wave(self, waveform_name, freq, secs, noise_sd=0, amplitude=1, x_intercept=0, channels=1):
        """
        Generates a waveform using the specified waveform function.

        Parameters:
        - waveform_name (str): Name of the waveform function.
        - freq (float): The frequency of the waveform.
        - secs (float): The duration of the waveform in seconds.
        - noise_sd (float): Standard deviation of noise to be added (default is 0).
        - amplitude (float): Amplitude of the waveform (default is 0.5).
        - x_intercept (float): Phase shift of the waveform (default is 0).
        - channels (int): Number of channels to be output (default is 1).

        Returns:
        - out (numpy.ndarray): The generated waveform.
        """
        if waveform_name not in self.waveform_functions:
            raise ValueError(f"Waveform function '{waveform_name}' not supported.")

        waveform_fn = self.waveform_functions[waveform_name]
        num = int(self.sr * secs)
        out = np.zeros((num, channels))
        time = np.arange(num) / self.sr
        for channel in range(channels):
            out[:, channel] = waveform_fn(time, freq, amplitude, x_intercept)

        # Add noise
        out += np.random.normal(0, noise_sd, size=(num, channels))

        if channels == 1:
            return out[:, 0]
        else:
            return out

    def gen_sequence(self, waveform_name, freqs, secs, amplitude=1, x_intercept=0, channels=1):
        """
        Generates a sequence of waveforms with different frequencies and durations.

        Parameters:
        - waveform_name (str): Name of the waveform function.
        - freqs (list): List of frequencies for each waveform.
        - secs (list): List of durations (in seconds) for each waveform.
        - amplitude (float): Amplitude of the waveforms (default is 1).
        - x_intercept (float): Phase shift of the waveforms (default is 0).
        - channels (int): Number of channels to be output (default is 1).

        Returns:
        - generated_sequence (numpy.ndarray): The concatenated sequence of generated waveforms.
        """
        out_lst = []
        for freq, sec in zip(freqs, secs):
            wave = self.gen_wave(waveform_name, freq, sec, amplitude=amplitude, x_intercept=x_intercept, channels=channels)
            out_lst.append(wave)
        return np.concatenate(out_lst)
