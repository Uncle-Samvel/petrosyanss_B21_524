import os

import numpy as np
import matplotlib.pyplot as plt
import itertools

from scipy.io import wavfile
from scipy import signal
from tqdm import tqdm

dpi = 1000


def spectogram(samples, sample_rate, filename):
    freq, t, spec = signal.spectrogram(samples, sample_rate, scaling='spectrum', window=('hann'))

    spec = np.log10(spec + 1)
    plt.pcolormesh(t, freq, spec, shading='gouraud', vmin=spec.min(), vmax=spec.max())
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')

    plt.savefig(filename)

    return freq, t, spec


def create_butter_filter(sample_rate, data, output_dir):
    b, a = signal.butter(10, 0.1, btype='lowpass')
    filtered_signal = signal.filtfilt(b, a, data)
    wavfile.write(os.path.join(output_dir, 'butter.wav'), sample_rate, filtered_signal.astype(np.int16))
    spectogram(filtered_signal, sample_rate, os.path.join(output_dir, 'butter.png'))


def create_savgol_filter(sample_rate, data, output_dir):
    denoised_savgol = signal.savgol_filter(data, 75, 5)
    wavfile.write(os.path.join(output_dir, 'savgol.wav'), sample_rate, denoised_savgol.astype(np.int16))
    spectogram(denoised_savgol, sample_rate, os.path.join(output_dir, 'savgol.png'))


def find_peaks(sample_rate, data, output_dir):
    peaks = set()
    threshold_time = 0.1
    threshold_freq = 50

    freq, t, spec = spectogram(data, sample_rate, os.path.join(output_dir, 'input.png'))

    for i in tqdm(range(len(freq)), desc='find_peaks'):
        for j in range(len(t)):
            time_window = np.asarray(abs(t - t[j]) < threshold_time).nonzero()[0]
            freq_window = np.asarray(abs(freq - freq[i]) < threshold_freq).nonzero()[0]
            indexes = np.array([x for x in itertools.product(freq_window, time_window)])
            flag = True
            for a, b in indexes:
                if spec[i, j] <= spec[a, b] and i != a and i != b:
                    flag = False
                    break

            if flag:
                peaks.add(t[j])

    with open(os.path.join(output_dir, 'peaks.txt'), 'w') as f:
        f.write(str(len(peaks)))
        f.write('\n')
        f.write(str(peaks))


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(current_dir, 'input')
    output_path = os.path.join(current_dir, 'output')
    os.makedirs(output_path, exist_ok=True)

    sample_rate, data = wavfile.read(os.path.join(input_path, 'cover.wav'))

    create_butter_filter(sample_rate, data, output_path)
    print('done butter filter')

    create_savgol_filter(sample_rate, data, output_path)
    print('done savgol filter')

    find_peaks(sample_rate, data, output_path)


if __name__ == '__main__':
    main()