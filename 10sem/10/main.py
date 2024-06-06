def get_max_tembr(file_path):
    data, sample_rate = librosa.load(file_path)
    chroma = librosa.feature.chroma_stft(y=data, sr=sample_rate)
    f0 = librosa.piptrack(y=data, sr=sample_rate, S=chroma)[0]
    max_f0 = np.argmax(f0)
    return max_f0


def spectrogram(samples, sample_rate, file_path):
    frequency, time_data, spec = signal.spectrogram(samples, sample_rate, window=('hann'))
    spec = np.log10(spec + 1)
    plt.pcolormesh(time_data, frequency, spec, shading='gouraud', vmin=spec.min(), vmax=spec.max())
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.savefig(file_path)

    return frequency, time_data, spec


def get_peaks(frequency, time_data, spec):
    delta_time = int(0.1 * len(time_data))
    delta_frequency = int(50 / (frequency[1] - frequency[0]))
    filtered = maximum_filter(spec, size=(delta_frequency, delta_time))

    peaks_mask = (spec == filtered)
    peak_values = spec[peaks_mask]
    peak_frequencies = frequency[peaks_mask.any(axis=1)]

    top_indices = np.argsort(peak_values)[-3:]
    top_frequencies = peak_frequencies[top_indices]

    return list(top_frequencies)


def get_max_min(voice_path):
    sound, sr = librosa.load(voice_path, sr=None)

    D = librosa.amplitude_to_db(np.abs(librosa.stft(sound)), ref=np.max)

    frequencies = librosa.fft_frequencies(sr=sr)
    mean_spec = np.mean(D, axis=1)

    idx_min = np.argmax(mean_spec > -80)
    idx_max = len(mean_spec) - np.argmax(mean_spec[::-1] > -80) - 1

    min_freq = frequencies[idx_min]
    max_freq = frequencies[idx_max]

    return max_freq, min_freq


def main():
    current_path = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_path, 'output')
    os.makedirs(output_path, exist_ok=True)

    input_path = os.path.join(current_path, 'input')
    humiliations = ['letter_a', 'letter_i', 'gav']
    humiliation_voice_paths = [
        (humiliation, os.path.join(input_path, '%s.wav' % humiliation))
        for humiliation in humiliations
    ]
    with open(os.path.join(output_path, 'res.txt'), 'w') as res_file:
        for humiliation, voice_path in humiliation_voice_paths:
            rate, samples = wavfile.read(voice_path)
            frequency, time_data, spec = spectrogram(samples, rate, os.path.join(output_path, '%s.png' % humiliation))
            max_freq, min_freq = get_max_min(voice_path)

            res_file.write('%s:\n' % humiliation)
            res_file.write('\tМаксимальная частота: %s\n' % max_freq)
            res_file.write('\tМинимальная частота: %s\n' % min_freq)
            res_file.write("\tНаиболее тембрально окрашенный основной тон: %s. "
                           "Это частота, для которой прослеживается наибольшее количество обертонов\n" % get_max_tembr(voice_path))
            if 'letter' in humiliation:
                res_file.write("\tТри самые сильные форманты: %s. "
                               "Это частоты с наибольшей энергией в некоторой окрестности\n" %
                               get_peaks(frequency, time_data, spec))


if __name__ == "__main__":
    main()