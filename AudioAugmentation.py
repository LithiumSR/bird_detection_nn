import librosa
import numpy as np

from common import Utils


class AudioAugmentation:

    def __init__(self):
        self.random_noise = False
        self.additive_noise = 0
        self.n_steps = -1
        self.bins_per_octave = 12
        self.time_stretch_rate = -1

    def augment_file(self, file_path):
        out, sr = Utils.read_audio_file(file_path)
        return self.augment_data(out, sr)

    def augment_data(self, data, sr):
        if self.time_stretch_rate != -1:
            data = AudioAugmentation._time_stretch(data, self.time_stretch_rate, len(data))
        if self.additive_noise > 0:
            data = AudioAugmentation._add_noise(data, self.additive_noise)
        if self.random_noise:
            data = AudioAugmentation._add_random_noise(data)
        if self.n_steps != -1:
            data = AudioAugmentation._pitch_shift(data, sr, self.n_steps, self.bins_per_octave)

        # data = data[:int(sr * 1)]
        # data = data[np.newaxis, :]
        return data

    def get_file_label(self):
        ret = ""
        if self.additive_noise > 0:
            ret += "an=" + str(self.additive_noise) + ","
        if self.random_noise:
            ret += "rn=y" + ","
        if self.n_steps != -1:
            ret += "ps=y" + ","
        if self.time_stretch_rate != -1:
            ret += "tsr=" + str(self.time_stretch_rate).replace(".",",")
        return ret

    def add_random_noise(self):
        self.random_noise = True

    @staticmethod
    def _add_random_noise(data):
        noise = np.random.randn(len(data))
        data_noise = data + 0.005 * noise
        return data_noise

    def add_noise(self, noise):
        self.additive_noise = noise

    @staticmethod
    def _add_noise(data, noise):
        noise = np.array([noise] * len(data))
        data_noise = data + noise
        return data_noise

    def pitch_shift(self, n_steps, bins_per_octave):
        self.n_steps = n_steps
        self.bins_per_octave = bins_per_octave

    @staticmethod
    def _pitch_shift(data, sr, n_steps, bins_per_octave):
        return librosa.effects.pitch_shift(data, sr, n_steps=n_steps, bins_per_octave=bins_per_octave)

    @staticmethod
    def time_shift(data):
        return np.roll(data, 1600)

    def time_stretch(self, rate):
        self.time_stretch_rate = rate

    @staticmethod
    def _time_stretch(data, rate, input_length):
        if input_length is None:
            input_length = len(data)
        data = librosa.effects.time_stretch(data, rate)
        if len(data) > input_length:
            data = data[:input_length]
        else:
            data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
        return data
