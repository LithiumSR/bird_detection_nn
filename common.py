import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np


class Utils:
    @staticmethod
    def read_audio_file(file_path):
        data, sr = librosa.core.load(file_path, mono=True, sr=None)
        return data, sr

    @staticmethod
    def write_audio_file(file_path, data, sr):
        librosa.output.write_wav(file_path, data, sr)

    @staticmethod
    def get_plot_data(data, sr, graph_type):
        if graph_type == "melspectrogram":
            S = librosa.feature.melspectrogram(y=data, sr=sr)
            return librosa.power_to_db(S, ref=np.max)

        elif graph_type == "melspectrogram-energy":
            S = librosa.feature.melspectrogram(y=data, sr=sr, power=1)
            return librosa.amplitude_to_db(S, ref=np.max)

        elif graph_type == "spectrogram":
            stft = librosa.core.spectrum.stft(data, hop_length=512)
            return librosa.amplitude_to_db(np.abs(stft), ref=np.max)

        elif graph_type == "filterbank":
            S = librosa.feature.melspectrogram(y=data, sr=sr, hop_length=512)
            return librosa.core.amplitude_to_db(S, ref=np.max)

    @staticmethod
    def write_graph(data, sr, file_path, graph_type):
        fig = plt.figure(figsize=(10, 4))
        if graph_type == "melspectrogram":
            librosa.display.specshow(data, sr=sr,
                                     y_axis='mel', fmax=8000, x_axis='time', ax=None)
        elif graph_type == "melspectrogram-energy":
            librosa.display.specshow(data, sr=sr,
                                     y_axis='mel', fmax=8000, x_axis='time', ax=None)
        elif graph_type == "spectrogram":
            librosa.display.specshow(data, sr=sr, ax=None, y_axis='log',
                                     x_axis='time')
        elif graph_type == "filterbank":
            librosa.display.specshow(data, sr=sr, ax=None, y_axis='log', hop_length=512, x_axis='frames')
        plt.margins(0)
        for ax in fig.get_axes():
            ax.spines['left'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.xaxis.label.set_visible(False)
            ax.yaxis.label.set_visible(False)
            ax.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()
