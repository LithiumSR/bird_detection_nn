import glob
import ntpath
import os

import numpy as np
from keras.preprocessing import image

ntpath.basename("a/b/c")

from AudioAugmentation import AudioAugmentation


class DataParser:

    def __init__(self, type_folder="training", folders=None, graph_type="melspectrogram", batch_size=20):
        self.typeFolder = type_folder
        self.folders = folders
        self.graph_type = graph_type
        self.batch_size = batch_size
        self.augmentation = AudioAugmentation()
        self.labels = {}
        if self.folders is None:
            self.folders = ["ff1010bird"]
        self._load_labels()
        self.audio_files_name = self._get_audio_files_name()
        self.graph_files_name = self._get_graph_files_name()

    def set_augmentation(self, augmentation):
        self.augmentation = augmentation

    def get_dataset_generator(self):
        # random_files = np.random.choice(a=self.graph_files_name,size=self.batch_size)
        i = 0
        file_list = self.graph_files_name
        while True:
            samples = []
            for b in range(self.batch_size):
                if i == len(file_list):
                    i = 0
                    import random
                    random.shuffle(self.graph_files_name)
                sample = file_list[i]
                i += 1
                samples.append(sample)
            batch_input = self.get_input_graphs_data(samples)
            batch_output = self.get_input_labels(samples)
            # print(batch_output)
            yield (np.array(batch_input), np.array(batch_output))

    def find_graphs_from_graphs(self, list_filepaths):
        ret = []
        for el in list_filepaths:
            file_name = os.path.splitext(DataParser.path_leaf(el))[0]
            folder = os.path.basename(os.path.dirname(os.path.dirname(file)))
            ret.append(path = os.getcwd() + "/data/graphs/" + self.typeFolder + "/" + folder + "/"+file_name)
        return ret

    def get_audio_files_name(self):
        return self.audio_files_name

    def get_graph_files_name(self):
        return self.graph_files_name

    def _get_audio_files_name(self):
        entries = []
        for folder in self.folders:
            files = glob.glob(os.getcwd() + "/data/audio/" + self.typeFolder + "/" + folder + "/*.wav")
            entries.extend(files)
        return entries

    def _get_graph_files_name(self):
        entries = []
        for folder in self.folders:
            files = glob.glob(
                os.getcwd() + "/data/graphs/" + self.typeFolder + "/" + folder + "/" + self.graph_type + "/*.png")
            entries.extend(files)
        return entries

    def _load_labels(self):
        for folder in self.folders:
            path = os.getcwd() + "/data/audio/" + self.typeFolder + "/" + folder + "/labels.csv"
            with open(path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    elements = line.split(",")
                    if elements[2].strip() != "hasbird":
                        self.labels[folder + "_" + elements[0]] = int(elements[2].strip())

    def get_input_labels(self, files):
        labels = []
        for file in files:
            file_name = os.path.splitext(DataParser.path_leaf(file))[0]
            if "_" in file_name:
                file_name = file_name.split("_")[0]
            if "graphs" in file:
                folder = os.path.basename(os.path.dirname(os.path.dirname(file)))
            else:
                folder = os.path.basename((os.path.dirname(file)))
            if self.labels[folder + "_" + file_name] == 0:
                labels.append(0)
            else:
                labels.append(1)
        return labels

    @staticmethod
    def path_leaf(path):
        head, tail = ntpath.split(path)
        return tail or ntpath.basename(head)

    @staticmethod
    def get_input_graphs_data(files):
        entries = []
        for file in files:
            img = image.load_img(file, target_size=(224, 224))
            img = image.img_to_array(img)
            entries += [img]
        return entries
