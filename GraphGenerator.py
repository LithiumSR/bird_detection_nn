import errno
import os
import random
import sys
import argparse
import numpy as np
from tqdm import tqdm
from AudioAugmentation import AudioAugmentation
from DataParser import DataParser
from common import Utils


class GraphGenerator:

    def __init__(self, type_graph, folder_type, folders=None, augmentation=None,
                 skip_probability=0, save_raw=False, ):
        if folders is None:
            folders = ["ff1010bird"]
        self.files = DataParser(type_folder=folder_type, folders=folders, graph_type=type_graph).get_audio_files_name()
        self.type_graph = type_graph
        self.folder_type = folder_type
        self.aug = augmentation
        self.save_raw = save_raw
        self.skip_prob = skip_probability

    def generateGraph(self):
        for file in tqdm(self.files):
            if self.trigger_with_prob():
                pass
            folder = os.path.basename((os.path.dirname(file)))
            file_name = os.path.splitext(DataParser.path_leaf(file))[0]
            if self.aug is not None:
                file_name += "_" + self.aug.get_file_label()
            path_output_graph = os.getcwd() + "/data/graphs/" + self.folder_type + "/" + folder + "/" + self.type_graph + "/"
            self._makedirs(path_output_graph)
            path_output_graph = os.path.join(path_output_graph, file_name)
            data, sr = self._get_plot_data(file)
            self._write_graph(data, sr, path_output_graph)

            if self.save_raw:
                path_output_raw = os.getcwd() + "/data/raw/" + self.folder_type + "/" + folder + "/" + self.type_graph + "/"
                self._makedirs(path_output_raw)
                path_output_raw = os.path.join(path_output_raw, file_name)
                self._write_raw_file(data, path_output_raw)

    def _get_plot_data(self, file):
        data, sr = Utils.read_audio_file(file)
        if self.aug is not None:
            data = aug.augment_data(data, sr)
        return Utils.get_plot_data(data, sr, self.type_graph), sr

    def _write_graph(self, data, sr, save_path):
        Utils.write_graph(data, sr, save_path, self.type_graph)

    def _write_raw_file(self, data, save_path):
        arr = np.array(data)
        np.save(save_path, arr)

    def _makedirs(self, path):
        if not os.path.exists(os.path.dirname(path)):
            try:
                os.makedirs(os.path.dirname(path))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

    def trigger_with_prob(self):
        return random.random() > self.skip_prob


def main(type_graph, folder_type, folders, augmentation, skip_probability):
    g = GraphGenerator(type_graph, folder_type, folders, augmentation, skip_probability)
    g.generateGraph()

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Customization options for the graph generator")
    parser.add_argument("type_graph", nargs='?', default="spectrogram", help='Set type of graph')
    parser.add_argument("folder_type", nargs='?', default="training", help='Set folder type')
    parser.add_argument("folders", nargs='?', default="BirdVoxDCASE20k,ff1010bird,warblrb10k",
                        help='Folders that will be parsed')
    parser.add_argument("additive_noise", nargs='?', type=int, default=0, help='Additive noise')
    parser.add_argument("random_noise", nargs='?', type=bool, default=False, help='Random noise')
    parser.add_argument("time_stretch_rate", nargs='?', type=int, default=1, help='Time stretch')
    parser.add_argument("skip_probability", nargs='?', type=int, default=0, help='Time stretch')
    args = parser.parse_args()
    print(args)
    aug = AudioAugmentation()
    check = False
    if args.random_noise:
        aug.add_random_noise()
        check = True
    if args.additive_noise > 0:
        aug.add_noise(args.additive_noise)
        check = True
    if args.time_stretch_rate != 1 and args.time_stretch_rate > 0:
        aug.time_stretch(args.time_stretch_rate)
        check = True
    if not check:
        aug = None
    main(args.type_graph, args.folder_type, [item.strip() for item in args.folders.strip().split(',')], aug, args.skip_probability)
