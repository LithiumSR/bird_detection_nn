import errno
import os
import sys
import argparse
from tqdm import tqdm
from AudioAugmentation import AudioAugmentation
from DataParser import DataParser


class GraphGenerator:

    def __init__(self, type_graph="melspectrogram", folder_type="training", folders=["ff1010bird"], augmentation=None):
        parser = DataParser(type_folder=folder_type, folders=folders)
        self.files = parser.get_audio_files_name()
        self.type_graph = type_graph
        self.folder_type = folder_type
        self.aug = augmentation

    def generateGraph(self):
        for file in tqdm(self.files):
            folder = os.path.basename((os.path.dirname(file)))
            file_name = os.path.splitext(DataParser.path_leaf(file))[0]
            if self.aug is not None:
                file_name += "_" + self.aug.get_file_label()
            path_output = os.getcwd() + "/data/graphs/" + self.folder_type + "/" + folder + "/" + self.type_graph + "/"
            if not os.path.exists(os.path.dirname(path_output)):
                try:
                    os.makedirs(os.path.dirname(path_output))
                except OSError as exc:  # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise
            path_output = os.path.join(path_output, file_name)
            print(file_name)
            self._generateGraph(file, path_output)

    def _generateGraph(self, file, save_path):
        from common import Utils
        data, sr = Utils.read_audio_file(file)
        if self.aug is not None:
            data = aug.augment_data(data, sr)
        Utils.write_graph(data, sr, save_path, self.type_graph)


def main(type_graph, folder_type, folders, augmentation):
    g = GraphGenerator(type_graph, folder_type, folders, augmentation)
    g.generateGraph()

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Customization options for the graph generator")
    parser.add_argument("type_graph", nargs='?', default="melspectrogram", help='Set type of graph')
    parser.add_argument("folder_type", nargs='?', default="training", help='Set folder type')
    parser.add_argument("folders", nargs='?', type=list, default=["ff1010bird"],
                        help='Number of split in kcross validation (default '
                             'is 3)')
    parser.add_argument("additive_noise", nargs='?', type=int, default=0, help='Additive noise')
    parser.add_argument("random_noise", nargs='?', type=bool, default=False, help='Random noise')
    parser.add_argument("time_stretch_rate", nargs='?', type=int, default=1, help='Time stretch')
    args = parser.parse_args()
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
    main(args.type_graph, args.folder_type, args.folders, aug)
