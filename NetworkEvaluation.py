import argparse
import numpy as np
from keras.engine.saving import load_model
from sklearn.metrics import roc_auc_score

from DataParser import DataParser


class NetworkEvaluation:
    def __init__(self, folders, batch_size, models):
        self.models = []
        self.parsers_by_type = {}
        self.models_by_type = {}

        for model in models:
            if model[0] not in self.parsers_by_type:
                self.parsers_by_type[model[0]] = DataParser("testing", folders, model[0], batch_size=batch_size)
                self.models_by_type[model[0]] = []
            self.models_by_type[model[0]].append(load_model(model[1]))
        print(self.parsers_by_type)
        print(self.models_by_type)

    def evaluate(self):
        i = 0
        score = 0
        reference_parser = self.parsers_by_type[list(self.parsers_by_type.keys())[0]]
        generator = reference_parser.get_dataset_file_names_generator()
        n_iterations = len(reference_parser.graph_files_name) // reference_parser.batch_size
        while i < n_iterations:
            output_pos = [0] * reference_parser.batch_size
            (names, referenceOutput) = next(generator)
            for key, value in self.models_by_type.items():
                iteration_parser = self.parsers_by_type[key]
                graphs = iteration_parser.find_graphs_from_graphs(names)
                for model in value:
                    output = model.predict(np.array(iteration_parser.get_input_graphs_data(graphs)),
                                           batch_size=reference_parser.batch_size)
                    output_pos = np.amax([output_pos, list(map(lambda x: x[1], output))], axis=0)
            output_classes = list(map(lambda x: [1 - x, x], output_pos))
            output_classes = np.array(output_classes).argmax(axis=-1)
            score += roc_auc_score(referenceOutput, output_pos)
            i += 1
        print(score / i)


def main(models, folders, batch_size):
    evaluator = NetworkEvaluation(folders, batch_size, models)
    evaluator.evaluate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Customization options for the data learner")
    parser.add_argument("models", nargs='?',
                        default="melspectrogram,leonet_melspectrogram_ff_wr.h5 spectrogram,leonet_melspectrogram_ff_wr.h5",
                        help='Set models that will be used to '
                             'make the predictions')
    parser.add_argument("folders", nargs='?', default="BirdVoxDCASE20k",
                        help='Set of folders that will be used as the source of the graphs')
    parser.add_argument("batch_size", nargs='?', default=30,
                        help='Batch size of the files used to train the model')
    args = parser.parse_args()
    print([(item.strip().split(",")[0], item.strip().split(",")[1]) for item in args.models.strip().split(" ")])
    main([(item.strip().split(",")[0], item.strip().split(",")[1]) for item in args.models.strip().split(" ")],
         [item for item in args.folders.strip().split(',')], args.batch_size)
