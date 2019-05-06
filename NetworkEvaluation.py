import argparse
import numpy as np
from keras.engine.saving import load_model
from sklearn.metrics import roc_auc_score

from DataParser import DataParser


class NetworkEvaluation:
    def __init__(self, data_parser, models):
        self.parser = data_parser
        self.models = []
        for model_name in models:
            self.models.append(load_model(model_name))

    def evaluate(self):
        i = 0
        score = 0
        generator = self.parser.get_dataset_plot_generator()
        while i < len(self.parser.graph_files_name) // self.parser.batch_size:
            (inputTesting, outputTrue) = next(generator)
            output_pos = [0] * self.parser.batch_size
            for model in self.models:
                output = model.predict(inputTesting, batch_size=self.parser.batch_size)
                output_pos = np.amax([output_pos, list(map(lambda x: x[1], output))], axis=0)
            score += roc_auc_score(outputTrue, output_pos)
            i += 1
        print(score / i)


def main(models, folders, type_graph, batch_size):
    data_parser = DataParser(folders=folders, graph_type=type_graph, batch_size=batch_size, type_folder="testing")
    evaluator = NetworkEvaluation(data_parser, models)
    evaluator.evaluate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Customization options for the data learner")
    parser.add_argument("models", nargs='?', type=list, default=["leonetv2_melspectrogram.h5", "leonet.h5"],
                        help='Set models that will be used to '
                             'make the predictions')
    parser.add_argument("type_graph", nargs='?', default="melspectrogram", help='Decide the type of graphs from the '
                                                                                'testing set that will be analyzed')
    parser.add_argument("folders", nargs='?', type=list, default=["ff1010bird"],
                        help='Set of folders that will be used as the source of the graphs')
    args = parser.parse_args()
    main(args.models, args.folders, args.type_graph, batch_size=30)
