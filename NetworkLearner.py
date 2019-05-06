import argparse

from keras.utils import plot_model

from DataParser import DataParser


class DataLearner:
    def __init__(self, neural_network, parser, epochs=20):
        self.neural_network = neural_network
        self.epochs = epochs
        self.parser = parser

    def save(self, model):
        if self.neural_network == "vgg16":
            model.save("vgg16.h5")
        elif self.neural_network == "vgg19":
            model.save("vgg19.h5")
        elif self.neural_network == "cifar10":
            model.save("cifar10.h5")
        elif self.neural_network == "leonet":
            model.save("leonet.h5")
        elif self.neural_network == "leonetv2":
            model.save("leonetv2.h5")

    def train(self):
        generator = self.parser.get_dataset_generator()
        if self.neural_network == "vgg16":
            from models import VGG16
            model = VGG16.vgg16_model()
            model.fit_generator(generator, steps_per_epoch=len(self.parser.graph_files_name) // self.parser.batch_size, epochs=10)
            return model
        elif self.neural_network == "vgg19":
            from keras.applications import VGG19
            model = VGG19(include_top=True)
            model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.fit_generator(generator, steps_per_epoch=len(self.parser.graph_files_name) // self.parser.batch_size)
            return model
        elif self.neural_network == "cifar10":
            from models import Cifar10
            model = Cifar10.cifar10_model((1, 224, 224, 3))
            model.fit_generator(generator=generator, epochs=self.epochs,
                                steps_per_epoch=len(self.parser.graph_files_name) // self.parser.batch_size)
            return model
        elif self.neural_network == "leonet":
            from models import LeoNet
            model = LeoNet.leonet_model((1, 224, 224, 3))
            model.fit_generator(generator=generator, epochs=self.epochs,
                                steps_per_epoch=len(self.parser.graph_files_name) // self.parser.batch_size)
            return model

        elif self.neural_network == "leonetv2":
            from models import LeoNetV2
            model = LeoNetV2.LeoNetV2_model((1, 224, 224, 3))
            model.fit_generator(generator=generator, epochs=self.epochs,
                                steps_per_epoch=len(self.parser.graph_files_name) // self.parser.batch_size)
            return model


def main(neural_network, type_graph, folders, batch_size):
    data_parser = DataParser(folders=folders, graph_type=type_graph, batch_size=batch_size)
    data_parser2 = DataParser(folders=folders, graph_type=type_graph, batch_size=batch_size, type_folder="testing")
    learner = DataLearner(neural_network, data_parser)
    model = learner.train()
    learner.save(model)

    #from NetworkPredict import NetworkPredict
    #pred = NetworkPredict(data_parser2, model)
    #pred.predict()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Customization options for the data learner")
    parser.add_argument("type_graph", nargs='?', default="melspectrogram", help='Set type of graph')
    parser.add_argument("neural_network", nargs='?', default="leonet", help='Set nn type')
    parser.add_argument("folders", nargs='?', type=list, default=["ff1010bird"],
                        help='Set of folders that will be used as the source of the graphs')
    parser.add_argument("batch_size", nargs='?', default=20,
                        help='Batch size of the files used to train the model')
    parser.add_argument("epochs", nargs='?', default=1,
                        help='Number of epochs')

    args = parser.parse_args()
    main(args.neural_network, args.type_graph, args.folders, args.batch_size)
