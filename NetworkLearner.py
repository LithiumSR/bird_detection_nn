import argparse

from keras.callbacks import EarlyStopping, ModelCheckpoint

from DataParser import DataParser


class DataLearner:
    def __init__(self, neural_network, data_parser, epochs=40, early_stopping=True, save_best_checkpoint=True):
        self.neural_network = neural_network
        self.epochs = epochs
        self.data_parser = data_parser
        self.early_stopping = early_stopping
        self.save_best_checkpoint = save_best_checkpoint

    def save(self, model):
        model.save(self.get_model_save_name())

    def get_model_save_name(self, checkpoint=False):
        ret = self.neural_network + "_" + self.data_parser.graph_type + ".h5"
        if checkpoint:
            return "checkpoint_" + ret
        return ret

    def train(self):
        generator = self.data_parser.get_dataset_plot_generator()
        steps = len(self.data_parser.graph_files_name) // self.data_parser.batch_size
        cb = []
        if self.early_stopping:
            cb.append(EarlyStopping(monitor='val_loss', mode='min', verbose=1))

        if self.save_best_checkpoint:
            cb.append(ModelCheckpoint(self.get_model_save_name(checkpoint=True), monitor='val_loss', mode='min',
                                      save_best_only=True))

        if self.neural_network == "vgg16":
            from models import VGG16
            model = VGG16.vgg16_model()
            model.fit_generator(generator, steps_per_epoch=steps, epochs=self.epochs, callbacks=cb)
            return model
        elif self.neural_network == "vgg19":
            from keras.applications import VGG19
            model = VGG19(include_top=True)
            model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.fit_generator(generator, epochs=self.epochs, steps_per_epoch=steps, callbacks=cb)
            return model
        elif self.neural_network == "cifar10":
            from models import Cifar10
            model = Cifar10.cifar10_model((1, 224, 224, 3))
            model.fit_generator(generator=generator, epochs=self.epochs, steps_per_epoch=steps, callbacks=cb)
            return model
        elif self.neural_network == "leonet":
            from models import LeoNet
            model = LeoNet.leonet_model((1, 224, 224, 3))
            model.fit_generator(generator=generator, epochs=self.epochs, steps_per_epoch=steps, callbacks=cb)
            return model

        elif self.neural_network == "leonetv2":
            from models import LeoNetV2
            model = LeoNetV2.LeoNetV2_model((1, 224, 224, 3))
            model.fit_generator(generator=generator, epochs=self.epochs, steps_per_epoch=steps, callbacks=cb)
            return model


def main(neural_network, type_graph, folders, batch_size):
    data_parser = DataParser(folders=folders, graph_type=type_graph, batch_size=batch_size)
    learner = DataLearner(neural_network, data_parser)
    model = learner.train()
    learner.save(model)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Customization options for the data learner")
    parser.add_argument("type_graph", nargs='?', default="melspectrogram", help='Set type of graph')
    parser.add_argument("neural_network", nargs='?', default="leonetv2", help='Set nn type')
    parser.add_argument("folders", nargs='?', type=list, default=["ff1010bird"],
                        help='Set of folders that will be used as the source of the graphs')
    parser.add_argument("batch_size", nargs='?', default=30,
                        help='Batch size of the files used to train the model')
    parser.add_argument("epochs", nargs='?', default=1,
                        help='Number of epochs')

    args = parser.parse_args()
    main(args.neural_network, args.type_graph, args.folders, args.batch_size)
