import argparse

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD

from DataParser import DataParser


class DataLearner:
    def __init__(self, neural_network, data_parser, epochs=30, early_stopping=False, save_best_checkpoint=True,
                 use_validation_set=False):
        self.neural_network = neural_network
        self.epochs = epochs
        self.data_parser = data_parser
        self.early_stopping = early_stopping
        self.save_best_checkpoint = save_best_checkpoint
        self.use_validation_set = use_validation_set

    def save(self, model):
        model.save(self.get_model_save_name())

    def get_model_save_name(self, checkpoint=False):
        ret = self.neural_network + "_" + self.data_parser.graph_type + ".h5"
        if checkpoint:
            return "best_" + ret
        return ret

    def train(self):
        generator = self.data_parser.get_dataset_plot_generator()
        steps = len(self.data_parser.graph_files_name) // self.data_parser.batch_size
        if not self.use_validation_set:
            generator_val = None
            val_steps = None
        else:
            generator_val = self.data_parser.get_dataset_plot_val_generator()
            val_steps = len(self.data_parser.val_graph_files_name) // self.data_parser.batch_size

        cb = []
        if self.early_stopping:
            cb.append(EarlyStopping(monitor='loss', mode='min', verbose=1))

        if self.save_best_checkpoint:
            cb.append(ModelCheckpoint(self.get_model_save_name(checkpoint=True), monitor='loss', mode='min',
                                      save_best_only=True))

        if self.neural_network == "leonet":
            from models import LeoNet
            model = LeoNet.leonet_model((1, 224, 224, 3))
            history = model.fit_generator(generator=generator, validation_data=generator_val,
                                          validation_steps=val_steps, epochs=self.epochs, steps_per_epoch=steps,
                                          callbacks=cb)
            self.print_history(history)
            return model

        elif self.neural_network == "leonetv2":
            from models import LeoNetV2
            model = LeoNetV2.LeoNetV2_model((1, 224, 224, 3))
            model.fit_generator(generator=generator, validation_data=generator_val, validation_steps=val_steps,
                                epochs=self.epochs, steps_per_epoch=steps, callbacks=cb)
            return model

    def print_history(self, history):
        import matplotlib.pyplot as plt
        plt.plot(history.history['acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        if self.use_validation_set:
            plt.plot(history.history['val_loss'])
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        if self.use_validation_set:
            plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()


def main(neural_network, type_graph, folders, batch_size, val_percentage):
    data_parser = DataParser("training", folders, type_graph, batch_size=batch_size, val_percentage=val_percentage)
    learner = DataLearner(neural_network, data_parser, use_validation_set=False if val_percentage == 0.0 else True)
    model = learner.train()
    learner.save(model)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Customization options for the data learner")
    parser.add_argument("type_graph", nargs='?', default="melspectrogram", help='Set type of graph')
    parser.add_argument("neural_network", nargs='?', default="leonet", help='Set nn type')
    parser.add_argument("folders", nargs='?', type=list, default=["ff1010bird", "BirdVoxDCASE20k"],
                        help='Set of folders that will be used as the source of the graphs')
    parser.add_argument("batch_size", nargs='?', default=20,
                        help='Batch size of the files used to train the model')
    parser.add_argument("validation_percentage", nargs='?', default=0.0,
                        help='Batch size of the files used to train the model')
    parser.add_argument("epochs", nargs='?', default=1,
                        help='Number of epochs')

    args = parser.parse_args()
    main(args.neural_network, args.type_graph, args.folders, args.batch_size, args.validation_percentage)
