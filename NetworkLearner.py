import argparse

from keras.callbacks import EarlyStopping, ModelCheckpoint

from DataParser import DataParser


class DataLearner:
    def __init__(self, neural_network, data_parser, epochs=30, early_stopping=True, save_best_checkpoint=True,
                 use_validation_set=False, output=None):
        self.neural_network = neural_network
        self.epochs = epochs
        self.data_parser = data_parser
        self.early_stopping = early_stopping
        self.save_best_checkpoint = save_best_checkpoint
        self.use_validation_set = use_validation_set
        self.output = output

    def save(self, model):
        model.save(self.get_model_save_name())

    def get_model_save_name(self, checkpoint=False):
        if self.output is None:
            ret = self.neural_network + "_" + self.data_parser.graph_type + ".h5"
        else:
            ret = self.output
        if not ret.endswith(".h5"):
            ret += ".h5"

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
        if self.early_stopping and self.use_validation_set:
            cb.append(EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5))

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
            history = model.fit_generator(generator=generator, validation_data=generator_val,
                                          validation_steps=val_steps,
                                          epochs=self.epochs, steps_per_epoch=steps, callbacks=cb)
            self.print_history(history)
            return model

    def print_history(self, history):
        import matplotlib.pyplot as plt
        name = self.get_model_save_name()
        if self.output is not None:
            name = self.get_model_save_name()
        plt.plot(history.history['acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        if self.use_validation_set:
            plt.plot(history.history['val_loss'])
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(name + "_acc.png")
        plt.close()
        # summarize history for loss
        plt.plot(history.history['loss'])
        if self.use_validation_set:
            plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(name + "_loss.png")
        plt.close()


def main(neural_network, type_graph, folders, epochs, batch_size, val_percentage, output, early_stopping):
    data_parser = DataParser("training", folders, type_graph, batch_size=batch_size, val_percentage=val_percentage)
    learner = DataLearner(neural_network, data_parser, epochs=epochs,
                          use_validation_set=False if val_percentage == 0.0 else True, output=output, early_stopping=early_stopping)
    model = learner.train()
    learner.save(model)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Customization options for the data learner")
    parser.add_argument("type_graph", nargs='?', default="spectrogram", help='Set type of graph')
    parser.add_argument("neural_network", nargs='?', default="leonet", help='Set nn type')
    parser.add_argument("folders", nargs='?', default="BirdVoxDCASE20k,ff1010bird",
                        help='Set of folders that will be used as the source of the graphs')
    parser.add_argument("batch_size", nargs='?', type=int, default=20,
                        help='Batch size of the files used to train the model')
    parser.add_argument("validation_percentage", nargs='?', type=float, default=0.15,
                        help='Batch size of the files used to train the model')
    parser.add_argument("epochs", nargs='?', type=int, default=1,
                        help='Number of epochs')
    parser.add_argument("output", nargs='?', default="test2",
                        help='Output filename')
    parser.add_argument("early_stopping", nargs='?', default="False", type=bool,
                        help='Output filename')

    args = parser.parse_args()
    print(args)
    main(args.neural_network, args.type_graph, [item.strip() for item in args.folders.strip().split(',')], args.epochs, args.batch_size, args.validation_percentage,
         args.output, args.early_stopping)
