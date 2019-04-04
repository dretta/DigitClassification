try:
    import tensorflow as tf
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.layers import Dense, Dropout, Flatten
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, concatenate
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import RMSprop
    from tensorflow.keras import backend as K
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.losses import categorical_crossentropy
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import ReduceLROnPlateau
    from tensorflow.contrib.training.python.training.hparam import HParams
except Exception as importErr:
    print("-E- Module import failed")
    raise importErr

import os
import sys
import json

NUM_CLASSES = 10


class Mnist:
    def __init__(self, args):
        self.hyperParams = None

        assert len(args) <= 1, "usage: python Mnist.py <Path to Hyperparameters File>"
        if args:
            self.setHyperParameters(args[0])

        self.batchSize = self.hyperParams.get('batchSize', 128)
        self.epochs = self.hyperParams.get('epochs', 15)

        # Please use 28 as the lengths for the inference server
        self.rowSize = self.hyperParams.get('rows', 28)
        self.colSize = self.hyperParams.get('columns', 28)

        # Root Means Squared Optimizer Configuration
        self.RMSConfig = {
            'lr': self.hyperParams.get('lr', 0.001),
            'rho': self.hyperParams.get('rho', 0.9),
            'epsilon': self.hyperParams.get('epsilon', 1e-08),
            'decay': self.hyperParams.get('decay', 0.0)
        }

        # Reduce Learning Rate On Plateau Configuration
        self.RLRPConfig = {
            'monitor': self.hyperParams.get('monitor', "val_acc"),
            'patience': self.hyperParams.get('patience', 3),
            'verbose': self.hyperParams.get('verbose', 1),
            'factor': self.hyperParams.get('factor', 0.5),
            'min_lr': self.hyperParams.get('min_lr', 0.00001)
        }

        self.session = K.get_session()

        # Retrieve the Mnist data and clean it up
        self.inputShape = None
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        self.prepareData()

        self.runModel()

    def setHyperParameters(self, hFile):
        """!
        Parses through the Hyperparameters files to the HParam object

        @param hFile: The path of the Hyperparameters file
        """
        assert os.path.exists(hFile), "-E- Cannot locate file path {}".format(hFile)
        assert hFile.endswith(".json"), " -E- \"{}\" is not a JSON file".format(hFile)
        try:
            with open(hFile) as f:
                jsonObj = json.load(f)
        except Exception as e:
            print("-E- Cannot parse JSON file for hyperparameters.")
            raise e

        self.hyperParams = HParams()
        for key in jsonObj:
            self.hyperParams.add_hparam(key, jsonObj[key])

    def prepareData(self):
        """!
        Converts the Mnist data for proper input to the neural network
        """
        # Define the image (input) shapes
        self.x_train = self.x_train.reshape(self.x_train.shape[0], self.rowSize, self.colSize, 1)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], self.rowSize, self.colSize, 1)
        self.inputShape = (self.rowSize, self.colSize, 1)

        # Modify image values
        self.x_train = self.x_train.astype("float32")
        self.x_test = self.x_test.astype("float32")
        self.x_train /= 255
        self.x_test /= 255

        print("-I- x_train shape:", self.x_train.shape)
        print("-I-", self.x_train.shape[0], "train samples")
        print("-I-", self.x_test.shape[0], "test samples")

        # Turn the expected results into categories
        self.y_train = to_categorical(self.y_train, NUM_CLASSES)
        self.y_test = to_categorical(self.y_test, NUM_CLASSES)

    def runModel(self):
        """!
        Performs all of the operations used for the model
        """
        # Generate and compile the model
        model = self.createModel()
        model.compile(loss=categorical_crossentropy,
                      optimizer=RMSprop(**self.RMSConfig),
                      metrics=["accuracy"])

        # Generate more images to be trained on the model
        datagen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=10,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=False,
            vertical_flip=False)
        datagen.fit(self.x_train)

        # Train the model
        model.fit_generator(datagen.flow(self.x_train, self.y_train, batch_size=self.batchSize),
                            epochs=self.epochs, validation_data=(self.x_test, self.y_test),
                            verbose=1, steps_per_epoch=self.x_train.shape[0] // self.batchSize,
                            callbacks=[ReduceLROnPlateau(**self.RLRPConfig)])

        # Test the model
        score = model.evaluate(self.x_test, self.y_test, verbose=0)
        print("-I- Test loss:", score[0])
        print("-I- Test accuracy:", score[1])

        # Save the model
        frozen_graph = self.freeze_session(output_names=[out.op.name for out in model.outputs])
        tf.train.write_graph(frozen_graph, os.getcwd(), "Mnist.pb", as_text=False)

    def createModel(self):
        """!
        Make the Neural Network that defines the Model
        @return the keras-based model
        """

        input_layer = Input(shape=self.inputShape, name="input_tensor")

        conv1 = Conv2D(32, (1, 1), activation="relu")(input_layer)
        pool1 = MaxPooling2D(2, 2)(conv1)

        conv2_1 = Conv2D(64, (1, 1), activation="relu", padding="same")(pool1)
        pool2_1 = MaxPooling2D(2, 2)(conv2_1)
        drop2_1 = Dropout(0.25)(pool2_1)

        conv2_2 = Conv2D(64, (1, 1), activation="relu", padding="same")(pool1)
        pool2_2 = MaxPooling2D(2, 2)(conv2_2)
        drop2_2 = Dropout(0.25)(pool2_2)

        conv3_1 = Conv2D(256, (1, 1), activation="relu", padding="same")(drop2_1)
        conv3_2 = Conv2D(256, (1, 1), activation="relu", padding="same")(drop2_2)

        merged = concatenate([conv3_1, conv3_2], axis=-1)
        merged = Dropout(0.5)(merged)
        merged = Flatten()(merged)

        fc1 = Dense(1000, activation="relu")(merged)
        fc2 = Dense(500, activation="relu")(fc1)
        out = Dense(10, activation="softmax", name="output_tensor")(fc2)

        return Model(input_layer, out)

    def freeze_session(self, keep_var_names=None, output_names=None, clear_devices=True):
        """!
        Freezes the state of a session into a pruned computation graph.

        Creates a new computation graph where variable nodes are replaced by
        constants taking their current value in the session. The new graph will be
        pruned so subgraphs that are not necessary to compute the requested
        outputs are removed.
        @param keep_var_names A list of variable names that should not be frozen,
                              or None to freeze all the variables in the graph.
        @param output_names Names of the relevant graph outputs.
        @param clear_devices Remove the device directives from the graph for better portability.
        @return The frozen graph definition.
        """
        graph = self.session.graph
        with graph.as_default():
            freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
            output_names = output_names or []
            output_names += [v.op.name for v in tf.global_variables()]
            input_graph_def = graph.as_graph_def()
            if clear_devices:
                for node in input_graph_def.node:
                    node.device = ""
            return convert_variables_to_constants(self.session, input_graph_def, output_names, freeze_var_names)


if __name__ == "__main__":
    Mnist(sys.argv[1:])
