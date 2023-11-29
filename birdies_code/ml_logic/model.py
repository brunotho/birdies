#from keras.preprocessing.image import ImageDataGenerator#

from keras import Model, layers, Sequential, optimizers
from keras.layers import Dense, Flatten
from tensorflow.keras.layers.experimental.preprocessing import Resizing
import numpy as np
#from colorama import Fore, Style
from typing import Tuple
from birdies.ml_logic.encoders import ohe

def initialize_model(num_of_classes) -> Model:
    """
    Initialize the Neural Network with random weights
    """

    model = Sequential()

    model.add(Resizing(224, 224, interpolation='bilinear', input_shape=(None, None, 3)))

    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(Flatten())

    model.add(Dense(num_of_classes, activation='softmax'))

    return model



def compile_model(model: Model, learning_rate_=0.0005) -> Model:
    """
    Compile the Neural Network
    """
    #optimizer = optimizers.Adam(learning_rate=learning_rate)
    #model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=["mae"])
    optimizer_ = optimizers.Adam(learning_rate=learning_rate_)
    metrics_ = ["accuracy"]

    model.compile(loss="categorical_crossentropy", optimizer=optimizer_, metrics=metrics_)

    print("✅ Model compiled")

    return model


def train_model(model, X, y) -> Tuple[Model, dict]:
    """
    trains model; returns (model, history)
    """

    from tensorflow.keras.callbacks import EarlyStopping

    es = EarlyStopping(patience = 2)

    history = model.fit(X,
                    y,
                    validation_split = 0.3,
                    batch_size = 32,
                    epochs = 5,
                    callbacks = [es],
                    verbose = 1)

    print(f"✅ Model trained on {len(X)} rows with min val MAE: {round(np.min(history.history['val_mae']), 2)}")

    return model, history
