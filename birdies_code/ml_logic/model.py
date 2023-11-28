#from keras.preprocessing.image import ImageDataGenerator#

#from tensorflow import keras
from keras import Model, layers, Sequential
from keras.layers import Dense
from tensorflow.keras.layers.experimental.preprocessing import Resizing

def initialize_model(input_shape: tuple) -> Model:
    """
    Initialize the Neural Network with random weights
    """

    model = Sequential()

    model.add(Resizing(224, 224, interpolation='bilinear', input_shape=(None, None, 3)))

    model.add(layers.Conv2D(32, (3, 3), activation='relu')) #, input_shape=(32, 32, 3)
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(Dense(11000, activation='softmax'))
    print("✅ Model initialized")

    return model



def compile_model(model: Model, learning_rate=0.0005) -> Model:
    """
    Compile the Neural Network
    """
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=["mae"])

    print("✅ Model compiled")

    return model
