import numpy as np
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from birdies_code.ml_logic.preprocessing import to_cat


classes = len(np.unique(y_train))


def load_model():

    model = VGG16(weights="imagenet", include_top=False, input_shape=(32, 32, 3)) #default 224x224x3

    return model


def set_nontrainable_layers(model):
    model.trainable = False
    return model


def add_last_layers(model):
    '''Take a pre-trained model, set its parameters as non-trainable, and add additional trainable layers on top'''
    base_model = set_nontrainable_layers(model)
    flatten_layer = layers.Flatten()
    dense_layer = layers.Dense(500, activation='relu')
    prediction_layer = layers.Dense(classes, activation='softmax')

    model = models.Sequential([
        base_model,
        flatten_layer,
        dense_layer,
        prediction_layer
    ])

    return model


def build_model():
    model = load_model()
    model = add_last_layers(model)

    opt = optimizers.Adam(learning_rate=1e-4)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model


def vgg_preproc(X):
    return preprocess_input(X)


def fit_vgg(model, X_train, y_train):

    X_train = vgg_preproc(X_train)
    y_train = to_cat(y_train)

    es = EarlyStopping(monitor = 'val_accuracy',
                   mode = 'max',
                   patience = 5,
                   verbose = 1,
                   restore_best_weights = True)

    history = model.fit(X_train, y_train,
                    validation_split=0.2,
                    epochs=50,
                    batch_size=16,
                    callbacks=[es])

    return history, model
