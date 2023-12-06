import numpy as np
import tensorflow as tf

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras import optimizers
from tensorflow.image import resize_with_crop_or_pad, resize
from tensorflow.keras.layers import Conv2D,Add,MaxPooling2D, Dense, BatchNormalization,Input,Flatten, Dropout,GlobalMaxPooling2D,Lambda
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras import layers, models, Input
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.saving import save_model
from tensorflow.keras.regularizers import l1, l2, l1_l2
import pickle


def define_sets_and_params(data_dir, classes=50, input_shape_=(256, 256, 3)):
    """
    generates train, test, val datasets from image directory
    """
    classes = classes
    input_shape_ = input_shape_
    data_dir = data_dir

    dataset_train = image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        color_mode="rgb",
        shuffle=True,
        batch_size=16,
        image_size=(256, 256),
        crop_to_aspect_ratio=False,
        validation_split=0.3,
        subset="training",
        seed=1,
    )

    validation_ds = image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        color_mode="rgb",
        shuffle=True,
        batch_size=16,
        image_size=(256, 256),
        crop_to_aspect_ratio=False,
        validation_split=0.3,
        subset="validation",
        seed=1,
    )

    cut = round((tf.data.experimental.cardinality(validation_ds)).numpy()/2)
    dataset_test = validation_ds.take(cut)
    dataset_val = validation_ds.skip(cut)

    return dataset_train,  dataset_test, dataset_val, classes


def load_model(input_shape_=(256, 256, 3)): #VGG16 base model
    """
    loading vgg16 base model
    """
    model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape_)

    return model


def defining_model(base_model_, classes, input_shape_=(256, 256, 3)):

    #base_model = VGG16(weights="imagenet", include_top=False, input_shape=inputshape)
    base_model = base_model_
    base_model.trainable = False

    for layer in base_model.layers[-4:]:
        layer.trainable = True

    layer11 = layers.Conv2D(512, (3, 3), activation='relu')
    layer111 = layers.MaxPooling2D(2, 2)

    flattening_layer = layers.Flatten()
    #global_layer = layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(x)

    layer2 = layers.Dense(512, activation="relu", kernel_regularizer=l2(0.01))
    layer3 = layers.BatchNormalization()
    layer4 = layers.Dropout(0.3)

    layer5 = layers.Dense(256, activation="relu", kernel_regularizer=l2(0.01))
    layer6 = layers.BatchNormalization()
    layer7 = layers.Dropout(0.2)

    layer8 = layers.Dense(128, activation="relu")
    layer88 = layers.BatchNormalization()
    layer888 = layers.Dropout(0.1)

    output_layer = layers.Dense(classes, activation="softmax")


    inputs = tf.keras.Input(shape=input_shape_)
    # x = data_augmentation(inputs)
    x = preprocess_input(inputs)
    x = base_model(x)
    x = layers.GlobalAveragePooling2D()(x)
    #x = layer11(x)
    #x = layer111(x)
    # x = global_average_layer(x)
    #x = flattening_layer(x)
    # x = tf.keras.layers.Dropout(0.2)(x)
    x = layer2(x)
    x = layer3(x)
    x = layer4(x)
    x = layer5(x)
    x = layer6(x)
    x = layer7(x)
    x = layer8(x)
    x = layer88(x)
    x = layer888(x)
    outputs = output_layer(x)
    model = Model(inputs, outputs)

    return model


def compile_model(model, lr=0.001):


  initial_learning_rate = lr # start with default Adam value 0.001

  lr_schedule = ExponentialDecay(
    #     # Every 5000 iterations, multiply the learning rate by 0.7
  initial_learning_rate, decay_steps = 25, decay_rate = 0.7,
  )

  opt = optimizers.Adam(initial_learning_rate)

  model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy', 'Precision', 'Recall'])
  return model


def train_model(model, dataset_train, dataset_val):
    #model = compile_model(model)

    mc = ModelCheckpoint(
        "weights",
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    es = EarlyStopping(
        monitor = 'val_accuracy',
        mode = 'max',
        patience = 10,
        verbose = 1,
        restore_best_weights = True)

    plat = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.1,
        patience=5,
        verbose=1,
        mode="auto",
        min_delta=0.0001,
        cooldown=0,
        min_lr=0,
    )

    history = model.fit(dataset_train,
                        validation_data=dataset_val,
                        epochs=100,
                        batch_size=16,
                        callbacks=[es, plat, mc])

    return model, history


def saving_model(model, save_dir):
    save_model(model, save_dir)

def saving_history(history, history_save_name):
    with open(history_save_name, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    ## How to open saved history
    # with open('history_save_name', "rb") as file_pi:
    #     history = pickle.load(file_pi)

def saving_ds_test(ds_test, ds_test_save_name):
    with open(ds_test_save_name, 'wb') as file_pi:
        pickle.dump(ds_test, file_pi)

    ## How to open saved history
    # with open('history_save_name', "rb") as file_pi:
    #     history = pickle.load(file_pi)

def run_all(data_dir, save_dir, history_save_name, ds_test_save_name):
    ds_train, ds_test, ds_val, classes_ = define_sets_and_params(data_dir)
    base_model_ = load_model()
    model = defining_model(base_model_=base_model_, classes=classes_)
    model = compile_model(model)
    model, history = train_model(model, ds_train, ds_val)
    saving_model(model, save_dir)
    saving_history(history, history_save_name)
    saving_ds_test(ds_test, ds_test_save_name)

    return model, history


#from birdies_code.train_model_folder.train_model_adjusted import run_all

run_all(data_dir="image_data", save_dir="saved_model", history_save_name="saved_history.pickle", ds_test_save_name="saved_test_dataset")
