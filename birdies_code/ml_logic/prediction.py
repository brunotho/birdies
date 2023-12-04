import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.image import resize #resize_with_crop_or_pad,
from tensorflow.keras.applications.vgg16 import preprocess_input


def load_model_():
    model = load_model("dirty_model_5")
    return model

def preproc_image(X_pred):
    X_pred = tf.convert_to_tensor(X_pred)
    X_pred = resize(X_pred, (256,256))
    X_pred = preprocess_input(X_pred)
    X_pred = tf.expand_dims(X_pred, axis=0)
    return X_pred

def model_predict(model, X_pred):
    y_pred = model.predict(X_pred)
    return y_pred

def dirty_to_output(y_pred):
    """
    converts y_pred array to dictionary of top 3 likely birds: {(species_num, proba),(),()}
    """
    probabilities = np.ndarray.tolist(y_pred)[0]
    class_names = ['02091',
    '10474',
    '10491',
    '10512',
    '10588',
    '10705',
    '10828',
    '10848',
    '10883',
    '10921']

    temp = [(key, value)
            for index, (key, value) in enumerate(zip(class_names, probabilities))]

    dirty_dict = dict(temp)

    key_1 = sorted(dirty_dict, key=dirty_dict.get)[-1]
    value_1 = dirty_dict[key_1]
    tuple_1 = (key_1, round(value_1, 2))
    key_2 = sorted(dirty_dict, key=dirty_dict.get)[-2]
    value_2 = dirty_dict[key_2]
    tuple_2 = (key_2, round(value_2, 2))
    key_3 = sorted(dirty_dict, key=dirty_dict.get)[-3]
    value_3 = dirty_dict[key_3]
    tuple_3 = (key_3, round(value_3, 2))

    prediction = dict([("pred_1", tuple_1), ("pred_2", tuple_2), ("pred_3", tuple_3)])

    return prediction

def prediction(model, X_pred):
    X_pred = preproc_image(X_pred)
    y_pred = model_predict(model, X_pred)
    prediction = dirty_to_output(y_pred)
    return prediction
