from tensorflow.keras.utils import to_categorical

def ohe(y, num_of_classes):
    y = to_categorical(y, num_classes=num_of_classes)
    return y
