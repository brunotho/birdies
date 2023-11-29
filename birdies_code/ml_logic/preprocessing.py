from tensorflow.keras.utils import to_categorical

def to_cat(y, num_classes=10):
    # full classes = 10982
    return to_categorical(y, num_classes)



def raw_img_to_input_size(image):
