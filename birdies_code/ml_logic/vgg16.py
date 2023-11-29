from tensorflow.keras.applications.vgg16 import VGG16

def load_vgg16():
    model = VGG16(weights="imagenet", include_top=False, input_shape=X_train[0].shape)

    return model
