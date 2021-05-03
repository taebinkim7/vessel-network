import tensorflow as tf
from tensorflow.keras import layers, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate

def get_unet(patch_size, dropout_rate=0.2):
    input_size = (patch_size, patch_size, 3)

    inputs = Input(input_size)
    c1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    d1 = Dropout(dropout_rate)(c1)
    c1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(d1)

    p2 = MaxPooling2D(pool_size=(2, 2))(c1)
    c2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(p2)
    d2 = Dropout(dropout_rate)(c2)
    c2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(d2)

    p3 = MaxPooling2D(pool_size=(2, 2))(c2)
    c3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(p3)
    d3 = Dropout(dropout_rate)(c3)
    c3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(d3)

    p4 = MaxPooling2D(pool_size=(2, 2))(c3)
    c4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(p4)
    d4 = Dropout(dropout_rate)(c4)
    c4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(d4)

    u5 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(c4))
    m5 = concatenate([c3, u5], axis=3)
    c5 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(m5)
    d5 = Dropout(dropout_rate)(c5)
    c5 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(d5)

    u6 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(c5))
    m6 = concatenate([c2, u6], axis=3)
    c6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(m6)
    d6 = Dropout(dropout_rate)(c6)
    c6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(d6)

    u7 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(c6))
    m7 = concatenate([c1, u7], axis=3)
    c7 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(m7)
    d7 = Dropout(dropout_rate)(c7)
    c7 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(d7)
    c7 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c7)

    outputs = Conv2D(1, 1, activation='sigmoid')(c7)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model
