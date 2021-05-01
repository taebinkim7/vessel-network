import tensorflow as tf
from tensorflow.keras import layers, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate

def get_unet(input_size=(256, 256, 1), rate=0.5, pretrained_weights=None):
    inputs = Input(input_size)
    c1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs) # (256, 256, c)
    d1 = Dropout(rate)(c1)
    c1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(d1)

    p2 = MaxPooling2D(pool_size=(2, 2))(c1) # (128, 128, c)
    c2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(p2)
    d2 = Dropout(rate)(c2)
    c2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(d2)

    p3 = MaxPooling2D(pool_size=(2, 2))(c2) # (64, 64, c)
    c3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(p3)
    d3 = Dropout(rate)(c3)
    c3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(d3)

    p4 = MaxPooling2D(pool_size=(2, 2))(c3) # (32, 32, c)
    c4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(p4)
    d4 = Dropout(rate)(c4)
    c4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(d4)

    p5 = MaxPooling2D(pool_size=(2, 2))(c4) # (16, 16, c)
    c5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(p5)
    d5 = Dropout(rate)(c5)
    c5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(d5)

    u6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(c5)) # (32, 32, c)
    m6 = concatenate([c4, u6], axis=3)
    c6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(m6)
    d6 = Dropout(rate)(c6)
    c6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(d6)

    u7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(c6)) # (64, 64, c)
    m7 = concatenate([c3, u7], axis=3)
    c7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(m7)
    d7 = Dropout(rate)(c7)
    c7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(d7)

    u8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(c7)) # (128, 128, c)
    m8 = concatenate([c2, u8], axis=3)
    c8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(m8)
    d8 = Dropout(rate)(c8)
    c8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(d8)

    u9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(c8)) # (256, 256, c)
    m9 = concatenate([c1, u9], axis=3)
    c9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(m9)
    d9 = Dropout(rate)(c9)
    c9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c9)
    c9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c9)

    outputs = Conv2D(1, 1, activation='sigmoid')(c9)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model

# # plot graph of model
# model = get_unet()
# tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=False) # None represents the batch size
