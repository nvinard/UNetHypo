import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, Conv1D, Conv2DTranspose, Dense, Dropout, Flatten, MaxPooling2D, MaxPooling1D, Softmax, Activation)
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Sequential


# Function to produce downsampling layers
def downsample(x, filters, strides, size=(3,3), apply_batchnorm=True):
    initializer = 'glorot_uniform'

    x = tf.keras.layers.Conv2D(
      filters, kernel_size=size, strides=strides, padding='same',
      kernel_initializer=initializer, activation=tf.keras.layers.ReLU(), use_bias=False)(x)
    if apply_batchnorm:
        x = tf.keras.layers.BatchNormalization()(x)

    return x

# Function to produce upsampling layers
def upsample(x, filters, strides, size=(3,3)):

    initializer = 'glorot_uniform'
    x = tf.keras.layers.Conv2DTranspose(
        filters, kernel_size=size, strides=strides, padding='same',
        kernel_initializer=initializer, activation=tf.keras.layers.ReLU(), use_bias=False)(x)#, kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)

    x = tf.keras.layers.BatchNormalization()(x)

    return x

def Generator():
    filter_down = [32, 32, 32, 32, 32, 64, 64, 64, 64, 128, 128, 128, 128, 256, 256]
    filter_up = [256, 256, 128, 128, 128, 128, 64, 64, 40, 40, 40, 40]
    dropout = [True, True, True, True, True, True, False, False, False, False, False, False]
    conc = [True, True, True, True, True, True, True, True, False, False, False, False]
    strides_down = [(1,1),(2,1),(1,1),(2,1),(1,1),(2,1),(1,1),(2,2),(1,1),
                    (2,2),(1,1),(2,2),(1,1),(2,2),(1,1)]
    strides_up = [(1,1), (1,1),(2,2),(1,1),(2,2),(1,1),(2,2),(1,1),
                  (1,1),(1,1),(1,1),(1,1)]

    initializer = 'glorot_uniform'

    # final layer to reduce the channel to 1
    last = tf.keras.layers.Conv2DTranspose(40, (3,3), strides=(1,1), padding='same', kernel_initializer=initializer, activation='sigmoid', use_bias=False)

    concat = tf.keras.layers.Concatenate()
    inputs = tf.keras.layers.Input(shape=[1401,96,1])
    x = inputs

    # Downsampling through the model
    skips = []
    count = 0
    for down in strides_down:
        x = downsample(x, filter_down[count], down, apply_batchnorm=True)
        if count > 6:
            skips.append(x)
        count+=1

    skips = reversed(skips)

    # Upsampling and establishing the skip connections
    c=0
    for up, skip in zip(strides_up, skips):
        x = upsample(x, filter_up[c], up)
        x = tf.keras.layers.Concatenate()([x, skip])
        #x = tf.keras.layers.SpatialDropout2D(0.5)(x)
        c+=1

    for up in strides_up[c:]:
        x = upsample(x,filter_up[c],up)
        c+=1

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)
