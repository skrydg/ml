import tensorflow as tf
import numpy as np

class VanilaAutoEncoder(tf.keras.Model):
    def __init__(self, input_shape, hidden_size):
        super(VanilaAutoEncoder, self).__init__()

        input_shape = [i for i in input_shape]
        input_size = np.prod(input_shape)

        self.model_layers = [
            tf.keras.layers.Reshape([input_size]),
            tf.keras.layers.Dense(units=hidden_size, activation='relu'),
            tf.keras.layers.Dense(units=input_size, activation='sigmoid'),
            tf.keras.layers.Reshape(input_shape)
        ]

    def call(self, x, *args, **kwargs):
        for l in self.model_layers:
            x = l(x, *args, **kwargs)

        return x

class VanilaDeepAutoEncoder(tf.keras.Model):
    def __init__(self, input_shape, hidden_size):
        super(VanilaDeepAutoEncoder, self).__init__()

        input_shape = [i for i in input_shape]
        input_size = np.prod(input_shape)

        midle_size = int((input_size * hidden_size) ** 0.5)

        self.model_layers = [
            tf.keras.layers.Reshape([input_size]),

            tf.keras.layers.Dense(units=midle_size, activation='relu'),

            tf.keras.layers.Dense(units=hidden_size, activation='relu'),

            tf.keras.layers.Dense(units=midle_size, activation='relu'),

            tf.keras.layers.Dense(units=input_size, activation='sigmoid'),

            tf.keras.layers.Reshape(input_shape)
        ]

    def call(self, x, *args, **kwargs):
        for l in self.model_layers:
            x = l(x, *args, **kwargs)

        return x

def up_to_power_of_two(x):
    i = 1
    while i < x:
        i *= 2
    return i

class CNNAutoEncoder(tf.keras.Model):
    # input_shape: (None, height, width, channel_size)
    #
    def __init__(self, input_shape, hidden_size):
        super(CNNAutoEncoder, self).__init__()
        (self.height, self.width, channel_size) = input_shape

        self.up_height = up_to_power_of_two(self.height)
        self.up_width = up_to_power_of_two(self.width)

        self.filter_size1 = 8
        self.filter_size2 = 4

        self.model_layers = [
            # padding = ((top_pad, bottom_pad), (left_pad, right_pad))
            tf.keras.layers.ZeroPadding2D(padding=((0, self.up_height - self.height), (0, self.up_width - self.width)), data_format='channels_last'),

            # (None, 2^x, 2^y, channel_size)
            tf.keras.layers.Conv2D(self.filter_size1, kernel_size=(4, 4), padding='same', activation='relu', data_format="channels_last"),
            # (None, 2^x, 2^y, self.filter_size1)
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), data_format="channels_last"),

            # (None, 2^(x - 1), 2^(y - 1), self.filter_size1)
            tf.keras.layers.Conv2D(self.filter_size2, kernel_size=(4, 4), padding='same', activation='relu',
                                   data_format="channels_last"),
            # (None, 2^(x - 1), 2^(y - 1), self.filter_size1)
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), data_format="channels_last"),

            # (None, 2^(x - 2), 2^(y - 2), self.filter_size1)
            tf.keras.layers.Reshape([self.up_height * self.up_height * self.filter_size2 // 16 ]),

            # (None, 2^(x - 2) *  2^(y - 2) * self.filter_size2)
            tf.keras.layers.Dense(units=hidden_size, activation='relu'),


            # (None, hidden_size)
            tf.keras.layers.Dense(units=self.up_height * self.up_height * self.filter_size2 // 16, activation='relu'),
            # (None, 2^(x - 2) *  2^(y - 2) * filter_size2)
            tf.keras.layers.Reshape([self.up_height // (2 ** 2), self.up_height // (2 ** 2), self.filter_size2]),
            # (None, 2^(x - 2), 2^(y - 2), self.filter_size2)

            tf.keras.layers.UpSampling2D(size=(2, 2), data_format="channels_last"),
            # (None, 2^(x - 1), 2^(y - 1), self.filter_size2)

            tf.keras.layers.Conv2D(self.filter_size1, kernel_size=(4, 4), padding='same', activation='relu',
                                   data_format="channels_last"),
            # (None, 2^(x - 1), 2^(y - 1), self.filter_size1)

            tf.keras.layers.UpSampling2D(size=(2, 2), data_format="channels_last"),
            # (None, 2^x, 2^y, self.filter_size1)

            tf.keras.layers.Conv2D(channel_size, kernel_size=(4, 4), padding='same', activation='relu',
                                   data_format="channels_last"),
            # (None, 2^x, 2^y, channel_size)
        ]

    def call(self, x, *args, **kwargs):
        for l in self.model_layers:
            x = l(x, *args, **kwargs)

        return x[:, 0:self.height, 0:self.width, :]
