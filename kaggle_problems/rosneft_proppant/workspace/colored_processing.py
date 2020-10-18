import copy
from pathlib import Path
from helpers import *
from common import *
import tensorflow as tf
import json

COUNT_ELEM = 2500
BINS_NAME = ['16', '18', '20', '25', '30', '40']
BINS_SIZE = [1.18, 1, 0.85, 0.71, 0.6, 0.5]


def metric(true, predicted):
    return tf.keras.backend.mean(tf.math.reduce_sum((true - predicted) ** 2 / (true + predicted), axis=1))

class Processor:
    model = tf.keras.models.load_model("./models/model_benchmark_colored", compile=False)
    model.compile(
        loss=metric,
        optimizer='rmsprop',
        metrics=['mse']
    )

    def __init__(self, img, img_number):
        self.debug = False
        self.img_number = img_number
        self.img = img

    def with_debug(self, debug_dir):
        self.debug = True
        self.debug_dir = debug_dir
        Path(self.debug_dir).mkdir(exist_ok=True, parents=True)

    def process(self):
        self.img = self.img.astype(np.float)
        bins = self.model.predict(np.array([self.img]))[0]

        if self.debug:
            with open("{}/predicted_bins".format(self.debug_dir), 'w') as f:
                json.dump([float(i) for i in bins], f)

        assert(len(bins) == len(BINS_NAME))
        result_circles = []
        for b, size in zip(bins, BINS_SIZE):
            cnt = int(round(b * COUNT_ELEM))
            result_circles.extend([(None, None, (size + 1e-5) / 2. * COEF)] * cnt)

        return result_circles
