import os
import sys
while not os.getcwd().endswith('ml'):
    os.chdir('..')
sys.path.insert(0, os.getcwd())

import matplotlib.pyplot as plt
import numpy as np

def display_history_metrics(history, model_name):
    keys = [i for i in history.history.keys() if (not i.startswith("val_"))]
    assert(len(keys) > 0, "keys must not be empty")
    epochs = len(history.history[keys[0]])

    fig, ax = plt.subplots(len(keys), 1, figsize=(12, 5 * len(keys)))
    for i, metric in zip(ax, keys):
        i.plot(history.history[metric], color='b', label="Training {} for {}".format(metric, model_name))
        i.plot(history.history["val_{}".format(metric)], color='r', label="Validation {} for {}".format(metric, model_name))
        i.set_xticks(np.arange(1, epochs, 1))
        i.legend(loc="best")