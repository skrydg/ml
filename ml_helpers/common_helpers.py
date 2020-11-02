import os
import sys
while not os.getcwd().endswith('ml'):
    os.chdir('..')
sys.path.insert(0, os.getcwd())

import matplotlib.pyplot as plt
import numpy as np

def display_history_metrics(history, model_name):
    try:
        history = history.history
    except:
        pass

    keys = [i[len("val_"):] for i in history.keys() if (i.startswith("val_"))]

    assert(len(keys) > 0, "keys must not be empty")
    epochs = len(history[keys[0]])

    fig, ax = plt.subplots(len(keys), 1, figsize=(12, 5 * len(keys)))
    if not isinstance(ax, list):
        ax = [ax]

    for i, metric in zip(ax, keys):
        i.plot(history[metric], color='b', label="Training {} for {}".format(metric, model_name))
        i.plot(history["val_{}".format(metric)], color='r', label="Validation {} for {}".format(metric, model_name))
        i.set_xticks(np.arange(1, epochs, 1))
        i.legend(loc="best")
    plt.show()