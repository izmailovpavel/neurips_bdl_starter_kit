'''Examples of organizer-provided metrics.
You can just replace this code by your own.
Make sure to indicate the name of the function that you chose as metric function
in the file metric.txt. E.g. example_metric, because this file may contain more 
than one function, hence you must specify the name of the function that is your metric.'''

import numpy as np

def agreement(predictions: np.array, reference: np.array):
    """Returns 1 if predictions match and 0 otherwise."""
    return (predictions.argmax(axis=-1) == reference.argmax(axis=-1)).mean()


def total_variation_distance(predictions: np.array, reference: np.array):
    """Returns total variation distance."""
    return np.abs(predictions - reference).sum(axis=-1).mean() / 2.
