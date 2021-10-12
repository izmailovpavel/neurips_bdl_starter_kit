'''Examples of organizer-provided metrics.
You can just replace this code by your own.
Make sure to indicate the name of the function that you chose as metric function
in the file metric.txt. E.g. example_metric, because this file may contain more 
than one function, hence you must specify the name of the function that is your metric.'''

import numpy as np
import scipy

def agreement(predictions: np.array, reference: np.array):
    """Returns 1 if predictions match and 0 otherwise."""
    return (predictions.argmax(axis=-1) == reference.argmax(axis=-1)).mean()


def total_variation_distance(predictions: np.array, reference: np.array):
    """Returns total variation distance."""
    return np.abs(predictions - reference).sum(axis=-1).mean() / 2.


def w2_distance(predictions: np.array, reference: np.array):
    """Returns W-2 distance """
  NUM_SAMPLES_REQUIRED = 1000
  assert predictions.shape[0] == reference.shape[0], "wrong predictions shape"
  assert predictions.shape[1] == NUM_SAMPLES_REQUIRED, "wrong number of samples"
  return -np.mean([scipy.stats.wasserstein_distance(pred, ref) for 
                   pred, ref in zip(predictions, reference)])
