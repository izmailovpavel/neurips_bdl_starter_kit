# IMDB

IMDB is a binary classification, sentiment dataset. Please refer to
[the website](https://ai.stanford.edu/~amaas/data/sentiment/) for more information.

## Obtaining the dataset

We have uploaded a copy of the dataset to a GCS bucket. To download a copy, use:

```bash
$ gsutil -m cp -r gs://neurips2021_bdl_competition/imdb_*.csv
```

This will download four files:
```
imdb_train_x.csv
imdb_train_y.csv
imdb_test_x.csv
imdb_test_y.csv
```

Each row in the files is a point in the dataset, with the `_x` files containing
features and the `_y` files containing labels.

We can load the files using NumPy, for example:

```python
import numpy as np
x_train = np.loadtxt("imdb_train_x.csv", dtype=np.int32)
y_train = np.loadtxt("imdb_train_y.csv", dtype=np.int32)
x_test = np.loadtxt("imdb_test_x.csv", dtype=np.int32)
y_test = np.loadtxt("imdb_test_y.csv", dtype=np.int32)
```


## HMC predictions

The HMC predictions are contained in `probs.csv`, which can be loaded via NumPy as well.

```python
import numpy as np
predictions = np.loadtxt("probs.csv", dtype=np.float32)
```

`predictions` will be an N x C (number of data points by number of classes)
array, where each entry corresponds to probability of classification.
