# Evaluation Phase Data and Scoring Script

We provide the evaluation phase HMC prediction data [here](https://github.com/izmailovpavel/neurips_bdl_starter_kit/tree/main/eval-phase/eval_data).
The scoring script for submissions is available [here](https://github.com/izmailovpavel/neurips_bdl_starter_kit/blob/main/eval-phase/submission-score-evaluation.ipynb).


If you use the data, please consider citing

```bibtex
@inproceedings{izmailov2021bayesian,
  title={What are Bayesian neural network posteriors really like?},
  author={Izmailov, Pavel and Vikram, Sharad and Hoffman, Matthew D and Wilson, Andrew Gordon Gordon},
  booktitle={International conference on machine learning},
  pages={4629--4640},
  year={2021},
  organization={PMLR}
}
```

and

```bibtex
@inproceedings{wilson2022evaluating,
  title={Evaluating approximate inference in Bayesian deep learning},
  author={Wilson, Andrew Gordon and Izmailov, Pavel and Hoffman, Matthew D and Gal, Yarin and Li, Yingzhen and Pradier, Melanie F and Vikram, Sharad and Foong, Andrew and Lotfi, Sanae and Farquhar, Sebastian},
  booktitle={NeurIPS 2021 Competitions and Demonstrations Track},
  pages={113--124},
  year={2022},
  organization={PMLR}
}
```

## Input data for CIFAR-10

For CIFAR-10, [@PaulScemama](https://github.com/PaulScemama) kindly provided the data in a more convenient preprocessed format (see [discussion here](https://github.com/izmailovpavel/neurips_bdl_starter_kit/issues/4)), an `.npz` file with the following files:

- `'x_train'`: 50k training images for Cifar10.
- `'y_train'`: 50k training labels for Cifar10.
- `'x_test_v1'`: 10k test images from the Original Cifar10
- `'x_test_v2'`: 10k test images from a corrupted version of Cifar10 (`cifar10_corrupted/gaussian_noise_2`)
- `'x_test_v3'`: 10k test images from a corrupted version of Cifar10 (`cifar10_corrupted/brightness_3`)
- `'x_test_v4'`: 10k test images from a corrupted version of Cifar10 (`cifar10_corrupted/pixelate_4`)
- `'x_test_v5'`: 10k test images from a corrupted version of Cifar10 (`cifar10_corrupted/zoom_blur_5`)
- `'y_test'`: 10k test labels for the test images of Cifar10 (all the same for each version of images).


The data is available [here](https://drive.google.com/file/d/1buxwqOaXkCo26ZVhOIosB4lNBO6FTmRS/view?usp=sharing).
