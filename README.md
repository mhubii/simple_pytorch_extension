# Simple C++ Extension for PyTorch
Some simple pytorch extensions. Each extension can be built as described below.

## Gaussian Extension
Given the mean, this extension computes a 2D gaussian distribution with diagonal covariance. On backpropagation, it determines the mean that minimizes the difference between a reference distribution and the output of the layer. The gradient for this operation is given in [backward](https://github.com/mhubii/simple_pytorch_extension/blob/bd4b7ad24c02cde5e6665c1303da4443f256b5cd/gaussian/gaussian_extension.cpp#L19).

<br>
<figure>
  <p align="center"><img src="gaussian/img/output_gif.gif" width="50%" height="50%"></p>
  <figcaption>Fig. 1: Evolution of learned average for gaussian layer.</figcaption>
</figure>
<br><br>

## Build
```shell
# for example
cd gaussian
python setup.py install
```

## Run
```shell
# for example
cd gaussian
python main.py
```
