
# Saga - A small C++ framework for deep learning.

This project is still very much work in progress.

# Features

* Relies exclusively on NVIDIA's cuDNN and cuda libraries. Ie, this does not work without an NVIDIA GPU.

* Fully flexible tensor layouts (ie both NCHW and NHWC tensor are fully supported)

* FP32 and FP16 inference and training mode.

* Adam optimizer with mixed precision training and dynamic gradient scaling.

* Data augmentation using 2d affine transforms (scaling, rotation, translation)

* Supported layers:
  * Activation
  * Affine transformation
  * Batchnorm
  * Category Classifier
  * Concat
  * Convolution
  * Dropout
  * Elementwise math operations
  * Fully conneceted
  * Pooling
  * Softmax
  * Sum

* Node optimizations:
  * Concat is transformed to strided tensors
  * Element-wise sum is transformed to outputs with GEMM beta set to 1

* Double buffered tensors at edge of graph
  Allows updating next mini-batch and reading out values from previous
  mini-batch while GPU is process current mini-batch. Ensuring 100% GPU
  utilization


* Can load (some) [ONNX](https://onnx.ai) models

# Other

## Why?

I wanted to get a better understanding of how deep neural nets work.

## What about the name?

The etymology of the name Sága is generally held to be connected to the Old Norse verb sjá, meaning "to see" (from Proto-Germanic *sehwan).

https://en.wikipedia.org/wiki/S%C3%A1ga_and_S%C3%B6kkvabekkr

