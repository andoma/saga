
# Saga - A small C++ framework for deep learning.

This project is still very much work in progress.

# Features

* Relies exclusively on NVIDIA's cuDNN and cuda libraries. Ie, this does not work without an NVIDIA GPU.

* FP32 and FP16 inference and training mode.

* Convolution, BatchNorm, Fully-Connected, Dropout -layers

* Adam optimizer with mixed precision training.

* Can load (some) [ONNX](https://onnx.ai) models

# TODO

* Scaling of gradients when using mixed precision learning.

* Load more ONNX models



# Other

## Why?

I wanted to get a better understanding of how deep neural nets work.

## What about the name?

The etymology of the name Sága is generally held to be connected to the Old Norse verb sjá, meaning "to see" (from Proto-Germanic *sehwan).

https://en.wikipedia.org/wiki/S%C3%A1ga_and_S%C3%B6kkvabekkr

