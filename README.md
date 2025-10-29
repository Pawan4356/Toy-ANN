# PyTorch Fashion MNIST Classifier

## Overview

This project implements a simple **Artificial Neural Network (ANN)** using **PyTorch** to classify images from the **Fashion MNIST** dataset. It covers data loading, preprocessing, model creation, training, and evaluation, following CampusX tutorials on ANN and GPU training.

## Features

* Loads and preprocesses small chunk (6000 images) of `FMNIST.csv` (original) dataset (labels + 784 pixel values).
* Normalizes data and splits into 80% train / 20% test.
* Uses custom PyTorch `Dataset` and `DataLoader` for batching.
* Defines a **2-hidden-layer MLP** (128 and 64 neurons, ReLU activations).
* Trains for 100 epochs with **SGD + CrossEntropyLoss**.
* Evaluates and reports test accuracy.

## Model Summary

**Input:** 784 → **Hidden Layers:** 128 → 64 → **Output:** 10 classes.
ReLU activations are used; Softmax is handled by `CrossEntropyLoss`.

## Acknowledgements

  * **Source Tutorial (CPU):** [Building a ANN using PyTorch](http://www.youtube.com/watch?v=6EJaHBJhwDs)
  * **GPU Adaptation:** [Neural Network Training on GPU](http://www.youtube.com/watch?v=CabHrf9eOVs)
