# PyTorch Fashion MNIST Classifier

## Overview

This project is a simple implementation of an Artificial Neural Network (ANN) using PyTorch to classify images from the Fashion MNIST (FMNIST) dataset. The script handles data loading, preprocessing, model definition, training, and evaluation.

This code is based on the "Building a ANN using PyTorch" (Video 7) and "Neural Network Training on GPU" (Video 8) tutorials by CampusX.

## Features
* Loads the FMNIST dataset from a `FMNIST.csv` file.
* Visualizes a sample of 4 images from the dataset using `matplotlib`.
* Splits the data into training (80%) and testing (20%) sets using `scikit-learn`.
* Normalizes pixel values (scales from 0-255 to 0-1) for better model performance.
* Uses a custom PyTorch `Dataset` class to wrap the data.
* Implements `DataLoader` for efficient batching and shuffling.
* Defines a simple multi-layer perceptron (MLP) model with two hidden layers (128 and 64 neurons) and ReLU activations.
* Trains the model for 100 epochs using Stochastic Gradient Descent (SGD) and Cross-Entropy Loss.
* Evaluates the trained model on the test set and prints the final accuracy.

## Requirements

You will need the following Python libraries to run this script:
* `pandas`
* `torch` (PyTorch)
* `scikit-learn`
* `matplotlib`

You can install them using pip:
```bash
pip install pandas torch scikit-learn matplotlib
```

For PyTorch to use an NVIDIA **GPU (via CUDA)**, you can't just run `pip install torch`. You need to install a specific build of PyTorch that is compiled for your specific CUDA version. For CUDA 12.1:-

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```


## Dataset

This script requires the Fashion MNIST dataset to be in a file named **`FMNIST.csv`** in the same directory. The CSV file must be structured with the label in the first column and the 784 pixel values (28x28 flattened) in the subsequent columns.

## Usage

1.  Ensure you have all the required libraries installed.

2.  Place the `FMNIST.csv` file in the same folder as the Python script.

3.  Run the script from your terminal:

    ```bash
    python your_script_name.py
    ```

    *(Replace `your_script_name.py` with the actual name of your file)*

The script will print the average loss for each epoch during training and then output the final accuracy on the test data.

## Model Architecture

The neural network (`MnistNN`) is defined as a sequential model:

1.  **Input Layer:** 784 features (from 28x28 flattened images)
2.  **Hidden Layer 1:** Linear(784, 128) followed by `ReLU()`
3.  **Hidden Layer 2:** Linear(128, 64) followed by `ReLU()`
4.  **Output Layer:** Linear(64, 10) (representing the 10 classes)

*Note: A Softmax activation is not explicitly declared in the model, as it is automatically included within the `nn.CrossEntropyLoss` function during training.*

## Acknowledgements

  * **Source Tutorial (CPU):** [Building a ANN using PyTorch](http://www.youtube.com/watch?v=6EJaHBJhwDs)
  * **GPU Adaptation:** [Neural Network Training on GPU](http://www.youtube.com/watch?v=CabHrf9eOVs)
