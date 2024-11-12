# MNIST classifier in C++

This project is a C++ implementation of a simple MNIST classifier.

## Dataset
The MNIST dataset used in this project is downloaded from [GTDLBench](https://git-disl.github.io/GTDLBench/datasets/mnist_datasets/).

## Dependencies
No external dependencies are required. This project relies solely on standard C++ libraries.

## Libraries
The following standard libraries are used:
- `Eigen/Dense`
- `algorithm`
- `cmath`
- `fstream`
- `iostream`
- `limits`
- `random`
- `vector`

## Setup
No speciall configuration should be necessary.

## Usage
1. Place the following MNIST dataset files in the root directory:
    - `t10k-images-idx3-ubyte`
    - `t10k-labels-idx1-ubyte`
    - `train-images-idx3-ubyte`
    - `train-labels-idx1-ubyte`
2. Compile and run `main.cpp`.

```bash
g++ main.cpp -o mnist_classifier && ./mnist_classifier
```
