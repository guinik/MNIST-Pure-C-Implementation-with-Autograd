# MNIST C++ Implementation

This project is a C++ implementation of a simple neural network for the MNIST dataset.
It is inspired by [this C implementation](https://github.com/Magicalbat/videos/blob/main/machine-learning/main.c) and translated into modern C++ with autograd-like support.

---
# Purpose of the Project

This project was created as a learning exercise to better understand how computational graphs work in machine learning frameworks. The goal was to implement a simple neural network from scratch in **C++**, including forward propagation and a basic **autograd** system.

It also served as a hands-on way to practice topological sorting, which is essential for correctly computing operations in a computational graph while respecting dependencies between variables. By building this from scratch, I gained a deeper understanding of how frameworks like TensorFlow and PyTorch manage the flow of data and gradients.
---

## Project Structure

```
.
├── build/                 # CMake build output (ignored in git)
├── src/                   # C++ source files
│   ├── Matrix.cpp
│   ├── Matrix.hpp
│   ├── ModelContext.cpp
│   ├── ModelContext.hpp
│   ├── ModelTrainingDesc.hpp
│   ├── ModelVariable.cpp
│   ├── ModelVariables.hpp
│   ├── PRNG.cpp
│   ├── PRNG.hpp
│   ├── Types.hpp
│   └── mnist.cpp          # main program
├── mnist_download.py      # Python script to download MNIST dataset
├── CMakeLists.txt         # CMake build file
└── requirements.txt       # Python dependencies for MNIST download
```

## Requirements

You only need Python to download the MNIST dataset:

```
tensorflow_datasets
numpy
```

Or install via:

```bash
pip install -r requirements.txt
```

## Usage

1. Download the MNIST dataset and convert to binary `.mat` files:

```bash
python mnist_download.py
```

This will create:

* `build/train_images.mat`
* `build/train_labels.mat`
* `build/test_images.mat`
* `build/test_labels.mat`

2. Create the build folder and generate build files with CMake:

```bash
cmake -S . -B build
```

3. Build the project:

```bash
cmake --build build
```

4. Run the program:

```bash
./build/mnist    # or run from VS Code directly if using the IDE
```

The program will train (or run inference) on the MNIST dataset and display predictions for the test set.

---