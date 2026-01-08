cmake_minimum_required(VERSION 3.16)
project(MNIST_CPP)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Source files
set(SRC
    src/Matrix.cpp
    src/ModelContext.cpp
    src/ModelVariable.cpp
    src/PRNG.cpp
    src/mnist.cpp
)

add_executable(mnist ${SRC})

# Include headers
target_include_directories(mnist PRIVATE src)