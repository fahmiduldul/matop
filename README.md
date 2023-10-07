# MATOP (Matrix Operation)

## Overview

This is simple library to do high performance matrix operation in C++.
The goal of this libary is to be able run matrix operation using CUDA, CPU SIMD and OpenCL but currently only support CUDA.

The idea of this library is to separate the `.cpp` code and `.cu` code. This will enable the client code to be decoupled with the CUDA code.
Which provides high level abstraction on the client perspective.

## Build

*Prerequisite:*
- Make sure that you have NVidia CUDA toolkit installed.
- Have C++17 ready C/C++ compiler

*Steps:*
- run `nvcc src/main.cpp src/matrix_operation.cu -o main --std c++17`
- if there's no compilation error, simply run `.main`
