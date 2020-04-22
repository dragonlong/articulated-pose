#!/usr/bin/env bash
CUDA_DIR=/opt/apps/cuda/9.0.176/
CONDA_ENV_DIR=~/anaconda3/envs/dlubu36/
nvcc_bin=$CUDA_DIR/bin/nvcc

cuda_include_dir=$CUDA_DIR/include
tensorflow_include_dir=$CONDA_ENV_DIR/lib/python3.6/site-packages/tensorflow/include
tensorflow_external_dir=$CONDA_ENV_DIR/lib/python3.6/site-packages/tensorflow/include/external/nsync/public

cuda_library_dir=$CUDA_DIR/lib64/
tensorflow_library_dir=$CONDA_ENV_DIR/lib/python3.6/site-packages/tensorflow

