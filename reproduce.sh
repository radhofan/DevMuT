#!/usr/bin/env bash

# Install Miniconda
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda

# Set environment variables
export PATH="$HOME/miniconda/bin:$PATH"

# Set Up Conda env 
conda create -n DevMuT python=3.9 -y
conda activate DevMuT
conda install --file DevMuT/code/DevMuT/requirments.txt -c conda-forge -y
pip install absl-py==0.15.0
pip install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y

# Set environment variables
export CONTEXT_DEVICE_TARGET=GPU
export VICES=0,1

# Run 
python DevMuT/code/DevMuT/mutation_test.py
