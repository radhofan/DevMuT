#!/usr/bin/env bash

# Install Miniconda
curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -o miniconda.sh
bash miniconda.sh -b -u -p "$HOME/miniconda"

# Set environment variables
export PATH="$HOME/miniconda/bin:$PATH"
echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc

# Source Conda
source "$HOME/miniconda/etc/profile.d/conda.sh"

# First install libfmt explicitly
conda install -c conda-forge libfmt=11 -y

# Then install mamba
conda install -c conda-forge fmt
source "$HOME/miniconda/etc/profile.d/mamba.sh"

# Set Up Conda env 
mamba create -n DevMuT python=3.9 -y
conda activate DevMuT
pip install --upgrade pip setuptools wheel
pip install -r "/home/DevMuT/code/DevMuT/requirements.txt"

# Set environment variables
export CONTEXT_DEVICE_TARGET=GPU
export CUDA_VISIBLE_DEVICES=0,1

# Run 
python3 "/home/DevMuT/code/DevMuT/mutation_test.py"