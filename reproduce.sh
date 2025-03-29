#!/usr/bin/env bash

# Install Miniconda
curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -o miniconda.sh
bash miniconda.sh -b -u -p "$HOME/miniconda"

# Set environment variables
export PATH="$HOME/miniconda/bin:$PATH"
echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc

# Source Conda directly instead of relying on ~/.bashrc
source "$HOME/miniconda/etc/profile.d/conda.sh"
source "$HOME/miniconda/etc/profile.d/mamba.sh"  # Ensure Mamba is loaded
conda install -c conda-forge libfmt -y

# Install Mamba
conda install -c conda-forge mamba -y
mamba shell init --shell=bash
source "$HOME/miniconda/etc/profile.d/conda.sh"  # Ensure changes are applied

# Set Up Conda env 
mamba create -n DevMuT python=3.9 -y
source activate DevMuT  # Use `source` instead of `mamba activate`
pip install --upgrade pip setuptools wheel
pip install -r "/home/DevMuT/code/DevMuT/requirements.txt"

# Set environment variables
export CONTEXT_DEVICE_TARGET=GPU
export CUDA_VISIBLE_DEVICES=0,1

# Run 
python3 "/home/DevMuT/code/DevMuT/mutation_test.py"

