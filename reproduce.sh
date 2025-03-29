#!/usr/bin/env bash

# Install Miniconda
curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -o miniconda.sh
bash miniconda.sh -b -p "$HOME/miniconda"

# Set environment variables and source changes
export PATH="$HOME/miniconda/bin:$PATH"
echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc  

# Install Mamba
conda install -c conda-forge mamba -y
mamba shell init --shell=bash
source ~/.bashrc  

# Set Up Conda env 
mamba create -n DevMuT python=3.9 -y
mamba activate DevMuT
pip install --upgrade pip setuptools wheel
pip install -r "$HOME/DevMuT/code/DevMuT/requirements.txt"

# Set environment variables
export CONTEXT_DEVICE_TARGET=GPU
export CUDA_VISIBLE_DEVICES=0,1  # Fix the typo

# Run 
python3 "$HOME/DevMuT/code/DevMuT/mutation_test.py"
