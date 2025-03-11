#!/usr/bin/env bash

# Install Miniconda
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda

# Set environment variables
export PATH="$HOME/miniconda/bin:$PATH"

# Set Up Conda env 
conda create -n DevMuT python=3.9 -y
conda init
exec bash
conda activate DevMuT
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r DevMuT/code/DevMuT/requirements.txt

# Set environment variables
export CONTEXT_DEVICE_TARGET=GPU
export VICES=0,1

# Run 
python DevMuT/code/DevMuT/mutation_test.py
