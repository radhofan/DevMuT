#!/usr/bin/env bash

# Install Miniconda
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda

# Set environment variables
export PATH="$HOME/miniconda/bin:$PATH"

# Set Up Conda env 
conda create -n DevMuT python=3.9 -y
conda activate DevMuT
# pip install -r DevMuT/code/DevMuT/requirments.txt

REQUIREMENTS_FILE="DevMuT/code/DevMuT/requirments.txt"
while IFS= read -r package; do
    # Skip empty lines or comments
    if [[ -z "$package" || "$package" == "#"* ]]; then
        continue
    fi

    echo "Installing $package..."
    if ! pip install "$package"; then
        echo "Failed to install $package"
    fi
done < "$REQUIREMENTS_FILE"

echo "Installation process completed."

# Set environment variables
export CONTEXT_DEVICE_TARGET=GPU
export VICES=0, 1

# Run 
python DevMuT/code/DevMuT/mutation_test.py
