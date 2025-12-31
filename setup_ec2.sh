#!/bin/bash
# save as setup_ec2.sh

# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install essential packages
sudo apt-get install -y build-essential git wget curl htop tmux

# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
rm Miniconda3-latest-Linux-x86_64.sh
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda init

# Create Python environment
conda create -n embodied python=3.10 -y
conda activate embodied

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Clone your repository
cd ~
git clone https://github.com/YOUR_USERNAME/embodied-minds.git
cd embodied-minds

# Install project dependencies
pip install -r requirements.txt

# Install additional ML tools
pip install transformers accelerate bitsandbytes einops
pip install wandb tensorboard
pip install jupyter jupyterlab

# Setup EmbodiedBench dataset
bash scripts/setup_embodiedbench.sh

echo "Setup complete! Activate environment with: conda activate embodied"