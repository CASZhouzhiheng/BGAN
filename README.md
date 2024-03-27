# BGAN


## Overview
This repository contains the implementation of the BGAN model using PyTorch and DGL. The model is specifically designed for fMRI analysis and is applied in the paper titled "**[Classification of Alzheimer’s Disease Based on Graph Neural Network Method]**" For detailed information, please refer to the paper.

## Paper Reference
If you use or refer to this BGAN model in your work, please cite the following paper:
"**[Classification of Alzheimer’s Disease Based on Graph Neural Network Method]**"

## Requirements
PyTorch,DGL,NumPy

## Example Usage
```python
# Importing the BGAN model
from BGAN_model import BGAN

# Creating an instance of the BGAN model
model = FCbasedGCN(in_dim=..., out_dim=..., n_class=...)

...

# Forward pass
output = model(graph, feature)
