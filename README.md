# Scyclone-PyTorch
Reimplmentation of voice conversion system "Scyclone" with PyTorch.

## Install
(If your machine needs specific PyTorch version (e.g. old CUDA compatible version), install it before Scyclone installation.)  

`pip install git+https://github.com/tarepan/Scyclone-PyTorch`


## How to use
Notebook `Scyclone_PyTorch.ipynb`

## Original paper
Masaya Tanaka, et al.. (2020). [Scyclone: High-Quality and Parallel-Data-Free Voice Conversion Using Spectrogram and Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/2005.03334). Arxiv 2005.03334.

## Dependency Notes
### PyTorch version
PyTorch version: PyTorch v1.6 is working (We checked with v1.6.0).  

For dependency resolution, we do **NOT** explicitly specify the compatible versions.  
PyTorch have several distributions for various environment (e.g. compatible CUDA version.)  
Unfortunately it make dependency version management complicated for dependency management system.  
In our case, the system `poetry` cannot handle cuda variant string (e.g. `torch>=1.6.0` cannot accept `1.6.0+cu101`.)  
In order to resolve this problem, we use `torch==*`, it is equal to no version specification.  
`Setup.py` could resolve this problem (e.g. `torchaudio`'s `setup.py`), but we will not bet our effort to this hacky method.  
