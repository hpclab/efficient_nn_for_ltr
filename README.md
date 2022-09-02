# DistillMLPToRank
This is the official repository of F. M. Nardini, C. Rulli, S. Trani, R. Venturini, ["Distilled Neural Networks for Efficient Learning to Rank"](https://ieeexplore.ieee.org/abstract/document/9716821). IEEE TKDE. 2022


## Requirements

This code has been tested with python 3.7 and PyTorch 1.6. In this PyTorch version, there was not a native way to implement pruning, so we rely on the [distiller](https://github.com/IntelLabs/distiller/) library.
__You will need to install it separately, following the developers guidelines.__

To install the other dependencies, simply type 
```
pip install -r requirements
```


## Training a model 



