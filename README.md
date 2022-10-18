# DistillMLPToRank
This is the official repository of F. M. Nardini, C. Rulli, S. Trani, R. Venturini, ["Distilled Neural Networks for Efficient Learning to Rank"](https://ieeexplore.ieee.org/abstract/document/9716821). IEEE TKDE. 2022


## Requirements

This code has been tested with python 3.7 and PyTorch 1.6. In this PyTorch version, there was not a native way to implement pruning, so we rely on the [distiller](https://github.com/IntelLabs/distiller/) library. This amy cause some difficultier during the installation as Distiller's dependencies are now broken. 
__To install distiller__, clone its repository, type ```cd distiller``` and then comment the following dependencies in ```requirements.txt.```

- torch
- tensorflow
- torchvision
- pretrainedmodels 

To install the other dependencies, simply type 
```
pip install -r requirements
```

## Download Dataset

Download link for the datasets:
- MSN30k [here](https://www.microsoft.com/en-us/research/project/mslr/)
- Istella-S [here](http://quickrank.isti.cnr.it/istella-dataset/)

## Training a model 

The script ```train.py``` allows to train a model by distillation from a pre-trained ensemble of regression trees. We provide the tree-based models, thus you do not have to it on your own. Run 

```python train.py --help```

to see all the training options.  

### Example

```python train.py --dataset-name msn30k --original-model LM600_msn --original-model-path msn30k_256leaves_ensemble.xml --name sample_model```

Inside the folder ```logs``` you will find a subfolder named "sample_model" that contains the logs file.



