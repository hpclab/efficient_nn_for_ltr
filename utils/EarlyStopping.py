"""
Largely inspired by https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py

"""


import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, mode = "min", patience=10, verbose=False, delta=0, ):
        """
        Args:
            mode (string) : One of "min" or "max". In "min" mode, early stopping criterion applies when the monitored
                            quantity has stopped decreasing. Conversely, in "max" mode, it applies when the metric has stop increasing

            patience (int): How long to wait after last metric improvement.
                            Default: 7
            verbose (bool): If True, prints a message for each metric improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.mode = mode
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.metric = np.Inf
        self.delta = delta

    def __call__(self, score, model):


        if self.best_score is None:
            self.best_score = score

        if self.is_better(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter+=1
            if self.counter >= self.patience:
                self.early_stop = True

    def is_better(self, score):
        if self.mode == "min":
            return score < self.best_score -self.delta
        else:
            return score > self.best_score + self.delta
