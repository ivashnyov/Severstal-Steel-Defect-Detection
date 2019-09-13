import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cv2
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, WeightedRandomSampler, SubsetRandomSampler, DataLoader
from catalyst.contrib.schedulers import OneCycleLR, ReduceLROnPlateau, StepLR, MultiStepLR
from catalyst.dl.runner import SupervisedRunner
from catalyst.dl.callbacks import *
from catalyst.dl.core.state import RunnerState
from catalyst.dl.core import MetricCallback
from catalyst.dl.callbacks import CriterionCallback
import albumentations as A
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import collections
from pytorch_toolbelt import losses as L
from catalyst.contrib import criterion as C
from utils import *

if __name__ == '__main__':
    sample_submission_path = 'data/sample_submission.csv'
    train_df_path = 'data/train.csv'
    data_folder = "data/train_images/"
    test_data_folder = "data/test_images/"
    crop_size = 224
    batch_size = 64
    num_workers = 8
    num_epochs = 20
    lr = 1e-5
    logdir = 'logs/train_Adam_SGD_224_D4Augs/'
    aug_train = A.Compose([
        A.CropNonEmptyMaskIfExists(height=crop_size, width = crop_size),
        A.VerticalFlip(p=0.5),              
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Transpose(p=0.5),
        A.Normalize()
    ],p=1.0)
    aug_val = A.Compose([
        A.CropNonEmptyMaskIfExists(height=crop_size, width = crop_size),
        A.Normalize()],p=1.0)
    
    dataloader_train = provider(
                data_folder=data_folder,
                df_path=train_df_path,
                phase='train',
                transforms=aug_train,
                batch_size=batch_size,
                num_workers=num_workers)
    dataloader_val = provider(
                data_folder=data_folder,
                df_path=train_df_path,
                phase='val',
                transforms=aug_val,
                batch_size=batch_size,
                num_workers=num_workers)
    model = smp.Unet('resnet34', encoder_weights='imagenet',
                 classes=4, activation='sigmoid')
    loaders = collections.OrderedDict()
    loaders["train"] = dataloader_train
    loaders["valid"] = dataloader_val
    runner = SupervisedRunner()
    #Train with Adam first
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = C.BCEDiceLoss()
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        logdir=logdir,
        callbacks=[DiceCallback(), IouCallback()],
        num_epochs=num_epochs,
        verbose=True
        )    
    #Train with SGD
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = OneCycleLR(optimizer=optimizer, num_steps=2*num_epochs, momentum_range=(0.8, 0.99), lr_range=(lr*10, lr/10), warmup_fraction=0.5)   
    criterion = C.BCEDiceLoss()
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        logdir=logdir,
        callbacks=[DiceCallback(), IouCallback()],
        num_epochs=2*num_epochs,
        verbose=True
        )      