import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cv2
import segmentation_models_pytorch as smp
from torch import nn
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
from functools import partial
import torch.nn as nn
from catalyst.dl.utils import criterion
from torch.nn.modules.loss import _Loss
from pytorch_toolbelt.losses.functional import sigmoid_focal_loss, reduced_focal_loss
from pytorch_toolbelt.utils.catalyst.metrics import *
from skimage import measure, morphology
from catalyst.dl.utils.criterion import dice
from torchvision import models

if __name__ == "__main__":
    sample_submission_path = 'data/sample_submission.csv'
    train_df_path = 'data/train.csv'
    data_folder = "data/train_images/"
    test_data_folder = "data/test_images/"
    crop_size = 256
    batch_size = 8
    num_workers = 32
    num_epochs = 100
    lr = 3e-4
    logdir = 'models/resnet34_resize'

    aug_val = A.Compose([A.PadIfNeeded(1600, 1600, value=0), A.Normalize()],p=1.0)
    aug_train_heavy = A.Compose([A.PadIfNeeded(1600, 1600, value=0),
        # A.CropNonEmptyMaskIfExists(height=crop_size, width = crop_size),
        A.VerticalFlip(p=0.5),              
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Transpose(p=0.5),
        A.OneOf([A.RandomContrast(), 
                 A.CLAHE(), 
                 A.RandomBrightness(), 
                 A.RandomGamma(),
                 A.RandomBrightnessContrast()],p=0.5),
        A.OneOf([A.GridDistortion(),
                 A.ElasticTransform(), 
                 A.OpticalDistortion(),
                 A.ShiftScaleRotate(),
                 A.RGBShift()
                ],p=0.5),
        A.CoarseDropout(),
        A.Normalize()
    ],p=1.0)

    dataloader_train = provider(
                data_folder=data_folder,
                df_path=train_df_path,
                phase='train',
                transforms=aug_train_heavy,
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

    optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
    criterion = smp.utils.losses.BCEDiceLoss(eps=1.)
    scheduler = ReduceLROnPlateau(optimizer=optimizer, 
                                factor=0.5, 
                                patience=5,
                               min_lr=1e-7)    
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        logdir=logdir,
        scheduler=scheduler,
        callbacks=[DiceCallback(), IouCallback()],
        num_epochs=num_epochs,
        fp16=True,
        verbose=True
        )    