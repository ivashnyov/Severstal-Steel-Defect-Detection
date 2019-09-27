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
from functools import partial
import torch.nn as nn
from catalyst.dl.utils import criterion
from pytorch_toolbelt import losses as L
from functools import partial
from torch.nn.modules.loss import _Loss
from pytorch_toolbelt.losses.functional import sigmoid_focal_loss, reduced_focal_loss
from pytorch_toolbelt.utils.catalyst.metrics import *
__all__ = ['BinaryFocalLoss', 'FocalLoss']



class DiceLoss(nn.Module):
    def __init__(
        self,
        eps: float = 1e-7,
        threshold: float = None,
        activation: str = "Sigmoid"
    ):
        super().__init__()

        self.loss_fn = partial(
            criterion.dice,
            eps=eps,
            threshold=threshold,
            activation=activation
        )

    def forward(self, logits, targets):
        dice = self.loss_fn(logits, targets)
        return 1 - dice
    
class BCEDiceLoss(nn.Module):
    def __init__(
        self,
        eps: float = 1e-7,
        threshold: float = None,
        activation: str = "Sigmoid",
        weight_dice : float = 1.0,
        weight_bce : float = 1.0
    ):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(
            eps=eps, threshold=threshold, activation=activation
        )
        self.weight_dice = weight_dice
        self.weight_bce = weight_bce
    def forward(self, outputs, targets):
        dice = self.dice_loss(outputs, targets)
        bce = self.bce_loss(outputs, targets)
        loss = self.weight_dice*dice + self.weight_bce*bce
        return loss
if __name__ == '__main__':
    sample_submission_path = 'data/sample_submission.csv'
    train_df_path = 'data/train.csv'
    data_folder = "data/train_images/"
    test_data_folder = "data/test_images/"
    crop_size = 256
    batch_size = 48
    num_workers = 8
    num_epochs = 30
    lr = 1e-4
    #logdir_old = 'logs/train_resnet50_Adam_heavyaugs_v2_softmax_FOCAL_DICE/'
    logdir = 'logs/train_resnext50_32x4d_Adam_v2_softmax_FOCAL_DICE_posttrain/'
    logdir_old = 'logs/train_resnext50_32x4d_Adam_heavyaugs_v2_softmax_FOCAL_DICE_posttrain/'
#    aug_train = A.Compose([
#        A.CropNonEmptyMaskIfExists(height=crop_size, width = crop_size),
#        A.VerticalFlip(p=0.5),              
#        A.RandomRotate90(p=0.5),
#        A.HorizontalFlip(p=0.5),
#        A.Transpose(p=0.5),
#        A.OneOf([A.RandomContrast(), 
#             A.CLAHE(), 
#             A.RandomBrightness(), 
#              A.RandomGamma(),
#             A.RandomBrightnessContrast()],p=0.5),
#        A.OneOf([A.GridDistortion(),
#              A.ElasticTransform(), 
#              A.OpticalDistortion(),
#              A.ShiftScaleRotate(),
#               A.RGBShift()
#                ],p=0.5),
#        A.CoarseDropout()
#        A.Normalize()
#    ],p=1.0)
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
                batch_size=8,
                num_workers=num_workers)
    model = smp.Unet('resnext50_32x4d', encoder_weights='instagram',
                 classes=5, activation='softmax')
    model.load_state_dict(torch.load(os.path.join(logdir_old,'checkpoints/best.pth'))['model_state_dict'])
    loaders = collections.OrderedDict()
    loaders["train"] = dataloader_train
    loaders["valid"] = dataloader_val
    runner = SupervisedRunner()
    #Train with Adam first
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
    criterion = L.JointLoss(first=L.FocalLoss(), first_weight=1.0, 
                            second=L.MulticlassDiceLoss(), second_weight=5.0)
    scheduler = ReduceLROnPlateau(optimizer=optimizer, 
                                factor=0.5, 
                                patience=5,
                               min_lr=1e-7)    
    #criterion = L.JointLoss(first=L.FocalLoss(), first_weight=1.0,
    #second=L.LovaszLoss(), second_weight=0.3)
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        logdir=logdir,
        scheduler=scheduler,
        callbacks=[JaccardScoreCallback(mode='multiclass')],
        num_epochs=num_epochs,
        verbose=True
        )    
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    #scheduler = OneCycleLR(optimizer=optimizer, num_steps=num_epochs*10, lr_range=(1e-3, 1e-5), warmup_fraction=0.5) 
    #runner.train(
    #    model=model,
    #    criterion=criterion,
    #    optimizer=optimizer,
    #    loaders=loaders,
    #    scheduler=scheduler,
    #    logdir=logdir,
    #    callbacks=[JaccardScoreCallback(mode='multiclass')],
    #    num_epochs=num_epochs*10,
    #    verbose=True
    #    )    
