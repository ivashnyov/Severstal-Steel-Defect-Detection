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
from radam import *
from skimage import measure
from skimage import morphology
from radam import *
from functools import partial
from catalyst.dl.utils.criterion import dice
class DiceLoss(nn.Module):
    def __init__(
        self,
        eps: float = 1e-7,
        threshold: float = None,
        activation: str = "Sigmoid"
    ):
        super().__init__()

        self.loss_fn = partial(
            dice,
            eps=eps,
            threshold=threshold,
            activation=activation
        )

    def forward(self, logits, targets):
        dice = self.loss_fn(logits, targets)
        return 1 - dice

class SteelDataset(Dataset):
    def __init__(self, df, data_folder, transforms, phase, erosion_factor = 7, dilution_factor = 5):
        self.df = df
        self.root = data_folder
        self.phase = phase
        self.transforms = transforms
        self.fnames = self.df.index.tolist()
        self.erosion_factor = erosion_factor
        self.dilution_factor = dilution_factor
    def __getitem__(self, idx):
        image_id, mask = make_mask(idx, self.df)
        #print(mask.max())
        image_path = os.path.join(self.root,  image_id)
        img = cv2.imread(image_path)
        if self.transforms is not None:
            augmented = self.transforms(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        all_masks_combined = (mask.sum(axis=2)>0).astype(int)
        all_masks_combined_eroded = morphology.binary_erosion(all_masks_combined,
                                                              morphology.square(self.erosion_factor))
        all_masks_boundaries = morphology.binary_dilation(all_masks_combined-all_masks_combined_eroded,
                                                          morphology.square(self.dilution_factor))  
        all_masks_combined = np.expand_dims(all_masks_combined,-1)
        all_masks_combined_eroded = np.expand_dims(all_masks_combined_eroded,-1) 
        all_masks_boundaries = np.expand_dims(all_masks_boundaries,-1) 
        mask = np.concatenate([mask, 
                               all_masks_combined, 
                               all_masks_combined_eroded, 
                               all_masks_boundaries],axis=2)
        img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        mask = torch.from_numpy(mask.transpose((2, 0, 1))).float()
        #mask = torch.from_numpy(mask).long()
        return img, mask
  
    def __len__(self):
        return len(self.fnames)
    
def provider(
    data_folder,
    df_path,
    phase,
    transforms,
    batch_size=8,
    num_workers=4,
):
    '''Returns dataloader for the model training'''
    df = pd.read_csv(df_path)
    # https://www.kaggle.com/amanooo/defect-detection-starter-u-net
    df['ImageId'], df['ClassId'] = zip(*df['ImageId_ClassId'].str.split('_'))
    df['ClassId'] = df['ClassId'].astype(int)
    df = df.pivot(index='ImageId',columns='ClassId',values='EncodedPixels')
    df['defects'] = df.count(axis=1)

    
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["defects"], random_state=69)
    df = train_df if phase == "train" else val_df
    image_dataset = SteelDataset(df, data_folder, transforms, phase)
    dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,   
    )

    return dataloader

if __name__ == '__main__':
    sample_submission_path = 'data/sample_submission.csv'
    train_df_path = 'data/train.csv'
    data_folder = "data/train_images/"
    test_data_folder = "data/test_images/"
    logdir_pretrain = 'logs/train_resnext101_32x16d_withboundaries_pretrain/' 
    logdir_train = 'logs/train_resnext101_32x16d_withboundaries_train/'
    batch_size=16
    batch_size_val=1
    num_workers=8
    ttatype = None
    crop_size = 256
    num_epochs_pretrain = 50
    num_epochs_train = 25
    lr = 3e-4
    aug_val = A.Compose([A.Normalize()],p=1.0)
    aug_train_heavy = A.Compose([
        A.CropNonEmptyMaskIfExists(height=crop_size, width = crop_size),
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
    aug_train = A.Compose([
        A.CropNonEmptyMaskIfExists(height=crop_size, width = crop_size),
        A.VerticalFlip(p=0.5),              
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Transpose(p=0.5),
        A.Normalize()
    ],p=1.0)    
    #train with heavy augs
    model = smp.Unet('resnext101_32x16d', encoder_weights='instagram',
                 classes=7, activation='sigmoid') #try heavier like _resnext101_32x16d
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
                batch_size=batch_size_val,
                num_workers=num_workers)
    loaders = collections.OrderedDict()
    loaders["train"] = dataloader_train
    loaders["valid"] = dataloader_val
    runner = SupervisedRunner()
    optimizer = RAdam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer=optimizer, 
                              factor=0.5, 
                              patience=5,
                              min_lr=1e-7)     
    criterion = L.JointLoss(first=nn.BCEWithLogitsLoss(), first_weight=1.0, 
                        second=DiceLoss(), second_weight=1.0)
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        logdir=logdir_pretrain,
        scheduler=scheduler,
        callbacks=[JaccardScoreCallback(mode='multilabel'),
               EarlyStoppingCallback(patience=10)],
        num_epochs=num_epochs_pretrain,
        verbose=True
        )   
    #Finilize with D4 augs
    model.load_state_dict(torch.load(os.path.join(logdir_pretrain,'checkpoints/best.pth'))['model_state_dict'])
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
                batch_size=batch_size_val,
                num_workers=num_workers)   
    loaders = collections.OrderedDict()
    loaders["train"] = dataloader_train
    loaders["valid"] = dataloader_val
    runner = SupervisedRunner()    
    optimizer = RAdam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer=optimizer, 
                              factor=0.5, 
                              patience=5,
                              min_lr=1e-7)     
    criterion = L.JointLoss(first=nn.BCEWithLogitsLoss(), first_weight=1.0, 
                        second=DiceLoss(), second_weight=1.0)
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        logdir=logdir_train,
        scheduler=scheduler,
        callbacks=[JaccardScoreCallback(mode='multilabel'),
               EarlyStoppingCallback(patience=10)],
        num_epochs=num_epochs_train,
        verbose=True
        ) 
