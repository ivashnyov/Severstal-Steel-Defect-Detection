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
from torchvision import models

class TestDataset(Dataset):
    '''Dataset for test prediction'''
    def __init__(self, root, df, transforms):
        self.root = root
        df['ImageId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
        self.fnames = df['ImageId'].unique().tolist()
        self.num_samples = len(self.fnames)
        self.transforms = transforms

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        path = os.path.join(self.root, fname)
        image = cv2.imread(path)
        images = self.transforms(image=image)["image"]
        images = torch.from_numpy(images.transpose((2, 0, 1))).float()
        return fname, images

    def __len__(self):
        return self.num_samples

def post_process(probability, threshold, min_size):
    '''Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored'''
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((256, 1600), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"
    sample_submission_path = 'data/sample_submission.csv'
    train_df_path = 'data/train.csv'
    data_folder = "data/train_images/"
    test_data_folder = "data/test_images/"
    crop_size = 256
    batch_size = 48
    num_workers = 16
    num_epochs = 50
    best_threshold = 0.5
    lr = 1e-4
    logdir = 'segmentator_resnet34'

    aug_val_seg = A.Compose([A.Normalize()],p=1.0)
    aug_val_class = A.Compose([A.Resize(crop_size, crop_size), A.Normalize()],p=1.0)

    print('best_threshold', best_threshold)
    min_size = 3500
    df = pd.read_csv(sample_submission_path)
    testset_seg = DataLoader(
        TestDataset(test_data_folder, df, aug_val_seg),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    testset_class = DataLoader(
        TestDataset(test_data_folder, df, aug_val_class),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    classifier = models.resnet34(pretrained=False)
    classifier.fc = nn.Linear(512, 1)
    classifier.load_state_dict(torch.load('classifier_resnet34/checkpoints/best_full.pth')['model_state_dict'])
    classifier.cuda()
    classifier.eval()

    segmentator = smp.Unet('resnet34', encoder_weights=None,
                     classes=4, activation=None)
    segmentator.load_state_dict(torch.load('segmentator_resnet34/checkpoints/best_full.pth')['model_state_dict'])
    segmentator.cuda()
    segmentator.eval()

    predictions = []
    with torch.no_grad():
        for batch_seg, batch_class in tqdm(zip(testset_seg, testset_class)):
            fnames, images_seg = batch_seg
            fnames, images_class = batch_class
            seg_preds = torch.sigmoid(segmentator(images_seg.cuda()))
            class_preds = classifier(images_class.cuda())
            seg_preds = seg_preds.detach().cpu().numpy()
            class_preds = class_preds.detach().cpu().numpy()
            for fname, preds, class_preds in zip(fnames, seg_preds, class_preds):
                if class_preds > 0.5:
                    for cls, pred in enumerate(preds):
                        pred, num = post_process(pred, best_threshold, min_size)
                        rle = mask2rle(pred)
                        name = fname + f"_{cls+1}"
                        predictions.append([name, rle])
                else:
                    for cls, pred in enumerate(preds):
                        name = fname + f"_{cls+1}"
                        predictions.append([name, ''])


    # save predictions to submission.csv
    df = pd.DataFrame(predictions, columns=['ImageId_ClassId', 'EncodedPixels'])
    df.to_csv("submission_segclassv2.csv", index=False)
    