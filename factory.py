import cv2
import albumentations as A
import numpy as np
import pandas as pd
import os
from scipy.ndimage import binary_dilation, binary_fill_holes
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image, tensor_from_mask_image
from pytorch_toolbelt import losses as L
from sklearn.model_selection import train_test_split
from torch import nn
import torch
from torch.nn import BCEWithLogitsLoss
from torch.nn import functional as F
from torch.utils.data import Dataset, WeightedRandomSampler, SubsetRandomSampler, DataLoader
from functools import partial

import numpy as np
import torch
from catalyst.dl import Callback, RunnerState, MetricCallback, CallbackOrder
from pytorch_toolbelt.utils.catalyst.visualization import get_tensorboard_logger
from pytorch_toolbelt.utils.torch_utils import to_numpy
from pytorch_toolbelt.utils.visualization import (
    render_figure_to_tensor,
    plot_confusion_matrix,
)
from sklearn.metrics import f1_score, confusion_matrix


def compute_boundary_mask(mask: np.ndarray) -> np.ndarray:
    dilated = binary_dilation(mask, structure=np.ones((9, 9), dtype=np.bool))
    dilated = binary_fill_holes(dilated)
    diff = dilated & ~mask
    diff = cv2.dilate(diff, kernel=(9, 9))
    diff = diff & ~mask
    #kernel = np.ones((4,),np.uint8)
    #diff = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
    return diff.astype(np.uint8)



def make_mask(row_id, df):
    '''Given a row index, return image_id and mask (256, 1600, 4) from the dataframe `df`'''
    fname = df.iloc[row_id].name
    labels = df.iloc[row_id][:4]
    masks = np.zeros((256, 1600, 4), dtype=np.float32) # float32 is V.Imp
    # 4:class 1～4 (ch:0～3)

    for idx, label in enumerate(labels.values):
        if label is not np.nan:
            label = label.split(" ")
            positions = map(int, label[0::2])
            length = map(int, label[1::2])
            mask = np.zeros(256 * 1600, dtype=np.uint8)
            for pos, le in zip(positions, length):
                mask[pos:(pos + le)] = 1
            masks[:, :, idx] = mask.reshape(256, 1600, order='F')
    return fname, masks


class SteelDatasetMulti(Dataset):
    def __init__(self, df, data_folder, transforms, phase, prepare_coarse = False, prepare_edges = False, prepare_class = False, prepare_full = False):
        self.df = df
        self.root = data_folder
        self.phase = phase
        self.transforms = transforms
        self.fnames = self.df.index.tolist()
        self.prepare_coarse = prepare_coarse
        self.prepare_edges = prepare_edges
        self.prepare_class = prepare_class
        self.prepare_full = prepare_full
        
    def __getitem__(self, idx):
        image_id, mask = make_mask(idx, self.df)
        image_path = os.path.join(self.root,  image_id)
        img = cv2.imread(image_path)
        if self.transforms is not None:
            augmented = self.transforms(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask'].astype(np.uint8)
        if self.prepare_full:
            all_masks_combined = (mask.sum(axis=2)>0).astype(np.uint8)        
            all_masks_combined = np.expand_dims(all_masks_combined, 2)
            mask = np.concatenate([mask, all_masks_combined], axis=2)
        if self.prepare_edges:
            all_masks_combined = (mask.sum(axis=2)>0).astype(np.uint8)  
            edges = compute_boundary_mask(all_masks_combined).astype(np.uint8)
            edges = np.expand_dims(edges, 2)
            mask = np.concatenate([mask, edges], axis=2)
        if self.prepare_coarse:
            coarse_mask = cv2.resize(mask,
                                     dsize=(mask.shape[1]//4, mask.shape[0]//4),
                                     interpolation=cv2.INTER_LINEAR)
            if self.prepare_edges:
                coarse_edges = cv2.resize(edges,
                                     dsize=(mask.shape[1]//4, mask.shape[0]//4),
                                     interpolation=cv2.INTER_LINEAR)
                coarse_edges = np.expand_dims(coarse_edges, 2)
        if self.prepare_class:
            if mask.sum()>0:
                has_defect = 1
            else:
                has_defect = 0
            
        data = {'features': tensor_from_rgb_image(img),
                'targets': tensor_from_mask_image(mask).float(),
                'image_id':image_id}
        
        #if self.prepare_coarse:
        #    data['coarse_targets'] =  tensor_from_mask_image(coarse_mask).float()

        #if self.prepare_edges:
        #    data['edges'] = tensor_from_mask_image(edges).float()
            
        #if self.prepare_edges and self.prepare_edges:
        #    data['coarse_edges'] = tensor_from_mask_image(coarse_edges).float()
            
        #if self.prepare_class:
        #    data['class'] = torch.from_numpy(np.expand_dims(has_defect, 0)).float()
            
            
        
        
        return data
  
    def __len__(self):
        return len(self.fnames)
    
    
def provider(
    data_folder,
    df_path,
    phase,
    transforms,    
    batch_size=8,
    num_workers=4,
    prepare_coarse = False, 
    prepare_edges = False,
    prepare_class = False, 
    prepare_full = False
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
    image_dataset = SteelDatasetMulti(df, data_folder, transforms, phase, prepare_coarse, prepare_edges, prepare_class, prepare_full)
    if phase=='train':
        shuffle = True
    else:
        shuffle = False
    dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=shuffle,   
    )

    return dataloader    


def light_augmentations(crop_size, safe_crop_around_mask = True):
    if safe_crop_around_mask:
        spatial_transform = A.Compose([
            A.CropNonEmptyMaskIfExists(height = crop_size, width = crop_size),
        ])
    else:
        spatial_transform = A.Compose([
            A.RandomCrop(height = crop_size, width = crop_size),
        ])

    return A.Compose([
        spatial_transform,

        # D4 Augmentations
        A.Compose([
            #A.Transpose(),            
            A.HorizontalFlip(),            
            A.VerticalFlip()
            #A.RandomRotate90(),
        ]),

        # Spatial-preserving augmentations:
        A.OneOf([
            A.Cutout(),
            A.GaussNoise(),
        ]),
        A.OneOf([
            A.RandomBrightnessContrast(),
            A.CLAHE(),
            A.HueSaturationValue(),
            A.RGBShift(),
            A.RandomGamma()
        ]),

        A.Normalize()
    ])


def medium_augmentations(crop_size, safe_crop_around_mask = True):
    if safe_crop_around_mask:
        spatial_transform = A.Compose([
            A.CropNonEmptyMaskIfExists(height = crop_size, width = crop_size),
        ])
    else:
        spatial_transform = A.Compose([
            A.RandomCrop(height = crop_size, width = crop_size),
        ])

    return A.Compose([
        spatial_transform,

        # Add occasion blur/sharpening
        A.OneOf([
            A.GaussianBlur(),
            A.MotionBlur(),
            A.IAASharpen()
        ]),

        # D4 Augmentations
        A.Compose([
            #A.Transpose(),
            A.VerticalFlip(),
            A.HorizontalFlip()
            #A.RandomRotate90(),
        ]),

        # Spatial-preserving augmentations:
        A.OneOf([
            A.Cutout(),
            A.GaussNoise(),
        ]),
        A.OneOf([
            A.RandomBrightnessContrast(),
            A.CLAHE(),
            A.HueSaturationValue(),
            A.RGBShift(),
            A.RandomGamma()
        ]),

        A.Normalize()
    ])


def hard_augmentations(crop_size, safe_crop_around_mask = True):
    if safe_crop_around_mask:
        spatial_transform = A.Compose([
            A.CropNonEmptyMaskIfExists(height = crop_size, width = crop_size),
            #A.PadIfNeeded(min_width=1792, min_height=256)
        ])
    else:
        spatial_transform = A.Compose([
            A.RandomCrop(height = crop_size, width = crop_size),
        ])

    return A.Compose([
        spatial_transform,

        # Add occasion blur
        A.OneOf([
            A.GaussianBlur(),
            A.GaussNoise(),
            A.IAAAdditiveGaussianNoise(),
            A.NoOp()
        ]),

        # D4 Augmentations
        A.Compose([
            #A.Transpose(),
            A.VerticalFlip(),
            A.HorizontalFlip()
            #A.RandomRotate90(),
        ]),

        A.Cutout(),
        # Spatial-preserving augmentations:
        A.OneOf([
            A.RandomBrightnessContrast(),
            A.CLAHE(),
            A.HueSaturationValue(),
            A.RGBShift(),
            A.RandomGamma(),
            A.NoOp()
        ]),

        A.Normalize()
    ])


def validation_augmentations():
    return A.Compose([
        A.Normalize()
    ],p=1.0)




def get_loss(loss_name: str, **kwargs):
    if loss_name.lower() == 'bce':
        return BCEWithLogitsLoss(**kwargs)

    if loss_name.lower() == 'focal':
        return L.BinaryFocalLoss(alpha=None, gamma=1.5, **kwargs)

    if loss_name.lower() == 'reduced_focal':
        return L.BinaryFocalLoss(alpha=None, gamma=1.5, reduced=True, **kwargs)

    if loss_name.lower() == 'bce_jaccard':
        return L.JointLoss(first=BCEWithLogitsLoss(), second=L.BinaryJaccardLoss(), first_weight=1.0, second_weight=0.5)

    if loss_name.lower() == 'bce_dice':
        return L.JointLoss(first=BCEWithLogitsLoss(), second=L.DiceLoss(mode='binary'), first_weight=1.0, second_weight=1.0)    

    if loss_name.lower() == 'bce_log_dice':
        return L.JointLoss(first=BCEWithLogitsLoss(), second=L.DiceLoss(mode='binary',log_loss=True), first_weight=1.0, second_weight=1.0)   
    
    if loss_name.lower() == 'bce_log_jaccard':
        return L.JointLoss(first=BCEWithLogitsLoss(), second=L.JaccardLoss(mode='binary',log_loss=True), first_weight=1.0, second_weight=0.5)

    if loss_name.lower() == 'bce_lovasz':
        return L.JointLoss(first=BCEWithLogitsLoss(), second=L.BinaryLovaszLoss(), first_weight=1.0, second_weight=0.25)

    raise KeyError(loss_name)
    
    
    
def get_optimizer(optimizer_name: str, parameters, lr: float, **kwargs):
    from torch.optim import SGD, Adam
    from radam import RAdam
    
    if optimizer_name.lower() == 'sgd':
        return SGD(parameters, lr, momentum=0.9, nesterov=True, **kwargs)

    if optimizer_name.lower() == 'adam':
        return Adam(parameters, lr, **kwargs)
    if optimizer_name.lower() == 'radam':
        return RAdam(parameters, lr, **kwargs)

    raise ValueError("Unsupported optimizer name " + optimizer_name)
    
def make_mask(row_id, df):
    '''Given a row index, return image_id and mask (256, 1600, 4) from the dataframe `df`'''
    fname = df.iloc[row_id].name
    labels = df.iloc[row_id][:4]
    masks = np.zeros((256, 1600, 4), dtype=np.float32) # float32 is V.Imp
    # 4:class 1～4 (ch:0～3)

    for idx, label in enumerate(labels.values):
        if label is not np.nan:
            label = label.split(" ")
            positions = map(int, label[0::2])
            length = map(int, label[1::2])
            mask = np.zeros(256 * 1600, dtype=np.uint8)
            for pos, le in zip(positions, length):
                mask[pos:(pos + le)] = 1
            masks[:, :, idx] = mask.reshape(256, 1600, order='F')
    return fname, masks

def mask2rle(img):
    '''
    img: numpy array, 1 -> mask, 0 -> background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def return_masks(df_path):
    df = pd.read_csv(df_path)
    # https://www.kaggle.com/amanooo/defect-detection-starter-u-net
    df['ImageId'], df['ClassId'] = zip(*df['ImageId_ClassId'].str.split('_'))
    df['ClassId'] = df['ClassId'].astype(int)
    df = df.pivot(index='ImageId',columns='ClassId',values='EncodedPixels')
    df['defects'] = df.count(axis=1)

    
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["defects"], random_state=69)
    return train_df, val_df

def dice(im1, im2, empty_score=1.0):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum







BINARY_MODE = "binary"
def binary_dice_iou_score(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    mode="dice",
    threshold=None,
    nan_score_on_empty=False,
    eps=1e-7,
) -> float:
    """
    Compute IoU score between two image tensors
    :param y_pred: Input image tensor of any shape
    :param y_true: Target image of any shape (must match size of y_pred)
    :param mode: Metric to compute (dice, iou)
    :param threshold: Optional binarization threshold to apply on @y_pred
    :param nan_score_on_empty: If true, return np.nan if target has no positive pixels;
        If false, return 1. if both target and input are empty, and 0 otherwise.
    :param eps: Small value to add to denominator for numerical stability
    :return: Float scalar
    """
    assert mode in {"dice", "iou"}

    # Binarize predictions
    if threshold is not None:
        y_pred = (y_pred > threshold).to(y_true.dtype)

    intersection = torch.sum(y_pred * y_true).item()
    cardinality = (torch.sum(y_pred) + torch.sum(y_true)).item()

    if mode == "dice":
        score = (2.0 * intersection) / (cardinality + eps)
    else:
        score = intersection / (cardinality + eps)

    has_targets = torch.sum(y_true) > 0
    has_predicted = torch.sum(y_pred) > 0

    if not has_targets:
        if nan_score_on_empty:
            score = np.nan
        else:
            score = float(not has_predicted)
    return score

def multilabel_dice_iou_score(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    mode="dice",
    threshold=None,
    eps=1e-7,
    nan_score_on_empty=False,
    classes_of_interest=None,
):
    ious = []
    num_classes = y_pred.size(0)

    if classes_of_interest is None:
        classes_of_interest = range(num_classes)

    for class_index in classes_of_interest:
        iou = binary_dice_iou_score(
            y_pred=y_pred[class_index],
            y_true=y_true[class_index],
            mode=mode,
            threshold=threshold,
            nan_score_on_empty=nan_score_on_empty,
            eps=eps,
        )
        ious.append(iou)

    return ious


class DiceScoreCallback(Callback):
    """
    A metric callback for computing either Dice or Jaccard metric
    which is computed across whole epoch, not per-batch.
    """

    def __init__(
        self,
        mode: str,
        metric="dice",
        input_key: str = "targets",
        output_key: str = "logits",
        nan_score_on_empty=True,
        prefix: str = None,
    ):
        """
        :param mode: One of: 'binary', 'multiclass', 'multilabel'.
        :param input_key: input key to use for precision calculation; specifies our `y_true`.
        :param output_key: output key to use for precision calculation; specifies our `y_pred`.
        :param accuracy_for_empty:
        """
        super().__init__(CallbackOrder.Metric)
        assert mode in {BINARY_MODE}

        if prefix is None:
            prefix = metric


        self.mode = mode
        self.prefix = prefix
        self.output_key = output_key
        self.input_key = input_key
        self.scores = []

        if self.mode == BINARY_MODE:
            self.score_fn = partial(
                binary_dice_iou_score,
                threshold=0.0,
                nan_score_on_empty=nan_score_on_empty,
                mode=metric,
            )

    def on_loader_start(self, state):
        self.scores = []

    @torch.no_grad()
    def on_batch_end(self, state: RunnerState):
        outputs = state.output[self.output_key].detach()
        targets = state.input[self.input_key].detach()

        batch_size = targets.size(0)
        #n_classes = targets.size(1)
        n_classes = 4
        score_per_image_class_pair = []
        for image_index in range(batch_size):
            for class_id in range(n_classes):
                score = self.score_fn(
                    y_pred=outputs[image_index,class_id,:,:],
                    y_true=targets[image_index,class_id,:,:]
                )
                score_per_image_class_pair.append(score)

        mean_score = np.nanmean(score_per_image_class_pair)
        state.metrics.add_batch_value(self.prefix, float(mean_score))
        self.scores.extend(score_per_image_class_pair)

    def on_loader_end(self, state):
        scores = np.array(self.scores)
        mean_score = np.nanmean(scores)

        state.metrics.epoch_values[state.loader_name][self.prefix] = float(mean_score)
