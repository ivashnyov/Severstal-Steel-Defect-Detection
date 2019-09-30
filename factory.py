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
from torch.nn import BCEWithLogitsLoss
from torch.nn import functional as F
from torch.utils.data import Dataset, WeightedRandomSampler, SubsetRandomSampler, DataLoader

def compute_boundary_mask(mask: np.ndarray) -> np.ndarray:
    dilated = binary_dilation(mask, structure=np.ones((5, 5), dtype=np.bool))
    dilated = binary_fill_holes(dilated)

    diff = dilated & ~mask
    diff = cv2.dilate(diff, kernel=(5, 5))
    diff = diff & ~mask
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
    def __init__(self, df, data_folder, transforms, phase, prepare_coarse = False, prepare_edges = False):
        self.df = df
        self.root = data_folder
        self.phase = phase
        self.transforms = transforms
        self.fnames = self.df.index.tolist()
        self.prepare_coarse = prepare_coarse
        self.prepare_edges = prepare_edges
        
    def __getitem__(self, idx):
        image_id, mask = make_mask(idx, self.df)
        image_path = os.path.join(self.root,  image_id)
        img = cv2.imread(image_path)
        if self.transforms is not None:
            augmented = self.transforms(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask'].astype(np.uint8)
        all_masks_combined = (mask.sum(axis=2)>0).astype(np.uint8)
        if self.prepare_edges:
            edges = compute_boundary_mask(all_masks_combined)
        if self.prepare_coarse:
            coarse_mask = cv2.resize(mask,
                                     dsize=(mask.shape[1]//4, mask.shape[0]//4),
                                     interpolation=cv2.INTER_LINEAR)
            
            coarse_all_masks_combined = cv2.resize(all_masks_combined,
                                                   dsize=(mask.shape[1]//4, mask.shape[0]//4),
                                                   interpolation=cv2.INTER_LINEAR)
            
        data = {'features': tensor_from_rgb_image(img),
                'targets': tensor_from_mask_image(mask).float(),
                'targets_combined' : tensor_from_mask_image(all_masks_combined).float(),
                'image_id':image_id}
        
        if self.prepare_coarse:
            data['coarse_targets'] =  tensor_from_mask_image(coarse_mask).float()
            data['coarse_targets_combined'] = tensor_from_mask_image(coarse_all_masks_combined).float()

        if self.prepare_edges:
            data['edges'] = edges               
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
    prepare_edges = False
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
    image_dataset = SteelDatasetMulti(df, data_folder, transforms, phase, prepare_coarse, prepare_edges)
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
            A.Transpose(),
            A.RandomRotate90(),
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
            A.Transpose(),
            A.RandomRotate90(),
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
            A.Transpose(),
            A.RandomRotate90(),
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

    if loss_name.lower() == 'bce_log_jaccard':
        return L.JointLoss(first=BCEWithLogitsLoss(), second=L.BinaryJaccardLogLoss(), first_weight=1.0, second_weight=0.5)

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