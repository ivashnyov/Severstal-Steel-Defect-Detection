import numpy as np
import pandas as pd
from torch.utils.data import Dataset, WeightedRandomSampler, SubsetRandomSampler, DataLoader
import albumentations as A
from sklearn.model_selection import train_test_split
import os
import cv2
import torch 
from tqdm import tqdm
from pytorch_toolbelt.inference.tiles import ImageSlicer, CudaTileMerger
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image, to_numpy
import matplotlib.pyplot as plot

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

def plot_image_gt_preds(idx, data_df, masks_df, augmentation, data_folder, model):
    f, ax = plt.subplots(4,2,figsize=(20,10))
    image_name = data_df.image_id.values[idx]
    image = cv2.imread(os.path.join(data_folder, image_name))
    augmented  = augmentation(image=image)
    image_processed = augmented['image']
    image_processed = torch.from_numpy(np.expand_dims(image_processed.transpose((2, 0, 1)),0)).float()
    predictions = torch.nn.Sigmoid()(model(image_processed.cuda())[0]).detach().cpu().numpy()
    predictions_bin = (predictions>0.5).astype(int)
    labels = masks_df.loc[image_name,:][:4]
    masks = np.zeros((256, 1600, 4), dtype=np.float32) # float32 is V.Imp
    for idx, label in enumerate(labels.values):
        if label is not np.nan:
            label = label.split(" ")
            positions = map(int, label[0::2])
            length = map(int, label[1::2])
            mask = np.zeros(256 * 1600, dtype=np.uint8)
            for pos, le in zip(positions, length):
                mask[pos:(pos + le)] = 1
            masks[:, :, idx] = mask.reshape(256, 1600, order='F')
    #assert(fname==data_df.image_id.values[idx])
    for defect_type in range(4):
        ax[defect_type,0].imshow(image)
        ax[defect_type,0].imshow(predictions[defect_type,:,:],alpha=0.5)
        ax[defect_type,1].imshow(image)
        ax[defect_type,1].imshow(masks[:,:,defect_type],alpha=0.5)        

def make_mask_allTogether(row_id, df):
    '''Given a row index, return image_id and mask (256, 1600, 4) from the dataframe `df`'''
    fname = df.iloc[row_id].name
    labels = df.iloc[row_id][:4]
    masks_total = np.zeros((256, 1600), dtype=np.float32) # float32 is V.Imp
    for idx, label in enumerate(labels.values):
        if label is not np.nan:
            label = label.split(" ")
            positions = map(int, label[0::2])
            length = map(int, label[1::2])
            mask = np.zeros(256 * 1600, dtype=np.uint8)
            for pos, le in zip(positions, length):
                mask[pos:(pos + le)] = 1
            mask = (idx+1)*mask.reshape(256, 1600, order='F') #get binary mask for class, multiply it 
            masks_total += mask
    return fname, masks_total

class SteelDataset(Dataset):
    def __init__(self, df, data_folder, transforms, phase):
        self.df = df
        self.root = data_folder
        self.phase = phase
        self.transforms = transforms
        self.fnames = self.df.index.tolist()
        
    def __getitem__(self, idx):
        image_id, mask = make_mask(idx, self.df)
        #print(mask.max())
        image_path = os.path.join(self.root,  image_id)
        img = cv2.imread(image_path)
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']
        
        img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        mask = torch.from_numpy(mask.transpose((2, 0, 1))).float()
        # torch.from_numpy(mask).long()
        # if mask.max() > 0:
        #     mask = torch.tensor(np.asarray([1])).float()
        # else:
        #     mask = torch.tensor(np.asarray([0])).float()
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
def run_validation(data_df, model, data_folder, augmentation, tiles=False):
    total_dice_coeffs = []
    mean_dice_per_image = []
    for image_n in tqdm(range(data_df.shape[0])):
        image = cv2.imread(os.path.join(data_folder,data_df.index.values[image_n]))
        augmented  = augmentation(image=image)
        image_processed = augmented['image']
        if tiles:
            tiler = ImageSlicer(image_processed.shape[:2], tile_size=(224, 224), tile_step=(56, 56), weight='mean')
            merger = CudaTileMerger(tiler.target_shape, 4, tiler.weight)
            tiles = [tensor_from_rgb_image(tile) for tile in tiler.split(image_processed)]
            for tiles_batch, coords_batch in DataLoader(list(zip(tiles, tiler.crops)), batch_size=16, pin_memory=True):
                tiles_batch = tiles_batch.float().cuda()
                pred_batch = torch.nn.Sigmoid()(model(tiles_batch))
                merger.integrate_batch(pred_batch, coords_batch)
            predictions = np.moveaxis(to_numpy(merger.merge()), 0, -1)
            predictions = tiler.crop_to_orignal_size(predictions)
        else:
            image_processed = torch.from_numpy(np.expand_dims(image_processed.transpose((2, 0, 1)),0)).float()
            predictions = torch.nn.Sigmoid()(model(image_processed.cuda())[0]).detach().cpu().numpy()
            predictions = np.moveaxis(predictions, 0, -1)
        predictions_bin = (predictions>0.5).astype(int)
        fname, masks = make_mask(image_n,data_df)
        dices_image = []
        for defect_type in range(4):
            computed_dice = dice(masks[:,:,defect_type],predictions_bin[:,:,defect_type])
            total_dice_coeffs.append(computed_dice)
            dices_image.append(computed_dice)
        mean_dice_per_image.append(np.mean(dices_image))
    return np.mean(total_dice_coeffs), mean_dice_per_image

def return_masks(df_path):
    df = pd.read_csv(df_path)
    # https://www.kaggle.com/amanooo/defect-detection-starter-u-net
    df['ImageId'], df['ClassId'] = zip(*df['ImageId_ClassId'].str.split('_'))
    df['ClassId'] = df['ClassId'].astype(int)
    df = df.pivot(index='ImageId',columns='ClassId',values='EncodedPixels')
    df['defects'] = df.count(axis=1)

    
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["defects"], random_state=69)
    return train_df, val_df