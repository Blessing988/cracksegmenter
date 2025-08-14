import os
import numpy as np
import numpy
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from tqdm import tqdm
import yaml
from losses import BinaryDiceLoss


with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)


def get_transforms():
    train_transforms = A.Compose([
            # Spatial-level transforms (applied to both image and mask)
            A.Resize(height=config['data']['image_size'], 
                     width=config['data']['image_size']), # should be removed or chnaged to 448 x 448 during binary segmentation
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            # make sure these are Python floats/tuples, and disable shear:
            # A.ShiftScaleRotate(
            #     shift_limit=0.1,                # float → uniform in [-0.1,0.1]
            #     scale_limit=(0.9, 1.1),         # tuple of floats
            #     rotate_limit=(-30.0, 30.0),     # tuple of floats
            #     shear_limit=0.0,                # force zero shear
            #     p=0.5,
            #     border_mode=cv2.BORDER_CONSTANT),
            
            # # Image-only transforms
            A.GaussNoise(p=0.2),
            A.ColorJitter(
                brightness=0.2, 
                contrast=0.2, 
                saturation=0.2, 
                hue=0.1, 
                p=0.5
            ),
            
            # Normalize the image using ResNet's mean and std
            A.Normalize(
                mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225)
            ),
            # Convert image and mask to PyTorch tensors
            ToTensorV2()
        ],
        additional_targets={"mask": "mask"} )

    val_transforms = A.Compose([
        A.Resize(height=512, width=512), # should be removed or chnaged to 448 x 448 during binary segmentation
        # Resize images and masks
        # Normalize the image using ResNet's mean and std
        A.Normalize(
            mean=(0.485, 0.456, 0.406), 
            std=(0.229, 0.224, 0.225)
        ),
        # Convert image and mask to PyTorch tensors
        ToTensorV2()
    ], 
    additional_targets={"mask": "mask"})
    
    return train_transforms, val_transforms



# def get_transforms():
#     train_transforms = A.Compose([
#         A.Resize(448, 448),

#         A.HorizontalFlip(p=0.5),
#         A.VerticalFlip(p=0.5),
#         A.RandomRotate90(p=0.5),

#         # rotate via OpenCV (no skimage)
#         A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, p=0.5),

#         # scale and translate via OpenCV
#         A.RandomScale(scale_limit=0.1, p=0.5),
#         A.RandomTranslate(percent=0.1, p=0.5, interpolation=cv2.INTER_LINEAR),

#         A.GaussNoise(p=0.2),
#         A.ColorJitter(
#             brightness=0.2,
#             contrast=0.2,
#             saturation=0.2,
#             hue=0.1,
#             p=0.5
#         ),

#         A.Normalize(
#             mean=(0.485, 0.456, 0.406),
#             std=(0.229, 0.224, 0.225)
#         ),
#         ToTensorV2()
#     ], additional_targets={"mask": "mask"})

#     val_transforms = A.Compose([
#         A.Resize(448, 448),
#         A.Normalize(
#             mean=(0.485, 0.456, 0.406),
#             std=(0.229, 0.224, 0.225)
#         ),
#         ToTensorV2()
#     ], additional_targets={"mask": "mask"})

#     return train_transforms, val_transforms



class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, num_classes=1):
        super(SegmentationDataset, self).__init__()
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
        

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        
        # image_path = os.path.join(self.images_dir, self.image_files[idx])
        # mask_path = os.path.join(self.masks_dir, self.image_files[idx])
        
        # # image = Image.open(image_path).convert("RGB")
        # # mask = Image.open(mask_path)
        ##########################################################
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.masks_dir, os.path.splitext(self.image_files[idx])[0] + '.jpg')

        ############################################################
            
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, 0)
        mask = np.where(mask==0, mask, 1)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

            
        if isinstance(image, numpy.ndarray):
            image = torch.from_numpy(image.transpose(2,0,1)).float() / 255.0
            
        if isinstance(mask, numpy.ndarray):
            mask  = torch.from_numpy(mask)
            
        # mask = torch.from_numpy(mask).long()
        return image, mask.long()

        
def get_dataloader(images_dir, masks_dir, batch_size, num_workers, num_classes,  transform, shuffle):
    dataset = SegmentationDataset(images_dir, masks_dir, transform, num_classes)
    # dataset = MultiClassSegmentationDataset(images_dir, masks_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                            num_workers=num_workers, pin_memory=True,
                             drop_last=True)
    return dataloader


