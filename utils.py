import os, cv2, random, shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import albumentations as albu
from torch.utils.data import Dataset as BaseDataset

class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    def __init__(self, images_dir, masks_dir=None, CLASSES=None, classes=None, augmentation=None, preprocessing=None):
        
        self.ids = os.listdir(images_dir)
        self.ids.sort()
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        
        if masks_dir:
            self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        else:
            self.masks_fps = None
        
        # convert str names to class values on masks
        self.class_values = [CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.masks_fps:
            mask = cv2.imread(self.masks_fps[i], 0)
            
            # extract certain classes from mask (e.g. cars)
            masks = [(mask == v) for v in self.class_values]
            mask = np.stack(masks, axis=-1).astype('float')
            
            # apply augmentations
            if self.augmentation:
                sample = self.augmentation(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']
                
            # apply preprocessing
            if self.preprocessing:
                sample = self.preprocessing(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']
                
            return image, mask
        
        else:
            # apply augmentations
            if self.augmentation:
                sample = self.augmentation(image=image)
                image = sample['image']
                
            # apply preprocessing
            if self.preprocessing:
                sample = self.preprocessing(image=image)
                image = sample['image']
            
            return image
        
        
    def __len__(self):
        return len(self.ids)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')
  

def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        albu.RandomCrop(height=320, width=320, always_apply=True),

        albu.GaussNoise(p=0.2),
        albu.Perspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightnessContrast(p=1), #RandomBrightness(p=1),
                albu.RandomGamma(p=1)
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.Sharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomBrightnessContrast(p=1), #RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    '''Add paddings to make image shape divisible by 32'''
    test_transform = [
        albu.PadIfNeeded(384, 480)
    ]
    return albu.Compose(test_transform)


def get_preprocessing(preprocessing_fn):
    '''Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    '''
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)



def get_preprocessing_no_mask(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor)
    ]
    
    return albu.Compose(_transform)


# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


def create_subfolders(PATH):
  splitted_folders_exist = os.path.exists(PATH + 'tr/') and os.path.exists(PATH + 'va/') and os.path.exists(PATH + 'te/')
  if not splitted_folders_exist:
      os.mkdir(PATH + 'tr/')
      os.mkdir(PATH + 'va/')
      os.mkdir(PATH + 'te/')


def fns_splitting(fns, size_tr, size_va, SEED):
  N = len(fns)
  n_tr = int(N*size_tr)
  n_va = int(N*size_va)
  n_te = N - n_tr - n_va

  random.seed(SEED)
  random.shuffle(fns)
  fns_tr = fns[:n_tr]
  fns_va = fns[n_tr:(n_tr+n_va)]
  fns_te = fns[-n_te:]

  assert len(set(fns_tr) & set(fns_va)) == 0
  assert len(set(fns_tr) & set(fns_te)) == 0
  assert len(set(fns_va) & set(fns_te)) == 0

  return fns_tr, fns_va, fns_te


def fns2subfolders(PATH, size_tr, size_va, SEED):
  fns = os.listdir(PATH)
  fns.sort()
    
  if len(fns) > 3:
      fns_tr, fns_va, fns_te = fns_splitting(fns, size_tr, size_va, SEED)
      for fn in fns_tr:
          shutil.move(PATH + fn, PATH + 'tr/' + fn)
      for fn in fns_va:
          shutil.move(PATH + fn, PATH + 'va/' + fn)
      for fn in fns_te:
          shutil.move(PATH + fn, PATH + 'te/' + fn)
      print('Finish moving files!')
  else:
      print('No file needs to be moved.')
