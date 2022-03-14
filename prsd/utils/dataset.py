import os
import logging
import numpy as np
import torch
from torch.utils.data import Dataset
from osgeo import gdal_array
# from ..io import imread
# from torch.utils.data import DataLoader,random_split


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir

        self.ids = [os.path.splitext(file)[0] for file in os.listdir(imgs_dir)
                    if not file.startswith('.')]
        self.imgs = sorted(os.listdir(self.imgs_dir))
        self.masks = sorted(os.listdir(self.masks_dir))
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        # idx = self.ids[i]
        # mask_file = glob(self.masks_dir + idx + '.*')
        # img_file = glob(self.imgs_dir + idx + '.*')

        #img_file = glob(self.imgs_dir + idx + '.*')
        
        img_file = os.path.join(self.imgs_dir,self.imgs[i])
        mask_file = os.path.join(self.masks_dir,self.masks[i])

        # assert len(mask_file) == 1, \
        #     f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        # assert len(img_file) == 1, \
        #     f'Either no image or multiple images found for the ID {idx}: {img_file}'
        # assert len(mask_file) == len(img_file), \
        #     f'The number of image and mask should be the same size, but are {len(img_file)} and {len(mask_file)}'
        #print(mask_file[0],img_file[0])
        
        
        img = gdal_array.LoadFile(img_file)
        #img = img/img.max()
        mask = gdal_array.LoadFile(mask_file)
        
        # mask,_,_ = imread(mask_file)
        # img,_,_ = imread(img_file)
        
        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }

if __name__ == '__main__':
    dir_img = "D:/ShanDong/high_quality_dataset256/img/"
    dir_mask = "D:/ShanDong/high_quality_dataset256/gt/"
    
    dataset = BasicDataset(dir_img, dir_mask)
    

