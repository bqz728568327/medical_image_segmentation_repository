import os
import SimpleITK as sitk
import numpy
import numpy as np
import pandas as pd
import torch
import torch.utils.data.dataset as Dataset
from torchvision.io import read_image
import random
import torchvision.transforms as transforms
random.seed(1)
import scipy.ndimage as ndimage
import util
from matplotlib import pyplot as plt

class ImageDataset(torch.nn.Module):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.image_files = [image for image in os.listdir(img_dir + os.sep + 'img')]
        self.label_files = [label for label in os.listdir(img_dir + os.sep + 'mask')]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_files)

    def load_sitkImage(self, im_path, msk_path):
        # return shape (Z X Y)
        # return data type numpy.ndarray
        img = sitk.ReadImage(im_path, sitk.sitkInt16)
        img_array = sitk.GetArrayFromImage(img)

        label = sitk.ReadImage(msk_path, sitk.sitkInt8)
        label_array = sitk.GetArrayFromImage(label)

        return img_array, label_array




    def img_transform(self, img, label):
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            # transforms.Normalize(norm_mean, norm_std),
        ])

        valid_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            # transforms.Normalize(norm_mean, norm_std),
        ])

        img = torch.from_numpy(train_transform(img)).float()
        label = torch.from_numpy(valid_transform(label)).float()

        return img, label

    def normalize_data(self, data, mean, std):
        # data：[4,144,144,144]
        # C, D, W, H = data.shape
        # temp_axis = np.zeros(C, D, W, H)
        # temp_axis = temp_axis[]
        # a = mean[:, np.newaxis, np.newaxis, np.newaxis]
        # print(data)
        data = data.astype(np.float)
        data -= mean
        data /= std
        # print(data)
        # data -= mean[:, np.newaxis, np.newaxis, np.newaxis]
        # data /= std[:, np.newaxis, np.newaxis, np.newaxis]
        return data



    def data_processing(self, im, msk):
        # 截断
        im = im / 2048
        im[im > 1] = 1
        im[im < -1] = -1

        im = im[np.newaxis, ...]

        msk[msk > 1] = 1

        return im, msk



    def __getitem__(self, idx):
        im_path = os.path.join(self.img_dir, 'img', self.image_files[idx])
        msk_path = os.path.join(self.img_dir, 'mask', self.label_files[idx])


        # 读取数据 得到narray类型的数组
        im, msk = self.load_sitkImage(im_path, msk_path)

        # 这部分的数据需要处理
        if 'verse' in im_path:
            im = im[:, ::-1, :]
            msk = msk[:, ::-1, :]

        # D H W => C D H W
        im, msk = self.data_processing(im, msk)



        return {
            'im': torch.as_tensor(im.copy()).float().contiguous(),
            'msk': torch.as_tensor(msk.copy()).float().contiguous()
        }


if __name__ == '__main__':
    img_dir = 'E:\\Data\\spine\\test'
    dataset = ImageDataset(img_dir=img_dir)
    for i in range(50, 51):
        data = dataset.__getitem__(i)
        im = data['im']
        msk = data['msk']
        print(im.shape)
