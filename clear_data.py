import SimpleITK as sitk
import numpy as np
import os


def load_itkimage(fpath, isMask=False):
    if not isMask:
        return sitk.ReadImage(fpath, sitk.sitkInt16)
    return sitk.ReadImage(fpath, sitk.sitkInt8)


im_save_dir = 'E:\\Data\\spine\\test_4\\mask'
msk_save_dir = 'E:\\Data\\spine\\test_4\\mask'

file_list = os.listdir(im_save_dir)
for filename in file_list:
    file_path = os.path.join(im_save_dir, filename)
    im = load_itkimage(file_path, False)
    h = im.GetSize()[0]
    w = im.GetSize()[1]
    # print(str(h) + '----' + str(w))
    if h != w:
        print(filename)
        os.remove(file_path)
