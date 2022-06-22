import SimpleITK as sitk
import numpy as np
import os
import util
from matplotlib import pyplot as plt
import nibabel as nib
from dipy.align.reslice import reslice


def resize_im_itk_pixel(itkimage, new_size=(128, 128, 64), is_mask=False):
    resampler = sitk.ResampleImageFilter()

    origin_size = itkimage.GetSize()
    origin_spacing = itkimage.GetSpacing()

    new_size = np.array(new_size, float)
    factor = origin_size / new_size
    new_spacing = origin_spacing * factor
    new_size = new_size.astype(np.int16)

    im_origin = itkimage.GetOrigin()  # 目标的起点 [x,y,z]
    im_direction = itkimage.GetDirection()  # 目标的方向 [冠,矢,横]=[z,y,x]

    # 设置目标图像的信息
    resampler.SetReferenceImage(itkimage)  # 需要重新采样的目标图像
    resampler.SetSize(new_size.tolist())
    resampler.SetOutputSpacing(new_spacing.tolist())
    resampler.SetOutputOrigin(im_origin)
    resampler.SetOutputDirection(im_direction)
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    if not is_mask:
        resampler.SetInterpolator(sitk.sitkLinear)  # CT使用线性插值法，mask使用临近插值
    else:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    return resampler.Execute(itkimage)

def resize_im_itk_spacing(itkimage, new_spacing=[1, 1, 1], is_mask=False):
    """
    将体数据重采样的指定的spacing大小\n
    paras：
    outpacing：指定的spacing，例如[1,1,1]
    vol：sitk读取的image信息，这里是体数据\n
    return：重采样后的数据
    """

    origin_size = itkimage.GetSize()
    origin_spacing = itkimage.GetSpacing()
    im_origin = itkimage.GetOrigin()
    im_direction = itkimage.GetDirection()

    transform = sitk.Transform()
    transform.SetIdentity()

    new_size = [0, 0, 0]
    new_size[0] = round(origin_size[0] * origin_spacing[0] / new_spacing[0])
    new_size[1] = round(origin_size[1] * origin_spacing[1] / new_spacing[1])
    new_size[2] = round(origin_size[2] * origin_spacing[2] / new_spacing[2])

    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(transform)
    resampler.SetReferenceImage(itkimage)  # 需要重新采样的目标图像
    resampler.SetSize(new_size)  # 目标图像大小
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputOrigin(im_origin)
    resampler.SetOutputDirection(im_direction)
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    if not is_mask:
        resampler.SetInterpolator(sitk.sitkLinear)  # CT使用线性插值法，mask使用临近插值
    else:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    return resampler.Execute(itkimage)

def load_itkimage(fpath, isMask=False):
    if not isMask:
        return sitk.ReadImage(fpath, sitk.sitkInt16)
    return sitk.ReadImage(fpath, sitk.sitkInt8)


def resize_processing(im_path, msk_path, scale=[1,1,1]):

    im = load_itkimage(im_path, isMask=False)
    msk = load_itkimage(msk_path, isMask=True)
    print(im.GetSize())
    resize_im = resize_im_itk_spacing(im, new_spacing=scale, is_mask=False)
    resize_msk = resize_im_itk_pixel(msk, new_size=resize_im.GetSize(), is_mask=True)
    print(resize_im.GetSize())
    return resize_im, resize_msk

def save_itkimage(im, save_im_path, msk, save_msk_path):
    sitk.WriteImage(im, save_im_path)
    sitk.WriteImage(msk, save_msk_path)

if __name__ == '__main__':
    root_dir = 'E:\\Data\\spine\\train\\rawdata'
    msk_root_dir = 'E:\\Data\\spine\\train\\derivatives'
    im_save_dir = 'E:\\Data\\spine\\test_4\\img'
    msk_save_dir = 'E:\\Data\\spine\\test_4\\mask'
    dir_list = os.listdir(root_dir)
    for dirname in dir_list:
        im_dir_path = os.path.join(root_dir, dirname)
        img_list = os.listdir(im_dir_path)
        msk_dir_path = os.path.join(msk_root_dir, dirname)
        msk_list = os.listdir(msk_dir_path)

        img_path = ''
        msk_path = ''
        for img_filename in img_list:
            if 'ct' in img_filename:
                im_path = os.path.join(im_dir_path, img_filename)
                save_ct_path = os.path.join(im_save_dir, img_filename)
        for msk_filename in msk_list:
            if 'msk.nii' in msk_filename:
                msk_path = os.path.join(msk_dir_path, msk_filename)
                save_msk_path = os.path.join(msk_save_dir, msk_filename)

        resize_im, resize_msk = resize_processing(im_path, msk_path, scale=[2,2,2])
        save_itkimage(resize_im, save_ct_path, resize_msk, save_msk_path)

