import SimpleITK as sitk
import nibabel as nib
import numpy as np
from dipy.align.reslice import reslice
from typing import Union, Tuple, List
from dcmstack import reorder_voxels
# def load_nib(fpath):
#     """
#     Load nifti image
#     :param fpath: path of nifti file
#     """
#     im = sitk.ReadImage(fpath, sitk.sitkInt16)
#     return im


def load_sitkImage(fpath):
    return sitk.ReadImage(fpath, sitk.sitkInt16)


def load_nib(fpath):
    """
    Load nifti image
    :param fpath: path of nifti file
    """
    im = nib.load(fpath)

    return im

# def resample_nib(im, )


def resample_nib(im, new_spacing=(1, 1, 1), order=0):
    """
    Resample nifti voxel array and corresponding affine
    :param im: nifti image
    :param new_spacing: new voxel size
    :param order: order of interpolation for resampling/reslicing, 0 nearest interpolation, 1 trilinear etc.
    :return new_im: resampled nifti image
    """
    header = im.header
    vox_zooms = header.get_zooms()
    vox_arr = im.get_fdata()
    vox_affine = im.affine
    # resample using DIPY.ALIGN
    if isinstance(new_spacing, int) or isinstance(new_spacing, float):
        new_spacing = (new_spacing[0], new_spacing[1], new_spacing[2])
    new_vox_arr, new_vox_affine = reslice(vox_arr, vox_affine, vox_zooms, new_spacing, order=order)
    # create resampled image
    new_im = nib.Nifti1Image(new_vox_arr, new_vox_affine, header)
    return new_im

def reorient_nib(im: Union[nib.Nifti1Image, nib.Nifti2Image], new_orient: str = 'RIP', datatype: type = np.float64) -> Union[nib.Nifti1Image, nib.Nifti2Image]:
    """
    Reorient nifti voxel array and corresponding affine
    ----------
    :param im: nifti image
    :param new_orient: A three character code specifying the desired starting point for rows, columns, and slices
                       in terms of the orthogonal axes of patient space:
                       (L)eft, (R)ight, (A)nterior, (P)osterior, (S)uperior, and (I)nferior
    :param datatype: specify the data type of return image
    :return new_im: reoriented nifti image
    """

    header = im.header
    vox_arr = im.get_fdata()
    vox_affine = im.affine
    # reorient using DCMStack
    new_vox_arr, new_vox_affine, _, _ = reorder_voxels(vox_arr, vox_affine, new_orient)
    # specify datatype
    new_vox_arr = new_vox_arr.astype(datatype)
    header.set_data_dtype(datatype)
    # create reoriented NIB image
    new_im = nib.Nifti1Image(new_vox_arr, new_vox_affine, header)

    return new_im


def transpose_compatible(arr, direction):
    """
    Transpose array to a compatible direction
    :param arr: numpy array
    :param direction: 'asl_to_np' or 'np_to_asl' only
    :return arr: transposed array
    """
    if direction == 'asl_to_np':
        arr = arr.transpose([1, 0, 2])[:, :, ::-1]
    if direction == 'np_to_asl':
        arr = arr[:, :, ::-1].transpose([1, 0, 2])
    else:
        'Direction can only be ASL to Anjany\'s numpy indexing or the other way around!'

    return arr



def resize_image_itk(itkimage, new_spacing=[1, 1, 1], is_mask=False):
    """
    将体数据重采样的指定的spacing大小\n
    paras：
    outpacing：指定的spacing，例如[1,1,1]
    vol：sitk读取的image信息，这里是体数据\n
    return：重采样后的数据
    """
    resampler = sitk.ResampleImageFilter()

    origin_size = itkimage.GetSize()
    origin_spacing = itkimage.GetSpacing()
    origin_origin = itkimage.GetOrigin()  # 目标的起点 [x,y,z]
    origin_direction = itkimage.GetDirection()  # 目标的方向 [冠,矢,横]=[z,y,x]

    new_size = [0, 0, 0]
    new_size[0] = round(origin_size[0] * origin_spacing[0] / new_spacing[0])
    new_size[1] = round(origin_size[1] * origin_spacing[1] / new_spacing[1])
    new_size[2] = round(origin_size[2] * origin_spacing[2] / new_spacing[2])

    # 设定重采样的一些参数
    resampler.SetReferenceImage(itkimage)  # 需要重新采样的目标图像
    # 设置目标图像的信息
    resampler.SetSize(new_size)  # 目标图像大小
    # resampler.SetOutputOrigin(origin_origin)

    DEFAULT_DIRECTION = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    # resampler.SetOutputDirection(DEFAULT_DIRECTION)
    resampler.SetOutputDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    resampler.SetOutputSpacing(new_spacing)
    # 根据需要重采样图像的情况设置不同的dype
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    if not is_mask:
        resampler.SetInterpolator(sitk.sitkLinear)  # CT使用线性插值法，mask使用临近插值
    else:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    return resampler.Execute(itkimage)




def restore_image_itk(itkimage, origin_image, is_mask=False):
    resampler = sitk.ResampleImageFilter()

    origin_size = origin_image.GetSize()
    origin_spacing = origin_image.GetSpacing()
    origin_origin = origin_image.GetOrigin()
    origin_direction = origin_image.GetDirection()

    # 设定重采样的一些参数
    resampler.SetReferenceImage(itkimage)  # 需要重新采样的目标图像
    # 设置目标图像的信息
    resampler.SetSize(origin_size)  # 目标图像大小
    resampler.SetOutputOrigin(origin_origin)
    resampler.SetOutputDirection(origin_direction)
    resampler.SetOutputSpacing(origin_spacing)
    # 根据需要重采样图像的情况设置不同的dype
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    if not is_mask:
        resampler.SetInterpolator(sitk.sitkLinear)  # CT使用线性插值法，mask使用临近插值
    else:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    return resampler.Execute(itkimage)
