import SimpleITK as sitk
import numpy as np
import os
import util
from matplotlib import pyplot as plt
import nibabel as nib
from dipy.align.reslice import reslice

def resize_image_itk(itkimage, new_size, is_mask=False):
    resampler = sitk.ResampleImageFilter()

    origin_size = itkimage.GetSize()
    origin_spacing = itkimage.GetSpacing()

    new_size = np.array(new_size, float)
    factor = origin_size / new_size
    new_spacing = origin_spacing * factor
    new_size = new_size.astype(np.int16)

    resampler.SetReferenceImage(itkimage)  # 需要重新采样的目标图像
    resampler.SetSize(new_size.tolist())
    resampler.SetOutputSpacing(new_spacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    if not is_mask:
        resampler.SetInterpolator(sitk.sitkLinear)  # CT使用线性插值法，mask使用临近插值
    else:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    return resampler.Execute(itkimage)

# def resize_image_itk(itkimage, new_size):


def pre_processing(dir_img, dir_label, dir_save, new_size=(512,512,64)):
    if not os.path.exists(dir_img):
        raise Exception('Not exists Path')
    file_list = os.listdir(dir_img)
    for filename in file_list:
        img_path = os.path.join(dir_img, filename)
        img = sitk.ReadImage(img_path, sitk.sitkInt16)
        img = resize_image_itk(img, new_size)
        print(img.GetSize())
        sitk.WriteImage(img, os.path.join(dir_save, 'img', filename))
        label_path = os.path.join(dir_label, filename)
        label = sitk.ReadImage(label_path, sitk.sitkInt8)
        label = resize_image_itk(label, new_size, is_mask=True)
        print(label.GetSize())

        sitk.WriteImage(label, os.path.join(dir_save, 'label' ,filename))

def load_msk(fpath):
    im = util.load_nib(fpath)
    return im

def load_img(fpath, scale):
    """
    Load, reorient, resample, transpose image for inference
    :param scale: downsampling to new voxel size in mm
    :return: preprocessed image array
    """
    im = util.load_nib(fpath)
    print(im.get_fdata().shape)
    #self.orig_orientation = nib.aff2axcodes(im.affine)
    im = util.reorient_nib(im, new_orient='PIR')
    im = util.resample_nib(im, new_spacing=(scale, scale, scale), order=3)

    im_arr = im.get_fdata()
    print(im_arr.shape)
    # im_arr = util.transpose_compatible(im_arr, direction='asl_to_np')
    im_arr = im_arr / 2048.
    im_arr[im_arr < -1] = -1
    im_arr[im_arr > 1] = 1

    return im_arr, im.affine, im.header

def load_label(fpath, scale):
    """
    Load, reorient, resample, transpose image for inference
    :param scale: downsampling to new voxel size in mm
    :return: preprocessed image array
    """
    im = util.load_nib(fpath)
    return im


def save_mask_nib(im_orig, mask_1mm):
    """
    Save segmentation mask based on the original image nifti attributes
    :param im_orig: original image nifti
    :param im_1mm: image nifti at 1 mm
    :param msk_1mm: segmentation mask array at 1 mm
    """
    new_spacing = im_orig.header.get_zooms()
    mask_arr = mask_1mm.get_fdata()
    mask_zooms = mask_1mm.header.get_zooms()
    msk_affine = mask_1mm.affine
    #msk_arr = utils.transpose_compatible(msk_1mm, direction='np_to_asl')
    #msk_zooms = (1., 1., 1.)
    # resample using DIPY.ALIGN
    new_vox_arr, new_vox_affine = reslice(mask_arr, msk_affine, mask_zooms, new_spacing, order=0)
    # adjust for differences in last dimension
    if im_orig.get_fdata().shape != new_vox_arr.shape:
        if im_orig.get_fdata().shape[2] < new_vox_arr.shape[2]:
            new_vox_corrected = new_vox_arr[:,:,1:]
        elif im_orig.get_fdata().shape[2] > new_vox_arr.shape[2]:
            new_vox_corrected = np.zeros(im_orig.get_fdata().shape)
            new_vox_corrected[:,:,1:] = new_vox_arr
        new_vox_arr = new_vox_corrected

    new_vox_arr = new_vox_arr.astype(np.int8)
    # create resampled image
    im_orig.set_sform(new_vox_affine)
    new_im = nib.Nifti1Image(new_vox_arr, new_vox_affine, im_orig.header)
    return new_im
    # nib.save(new_im, self.save_dir)
    # print('Segmentation saved at: ', self.save_dir)

if __name__ == '__main__':
    # dir_img = 'E:\\Data\\AMOS22\\AMOS22\\smallset\\valid\\img'
    # dir_label = 'E:\\Data\\AMOS22\\AMOS22\\smallset\\valid\\label'
    # dir_save = 'E:\\Data\\AMOS22\\AMOS22\\smallset\\valid_processed\\'
    # pre_processing(dir_img, dir_label, dir_save, (256,256,64))

    root_dir = 'E:\\Data\\spine\\train\\rawdata'
    msk_root_dir = 'E:\\Data\\spine\\train\\derivatives'
    im_save_dir = 'E:\\Data\\spine\\test\\img'
    msk_save_dir = 'E:\\Data\\spine\\test\\mask'
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


        # print(file_path)
        im, im_affine, im_header = load_img(im_path, 1)
        print(msk_path)
        msk = load_msk(msk_path)

        new_im = nib.Nifti1Image(im, im_affine, im_header)
        nib.save(new_im, save_ct_path)


        # new_msk = save_mask_nib(new_im, msk)
        #
        #
        # nib.save(new_msk, save_msk_path)
        print(new_im.get_fdata().shape)
        # print(new_msk.get_fdata().shape)

        # new_im = nib.Nifti1Image(im_localize, im_affine, im_header)
        # nib.save(new_im, save_msk_path)
        # nib.save(im_localize, save_msk_path)



        # plt.imshow(im_localize[50, :, :], cmap='gray')
        # plt.imshow(im_localize[50, :, :], cmap='gray')
        # plt.show()

        # file_path = os.path.join(dir_path, filename)


