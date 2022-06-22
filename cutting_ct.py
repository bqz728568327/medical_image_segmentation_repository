import os
import torch
import numpy as np
import SimpleITK as sitk

def msk_2_box(msk, threshold):
    """
    Compute the 3d bounding box coordinates from the localization heatmap
    :param msk: 3d spine localization heatmap
    :param threshold: intensity (probability) threshold
    :return: 3d bounding box coordinates
    """
    msk_temp = np.copy(msk)
    msk_temp[msk < threshold] = 0
    msk_temp = np.squeeze(msk_temp)
    nzs = np.nonzero(msk_temp)

    # print(nzs[0])
    if len(nzs[0]) > 0:
        d_min = np.amin(nzs[0])
        h_min = np.amin(nzs[1])
        w_min = np.amin(nzs[2])
        d_max = np.amax(nzs[0])
        h_max = np.amax(nzs[1])
        w_max = np.amax(nzs[2])
        # print(h_min)
        # print(h_max)
        # d h w
        return [d_min, d_max, h_min, h_max, w_min, w_max]
    else:
        h, w, d = msk_temp.shape

    return [0, h, 0, w, 0, d]


im_root_path = 'E:\\Data\\spine\\test\\img'
msk_root_path = 'E:\\Data\\spine\\test\\mask'

im_save_dir = 'E:\\Data\\spine\\cutting\\img'
msk_save_dir = 'E:\\Data\\spine\\cutting\\mask'


im_list = os.listdir(im_root_path)
msk_list = os.listdir(msk_root_path)
for i in range(len(im_list)):
    im_filename = im_list[i]
    msk_filename = msk_list[i]
    im_path = os.path.join(im_root_path, im_filename)
    msk_path = os.path.join(msk_root_path, msk_filename)

    im = sitk.ReadImage(im_path, sitk.sitkInt16)
    msk = sitk.ReadImage(msk_path, sitk.sitkInt8)
    im_arr = sitk.GetArrayViewFromImage(im)
    msk_arr = sitk.GetArrayFromImage(msk)


    bounding_box = msk_2_box(msk_arr, 0.5)
    print(bounding_box)
    # print(im.shape)
    new_im = im_arr[bounding_box[0]:bounding_box[1], bounding_box[2]:bounding_box[3], bounding_box[4]:bounding_box[5]]
    new_msk = msk_arr[bounding_box[0]:bounding_box[1], bounding_box[2]:bounding_box[3], bounding_box[4]:bounding_box[5]]
    # msk 必须为int8类型
    new_msk = new_msk.astype(np.int8)

    im_out = sitk.GetImageFromArray(new_im)
    msk_out = sitk.GetImageFromArray(new_msk)

    im_save_path = os.path.join(im_save_dir, im_filename)
    msk_save_path = os.path.join(msk_save_dir, msk_filename)

    
    sitk.WriteImage(im_out, im_save_path)
    sitk.WriteImage(msk_out, msk_save_path)

    # lower = [bounding_box[0],bounding_box[2],bounding_box[4]]
    # upper = [bounding_box[1], bounding_box[3], bounding_box[5]]
    # print(lower)
    # print(upper)



    # sitk.WriteImage(croped_img,"../cropedImage.nii.gz")
