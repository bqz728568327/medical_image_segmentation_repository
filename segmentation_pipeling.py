import matplotlib.pyplot as plt
import numpy as np
import time
import SimpleITK as sitk
from model.localization_network import LocalizationNet
import torch

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

    if len(nzs[0]) > 0:
        d_min = np.amin(nzs[0])
        h_min = np.amin(nzs[1])
        w_min = np.amin(nzs[2])
        d_max = np.amax(nzs[0])
        h_max = np.amax(nzs[1])
        w_max = np.amax(nzs[2])
        # d h w
        return [d_min, d_max, h_min, h_max, w_min, w_max]
    else:
        h, w, d = msk_temp.shape

    return [0, h, 0, w, 0, d]

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

img_path = 'E:\\Data\\spine\\BCA\\img\\N037.nii.gz'


if __name__ == '__main__':
    localization_model_save_path = 'checkpoints/localization_model.pth'
    classification_model_save_path = 'checkpoints/classification_model.pth'
    localiztion_net = LocalizationNet()
    classification_net = LocalizationNet()

    localiztion_net.cuda()
    localiztion_net.load_state_dict(torch.load(localization_model_save_path))
    classification_net.cuda()
    classification_net.load_state_dict(torch.load(classification_model_save_path))


    origin_im = sitk.ReadImage(img_path, sitk.sitkInt16)
    origin_im_arr = sitk.GetArrayFromImage(origin_im)

    im = resize_im_itk_spacing(origin_im, [2,2,2], is_mask=False)
    img_array = sitk.GetArrayFromImage(im)

    im = img_array[np.newaxis, np.newaxis, ...]
    im = im / 2048
    im[im > 1] = 1
    im[im < -1] = -1
    im = torch.as_tensor(im.copy()).float().contiguous().to(device='cuda', dtype=torch.float32)
    localization_pred = localiztion_net(im)

    localization_pred_arr = localization_pred.cpu().detach().numpy()

    bounding_box = msk_2_box(localization_pred_arr, 0.5)


    print(bounding_box)

    # print(img_array.shape)
    # print(localization_pred.shape)
    # print(bounding_box)
    new_im = origin_im_arr[bounding_box[0]*2:bounding_box[1]*2, bounding_box[2]*2:bounding_box[3]*2, bounding_box[4]*2:bounding_box[5]*2]
    print(new_im.shape)
    new_im = new_im[np.newaxis, np.newaxis, ...]
    new_im_tensor = torch.as_tensor(new_im.copy()).float().contiguous().to(device='cuda', dtype=torch.float32)
    classification_pred = classification_net(new_im_tensor)
    # print(classification_pred.shape)
    # plt.imshow(img_array[0, :, :], cmap='gray')
    # plt.show()
    # print(classification_pred.shape)
    classification_pred_arr = classification_pred.cpu().detach().numpy()
    classification_pred_arr = classification_pred_arr.squeeze() * 25
    classification_pred_arr = classification_pred_arr.astype(np.int8)
    # print(np.max(classification_pred_arr))

    # new_mask_out = sitk.GetImageFromArray(classification_pred_arr)
    # print(new_mask_out.GetSize())
    # print(origin_im.GetSize())

    msk = np.ones(origin_im_arr.shape, dtype=np.int8)
    print(msk.shape)
    msk[bounding_box[0]*2:bounding_box[1]*2, bounding_box[2]*2:bounding_box[3]*2, bounding_box[4]*2:bounding_box[5]*2] = classification_pred_arr[:,:,:]
    msk = msk.astype(np.int8)
    print(msk.shape)

    msk_save_path = 'E:\\Data\\spine\\BCA\\mask\\N037_mask.nii.gz'
    mask_out = sitk.GetImageFromArray(msk)
    sitk.WriteImage(mask_out, msk_save_path)