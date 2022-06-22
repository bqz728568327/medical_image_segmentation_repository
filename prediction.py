import torch
from model.unet3d import UNet3D
from utils.dataloader import ImageDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import time
import SimpleITK as sitk
from model.localization_network import LocalizationNet
import matplotlib.pyplot as plt
model_save_path = 'checkpoints/localization_model2.pth'
net = LocalizationNet()
net.cuda()
net.load_state_dict(torch.load(model_save_path))
# print(net)

img_path = 'E:\\Data\\spine\\test_4\\img\\sub-gl003_dir-ax_ct.nii.gz'
# img_path = 'E:\\Data\\spine\\BCA\\img\\N037_new_spacing_2.nii.gz'
msk_path = 'E:\\Data\\spine\\test_4\\mask\\sub-gl003_dir-ax_seg-vert_msk.nii.gz'

# mask_path = 'E:\\Data\\spine\\test\\mask\\sub-gl003_dir-ax_seg-vert_msk.nii.gz'
im = sitk.ReadImage(img_path, sitk.sitkInt16)
img_array = sitk.GetArrayFromImage(im)
plt.imshow(img_array[0, :, :], cmap='gray')
plt.show()
im = img_array[np.newaxis, np.newaxis, ...]
im = im / 2048
im[im > 1] = 1
im[im < -1] = -1
im = torch.as_tensor(im.copy()).float().contiguous().to(device='cuda', dtype=torch.float32)
pred = net(im)
print(pred.shape)
pred_arr = pred.cpu().detach().numpy()
pred_arr[pred_arr >= 0.5] = 1
pred_arr[pred_arr < 0.5] = 0
pred_arr = pred_arr.astype(np.int8)
print(pred_arr.dtype)
out = sitk.GetImageFromArray(pred_arr)
# out = out.astype(np.int16)
sitk.WriteImage(out, 'C:\\Users\\Bai\\Desktop\\simpleitk_save.nii.gz')

from model.multi_dice import dice_coeff

pred_arr = pred_arr.squeeze()
msk = sitk.ReadImage(msk_path, sitk.sitkInt8)
msk_array = sitk.GetArrayFromImage(msk)

plt.imshow(msk_array[100, :, :], cmap='gray')
plt.show()

plt.imshow(pred_arr[100, :, :], cmap='gray')
plt.show()

print(dice_coeff(pred.cpu().detach().numpy(), msk_array))




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

bounding_box = msk_2_box(pred_arr, 0.5)




# train_loader = DataLoader(dataset , shuffle=False, batch_size=1,  num_workers=0, pin_memory=True)
# criterion = nn.CrossEntropyLoss()
# for batch in train_loader:
#     # net.eval()
#     images = batch['image']
#     true_masks = batch['mask']
#     images = images.to(device='cuda', dtype=torch.float32)
#     true_masks = true_masks.to(device='cuda', dtype=torch.float32)
#
#     pred_mask = net(images)
#
#     pred_mask = F.softmax(pred_mask, dim=2)
#     pred_mask = torch.max(pred_mask, dim=1).values
#     loss = criterion(pred_mask.float(), true_masks.float())
#     # print(loss)
#     print(torch.min(pred_mask))
#     result = pred_mask.cpu().detach().numpy() * 15
#
#     result = np.floor(result)
#
#     print(np.max(result))
#     result = np.squeeze(result)
#     result = result.astype(np.int16)
#     print(result.shape)
#
#
#     out = sitk.GetImageFromArray(result)
#     # print(out.GetSize())
#     out = out.astype(np.int16)
#     sitk.WriteImage(out, 'simpleitk_save.nii.gz')
#
#     # time.sleep()
#

