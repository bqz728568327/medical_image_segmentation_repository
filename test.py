# import SimpleITK as sitk
# # import cv2
# # import matplotlib.pyplot as plt
# # import numpy as np
# #
# import numpy as np
# import util
# from matplotlib import pyplot as plt
# img_path = 'E:\\Data\\spine\\test\\img\\sub-verse535_dir-iso_ct.nii.gz'
# # label_path = 'E:\\Data\\spine\\test\\sub-verse535_dir-iso_seg-vert_msk.nii.gz'
#
# # img_path_or = 'E:\\Data\\spine\\test\\img\\gl003.nii.gz'
#
#
# im = util.load_sitkImage(img_path)
# #
# # im = util.load_nib(img_path)
# # msk = util.load_nib(label_path)
# # print(im_or.GetDirection())
# print(im.GetSize())
#
# data = sitk.GetArrayViewFromImage(im)
# # im[:, :, :] = data[:, ::-1, :]
#
# print(data.shape)
# new_data = data[:, ::-1, :]
# print(new_data.shape)
# print()
# # im_restore = util.restore_image_itk(im, im_or, is_mask=False)
# # print(im_restore.GetDirection())
# #
# # im_restore = sitk.GetArrayFromImage(im_restore)
# #
# # arr1 = data[::-1]
# plt.imshow(new_data[494, :, :], cmap='gray')
# # D H W
# # plt.imshow(data[100, ::-1, :], cmap='gray')
# #
# plt.show()



#
# print('im and msk Shape')
# print(im.GetSize())
# print(im.GetSpacing())
# print(im_or.GetDirection())
# print(im.GetDirection())
# print(msk.GetSize())
# print(msk.GetSpacing())
# # print(msk.GetDirection())
#
# resized_im = util.resize_image_itk(im, [1,1,1], False)
# resized_msk = util.resize_image_itk(msk, [1,1,1], True)
#
# print('resized im and msk Shape')
# print(resized_im.GetSize())
# print(resized_im.GetSpacing())
# # print(resized_im.GetDirection())
# print(resized_msk.GetSize())
# print(resized_msk.GetSpacing())
# print(resized_msk.GetDirection())
#
# restore_im = util.restore_image_itk(resized_im, im, False)
# restore_msk = util.restore_image_itk(resized_msk, msk, True)
#
# print('restored im and msk Shape')
# print(restore_im.GetSize())
# print(restore_im.GetSpacing())
# # print(restore_im.GetDirection())
# print(restore_msk.GetSize())
# print(restore_msk.GetSpacing())
# print(restore_msk.GetDirection())
#
# from model.localization_network import LocalizationNet
# import torch
#
# localization_net = LocalizationNet()
#
# data = torch.rand([1,1,268,149,149])
#
# pred = localization_net(data)
# print(pred.shape)

import torch
import torch.nn.functional as F

input = torch.randn(1,1,3,3,4)
print(input)

b = F.softmax(input, dim=0)  # 按列SoftMax,列和为1
print(b)

c = F.softmax(input, dim=1)  # 按行SoftMax,行和为1
print(c)

d = F.softmax(input, dim=2)  # 按列取max,
print(d)

e = F.softmax(input, dim=3)  # 按列取max,
print(e)

f = F.softmax(input, dim=4)  # 按列取max,
print(f)






