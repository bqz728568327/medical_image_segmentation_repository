# import os
# import SimpleITK as sitk
#
#
# ct_dir = 'E:\Data\Synapse multi-organ CT\Abdomen\RawData\Training\img'
# seg_dir = 'E:\Data\Synapse multi-organ CT\Abdomen\RawData\Training\label'
# i=1
# for ct_file in os.listdir(ct_dir):
#     print(i)
#     i = i+1
#     # 将CT和金标准入读内存
#     ct = sitk.ReadImage(os.path.join(ct_dir, ct_file), sitk.sitkInt16)
#     # ct_array:(629, 512, 512)
#     # 注意读取出来是z y x，即切片数量在最前
#     # 而 origin和position读取出来的是 x y z
#     ct_array = sitk.GetArrayFromImage(ct)
#     # vol_values=np.unique(ct_array) 有2708个值
#
#     seg = sitk.ReadImage(os.path.join(seg_dir, ct_file.replace('img', 'label')), sitk.sitkInt8)
#     seg_array = sitk.GetArrayFromImage(seg)
#     print(seg_array.shape)
#     print(type(seg_array))


from model.unet3d import UNet3D
import torch
data = torch.randn((1, 1, 66, 256, 256))
print(data.shape)

model = UNet3D(in_channels=1, out_channels=13)
print('start')
pre = model(data)
print(pre.shape)