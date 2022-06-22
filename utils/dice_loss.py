import torch


def dice_coeff(pred, target):
    smooth = 1e-8
    # num = pred.size()
    # print(num)

    # print('loss')
    # print(pred.shape)
    # print(target.shape)

    m1 = pred.flatten() # Flatten
    m2 = target.flatten()  # Flatten
    # print(m1.shape)
    # print(m2.shape)
    intersection = (m1 * m2).sum()
    dice = 1 - (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)
    return dice