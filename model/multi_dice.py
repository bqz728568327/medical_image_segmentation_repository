import numpy as np


def cal_subject_level_dice(prediction, target, class_num=2):# class_num是你分割的目标的类别个数
    '''
    step1: calculate the dice of each category
    step2: remove the dice of the empty category and background, and then calculate the mean of the remaining dices.
    :param prediction: the automated segmentation result, a numpy array with shape of (h, w, d)
    :param target: the ground truth mask, a numpy array with shape of (h, w, d)
    :param class_num: total number of categories
    :return:
    '''
    eps = 1e-10
    empty_value = -1.0
    dscs = empty_value * np.ones((class_num), dtype=np.float32)
    for i in range(0, class_num):
        if i not in target and i not in prediction:
            continue
        target_per_class = np.where(target == i, 1, 0).astype(np.float32)
        prediction_per_class = np.where(prediction == i, 1, 0).astype(np.float32)

        tp = np.sum(prediction_per_class * target_per_class)
        fp = np.sum(prediction_per_class) - tp
        fn = np.sum(target_per_class) - tp
        dsc = 2 * tp / (2 * tp + fp + fn + eps)
        dscs[i] = dsc
        print(dsc)
    dscs = np.where(dscs == -1.0, np.nan, dscs)
    subject_level_dice = np.nanmean(dscs[1:])
    return subject_level_dice

def dice_coeff(pred, target):
    smooth = 1e-8
    # num = pred.size()
    # print(num)
    m1 = pred.flatten()  # Flatten
    m2 = target.flatten()  # Flatten
    intersection = (m1 * m2).sum()
    print((2. * intersection + smooth))
    print((m1.sum() + m2.sum() + smooth))
    return 1 - (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

if __name__ == '__main__':
    im = np.random.random((50,10))
    msk = np.random.random((50,10))
    print(im)
    print(msk)
    di = dice_coeff(im, msk)
    print(di)
    # print(msk)
    im = im * 10
    # print(im.astype(int))
    # print(msk)
    # dice = cal_subject_level_dice(im.astype(int), msk, 10)
    # print(dice)