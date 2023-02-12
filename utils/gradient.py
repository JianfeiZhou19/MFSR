import numpy as np
import cv2

def conv(image, kernel):
    """卷积的一个实现.

    对于任意一个像素点，该本版采用了点积运算以及 np.sum()来进行快速的加权求和

    Args:
        图像: 尺寸为(Hi, Wi)的numpy数组.
        卷积核(kernel): 尺寸为(Hk, Wk)的numpy数组.

    Returns:
        out: 尺寸为(Hi, Wi)的numpy数组.
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0,pad_width0),(pad_width1,pad_width1))
    padded = np.pad(image, pad_width, mode='edge')

    for i in range(Hi):
        for j in range(Wi):
            out[i][j] = (padded[i:i + Hk,j:j + Wk] * kernel).sum()

    return out

if __name__ == '__main__':
    img = cv2.imread("/vepfs/Perception/perception-users/jianfei/self_exp/lightfield/log/SR_5x5_2x/MEG_Net/MEG_Net/results/VAL_epoch_01/HCI_new/origami/origami_CenterView.bmp", 0)
    kernel1 = np.array([[1, 0, -1], 
                       [2, 0, -2],
                       [1, 0, -1]])
    # kernel1 = np.array([[0, -1, 0],
    #                 [0, 0, 0],
    #                 [0, 1, 0]])
    kernel2 = np.array([[1, 2, 1], 
                        [0, 0, 0],
                        [-1, -2, -1]])
    res1 = conv(img, kernel1)
    res2 = conv(img, kernel2)
    res = np.clip(res1+res2, 0, 255)

    cv2.imwrite("test.jpg", res)