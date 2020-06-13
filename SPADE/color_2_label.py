"""
function: provide a RGB png image, change it to the label image according to ade20k's labels.
input: RGB img matrix in numpy
output: label img(greyscale img) matrix in numpy
author: zhouhonghong
date: 2020/06/13
"""
import numpy as np
from PIL import Image
import cv2
import os

# pil转numpy
# img = numpy.array(img)
# numpy转pil
# img = Image.fromarray(img.astype('uint8')).convert('RGB')

def color_2_label(input_img):
    #  手动设置以下两列表
    color_list = [[135, 206, 235], [34, 139, 34], [169, 169, 169], [124, 252, 0], [160, 82, 45],
                  [165, 42, 42], [0, 255, 255], [0, 0, 128], [0, 0, 0], [244, 164, 96], [220, 20, 60]]   # 11种预设颜色值
    label_list = [3, 5, 7, 10, 14, 17, 22, 27, 35, 47, 67]      # 11种label，排列顺序和color_list对应

    label_img = np.full((256, 256), 151) # 使用151(代表unknown)初始化
    # label_img = np.int8(label_img)

    for i in range(11):
        a = color_list[i]
        b = input_img - color_list[i]
        mapped_pixels = (input_img - color_list[i]) == 0 # 和第i个类别匹配的像素为true
        mapped_pixels_1 = mapped_pixels[:, :, 0] * mapped_pixels[:, :, 1] * mapped_pixels[:, :, 2]
        label_img[mapped_pixels_1] = label_list[i]

    return label_img

if __name__ == '__main__':
    # img = Image.open('../drawing/Drawing_board-master/output/skysea.jpg')
    img = cv2.imread('../drawing/Drawing_board-master/output/skysea.jpg', -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    size = (256, 256)
    shrink_img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    label_img = color_2_label(shrink_img) # lable_img 是单通道的numpy数组，256*256， 像素点数值代表类别ID