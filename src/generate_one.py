"""
function: provide one label mapping pic, generate one realistic pic.
author: zhouhonghong
date: 2020/06/13
"""

from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from data.base_dataset import get_params, get_transform

import torch
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import cv2

def color_2_label(input_img):
    #  手动设置以下两列表
    color_list = [[135, 206, 235], [34, 139, 34], [169, 169, 169], [124, 252, 0], [160, 82, 45],
                  [165, 42, 42], [0, 255, 255], [0, 0, 128], [0, 0, 0], [244, 164, 96], [220, 20, 60]]   # 11种预设颜色值
    label_list = [3, 5, 7, 10, 14, 17, 22, 27, 35, 47, 67]      # 11种label，排列顺序和color_list对应

    label_img = np.full((256, 256), 151) # 使用151(代表unknown)初始化
    # label_img = np.int8(label_img)

    for i in range(11):
        mapped_pixels = (input_img - color_list[i]) == 0 # 和第i个类别匹配的像素为true
        mapped_pixels_1 = mapped_pixels[:, :, 0] * mapped_pixels[:, :, 1] * mapped_pixels[:, :, 2]
        label_img[mapped_pixels_1] = label_list[i]

    return label_img


def generate_one(label_img_path, image_path):
    opt = TestOptions().parse()

    label = Image.open(label_img_path)
    # label_np = np.array(label)
    params = get_params(opt, label.size)
    transform_label = get_transform(opt, params, method=Image.NEAREST, normalize=False)
    label_tensor = transform_label(label) * 255.0
    label_tensor[label_tensor == 255] = opt.label_nc
    label_tensor_batch = torch.unsqueeze(label_tensor,0)
    if opt.no_instance:
        instance_tensor = 0

    image = Image.open(image_path)
    image = image.convert('RGB')
    transform_image = get_transform(opt, params)
    image_tensor = transform_image(image)

    input_dict = {'label': label_tensor_batch,
                  'instance': instance_tensor,
                  'image': image_tensor,
                  }

    # 加载模型

    model = Pix2PixModel(opt)
    model.eval()
    ## 得到3*256*256的图片数据
    generated_image = model(input_dict, mode='inference')
    generated_image = torch.squeeze(generated_image)
    image_numpy = generated_image.numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    img = image_numpy.astype(np.uint8)

    # # 给定image_path即可保存
    # image_pil.save(image_path.replace('.jpg', '.png'))
    return img

if __name__ =='__main__':

    # img = cv2.imread('../drawing/Drawing_board-master/output/skysea.jpg', -1)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # size = (256, 256)
    # shrink_img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    # label_img = color_2_label(shrink_img) # lable_img 是单通道的numpy数组，256*256， 像素点数值代表类别ID


    label_img_path = './label_imgs/02.png'
    image_path = './label_imgs/02.png'
    img = generate_one(label_img_path, image_path)
    plt.imshow(img)
    plt.show()


