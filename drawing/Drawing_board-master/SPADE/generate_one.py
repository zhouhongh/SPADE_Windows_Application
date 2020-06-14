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
    label_img_path = './label_imgs/02.png'
    image_path = './label_imgs/02.png'
    img = generate_one(label_img_path, image_path)
    plt.imshow(img)
    plt.show()


