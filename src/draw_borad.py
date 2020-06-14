import pygame
from pygame.locals import *
import math
import cv2 as cv
import numpy as np
import tkinter.filedialog
from tkinter import *
from PIL import Image
import matplotlib.pyplot as plt

from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from data.base_dataset import get_params, get_transform

import torch





# 无法导入SPADE下的color_2_label和generate_one函数
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

    global model
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

class Brush:
    def __init__(self, screen):
        self.screen = screen
        self.color = (0, 0, 0)  # 三元组表示颜色
        self.size = 1  # 初始尺寸
        self.drawing = False  # 是否选中
        self.last_pos = None  # 最后的位置
        self.style = False  # 刷子的样式
        self.brush = pygame.image.load("images/brush.png").convert_alpha()
        # 转换成xx  不用纠结   此变量表示原始图片，放大、缩小、变色在此原始图片基础上
        self.brush_now = self.brush.subsurface((0, 0), (1, 1))  # 子表面
        # 此变量表示 当前的刷子图像(可以变大变小)

    # added by zhh : fill
    def fill(self, pos, clicked_color):
        pixel_color_rgba = self.screen.get_at(pos)
        pixel_color_rgb = (pixel_color_rgba[0], pixel_color_rgba[1], pixel_color_rgba[2])
        if pixel_color_rgb != clicked_color:
            print("pass")
            pass
        else:
            # 设定pixel为选择的颜色
            print("filled one pixel in", pos)
            self.screen.set_at(pos, self.color)
            if pos[0] < 799:
                self.fill((pos[0] + 1, pos[1]), clicked_color)
            if pos[0] > 0:
                self.fill((pos[0] - 1, pos[1]), clicked_color)
            if pos[1] < 599:
                self.fill((pos[0], pos[1] + 1), clicked_color)
            if pos[1] > 0:
                self.fill((pos[0], pos[1] - 1), clicked_color)

    def start_draw(self, pos):
        self.drawing = True
        self.last_pos = pos

    def end_draw(self):
        self.drawing = False

    def set_brush_style(self, style):
        print("* set brush style to", style)
        self.style = style

    def get_brush_style(self):
        return self.style

    def get_current_brush(self):
        return self.brush_now

    def set_size(self, size):
        # 限制最小和最大的size
        if size < 1:
            size = 1
        elif size > 32:
            size = 32
        print("* set brush size to", size)
        self.size = size
        if self.style:  # 样式时调整样式尺寸，圆时无需调整，>11时还是会出问题
            self.brush_now = self.brush.subsurface((0, 0), (size * 2, size * 2))

    def get_size(self):
        return self.size

    def set_color(self, color):  # 换颜色
        self.color = color
        for i in range(self.brush.get_width()):
            for j in range(self.brush.get_height()):
                self.brush.set_at((i, j),
                                  color + (self.brush.get_at((i, j)).a,))
                # get_at获得某一像素位置的颜色   三元组可以相加  为什么是.a

    # set_at 获取某一像素位置的颜色 参数:像素坐标，颜色（三元组）
    def get_color(self):
        return self.color

    def draw(self, pos):
        if self.drawing:
            for p in self._get_points(pos):
                if self.style:
                    self.screen.blit(self.brush_now, p)  # blit填充
                else:
                    pygame.draw.circle(self.screen, self.color, p, self.size)
            self.last_pos = pos

    def _get_points(self, pos):  # 获取画板坐标
        points = [(self.last_pos[0], self.last_pos[1])]
        len_x = pos[0] - self.last_pos[0]
        len_y = pos[1] - self.last_pos[1]
        length = math.sqrt(len_x ** 2 + len_y ** 2)
        step_x = len_x / length
        step_y = len_y / length
        for i in range(int(length)):
            points.append((points[-1][0] + step_x, points[-1][1] + step_y))
        points = map(lambda x: (int(0.5 + x[0]), int(0.5 + x[1])), points)
        return list(set(points))


class Menu:
    """
    Rect裁剪的用法:https://blog.csdn.net/qq_34404196/article/details/80718380
    """

    def __init__(self, screen):
        self.screen = screen
        self.brush = None
        self.colors = [
            (135, 206, 235),
            (34, 139, 34),
            (169, 169, 169),
            (124, 252, 0),
            (160, 82, 45),
            (165, 42, 42),
            (0, 255, 255),
            (0, 0, 128),
            (220, 20, 60),
            (244, 164, 96),
            (0, 0, 0),
        ]
        text = [u"天", u"树", u"路", u"草", u"土", u"山", u"水", u"海", u"花", u"沙", u"石", ]
        self.colors_rect = []
        for (i, rgb) in enumerate(self.colors):
            rect = pygame.Rect(10, 254 + i * 32, 32, 32)
            self.colors_rect.append(rect)
        # 画笔
        self.pens = [
            pygame.image.load("images/pen1.png").convert_alpha(),
            pygame.image.load("images/pen2.png").convert_alpha(),
        ]
        self.pens_rect = []
        for (i, img) in enumerate(self.pens):
            rect = pygame.Rect(2, 10 + i * 40, 32, 32)
            self.pens_rect.append(rect)
        # 保存按钮
        self.save_img = pygame.image.load("images/save.png").convert_alpha()
        self.save_rect = pygame.Rect(37 + 2, 10, 32, 32)
        self.doSave = False
        # 导入到画板
        self.input_img = pygame.image.load("images/input.png").convert_alpha()
        self.input_rect = pygame.Rect(37 + 2, 10 + 40, 32, 32)
        self.doInput = False
        # 尺寸调整按钮
        self.sizes = [
            pygame.image.load("images/big.png").convert_alpha(),
            pygame.image.load("images/small.png").convert_alpha()
        ]
        self.sizes_rect = []
        for (i, img) in enumerate(self.sizes):
            rect = pygame.Rect(10 + i * 32, 138, 32, 32)
            self.sizes_rect.append(rect)

        # added by zhh
        # 油漆桶按钮
        # self.fill_rect = pygame.Rect(26, 550, 32, 32)
        self.fill_rect = pygame.Rect(2, 10 + 40 + 40, 32, 32)
        self.fill_image = pygame.image.load("images/bucket.png").convert_alpha()
        self.doFill = False
        # 输出到神经网络
        self.out2DNN_rect = pygame.Rect(37 + 2, 10 + 40 + 40, 32, 32)
        self.out2DNN_image = pygame.image.load("images/output.png").convert_alpha()
        self.doOut2DNN = False

        # 字体
        self.font = pygame.font.Font("black.TTC", 16)
        self.text_rect = []
        for i in text:
            text_surface = self.font.render(i, True, (0, 0, 0))
            self.text_rect.append(text_surface)

    def set_brush(self, brush):
        self.brush = brush

    def draw(self):  # draw buttons
        for (i, img) in enumerate(self.pens):  # 笔按钮
            self.screen.blit(img, self.pens_rect[i].topleft)
        for (i, img) in enumerate(self.sizes):  # 尺寸按钮
            self.screen.blit(img, self.sizes_rect[i].topleft)
        self.screen.fill((255, 255, 255), (10, 180, 64, 64))  # 尺寸框白底
        pygame.draw.rect(self.screen, (0, 0, 0), (10, 180, 64, 64), 1)
        size = self.brush.get_size()
        x = 10 + 32  # 中心点
        y = 180 + 32  # 中心点
        if self.brush.get_brush_style():  # 有样式
            x = x - size
            y = y - size
            self.screen.blit(self.brush.get_current_brush(), (x, y))
        else:  # 无样式
            pygame.draw.circle(self.screen,
                               self.brush.get_color(), (x, y), size)
        for (i, rgb) in enumerate(self.colors):  # 颜色选项
            pygame.draw.rect(self.screen, rgb, self.colors_rect[i])
        for (i, text) in enumerate(self.text_rect):  # 注释字
            self.screen.blit(text, (52, 262 + i * 32))
        # added by zhh
        self.screen.blit(self.fill_image, self.fill_rect.topleft)
        # 保存按钮
        self.screen.blit(self.save_img, self.save_rect.topleft)
        # 导入按钮
        self.screen.blit(self.input_img, self.input_rect.topleft)
        # 输出到DNN按钮
        self.screen.blit(self.out2DNN_image, self.out2DNN_rect.topleft)

    def click_button(self, pos):  # 颜色放进list中 分别对应 button, 返回ture表示有按键被按下
        for (i, rect) in enumerate(self.pens_rect):  # 画笔
            if rect.collidepoint(pos):
                self.doFill = False  # 退出油漆桶模式
                self.brush.set_brush_style(bool(i))
                return True
        for (i, rect) in enumerate(self.sizes_rect):  # 画笔尺寸
            if rect.collidepoint(pos):
                self.doFill = False  # 退出油漆桶模式
                if i:
                    self.brush.set_size(self.brush.get_size() - 1)
                else:
                    self.brush.set_size(self.brush.get_size() + 1)
                return True
        for (i, rect) in enumerate(self.colors_rect):  # 颜色
            if rect.collidepoint(pos):
                self.brush.set_color(self.colors[i])
                return True
        # added by zhh
        if self.fill_rect.collidepoint(pos):  # 油漆桶
            print("set doFill to True")
            self.doFill = True
            return True

        if self.save_rect.collidepoint(pos):  # 保存
            print("set Save to True")
            self.doSave = True
            return True
        if self.input_rect.collidepoint(pos):  # 导入
            print("set input to True")
            self.doInput = True
            return True
        if self.out2DNN_rect.collidepoint(pos):  # 导出到DNN
            print("set out2DNN to True")
            self.doOut2DNN = True
            return True
        return False


class Painter:
    def __init__(self):
        pygame.init()
        self.MENU_WIDTH = 74
        self.BIGRATIO = 2.5
        self.SCREEN_WIDTH = int(256 * self.BIGRATIO) + self.MENU_WIDTH  # 确保画板保持256.256比例
        self.SCREEN_HEIGHT = int(256 * self.BIGRATIO)

        self.BROAD_WIDTH = self.SCREEN_WIDTH - self.MENU_WIDTH
        self.MENU_HEIGHT = self.SCREEN_HEIGHT
        self.BROAD_HEIGHT = self.SCREEN_HEIGHT

        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))  # 设置屏幕尺寸
        self.sub_screen = \
            self.screen.subsurface((self.MENU_WIDTH, 0), (self.SCREEN_WIDTH - self.MENU_WIDTH, self.SCREEN_HEIGHT - 0))
        pygame.display.set_caption("Painter")  # caption标题  设置标题
        self.clock = pygame.time.Clock()  # 帧速率  帮助我们确保程序以某一个最大的FPS运行
        self.brush = Brush(self.sub_screen)  # 屏幕作为参数传给Brush类
        self.menu = Menu(self.screen)  #
        self.menu.set_brush(self.brush)

    def fill_by_cv2(self, pos):
        # if self.screen.get_at(event.pos) != self.brush.color: clicked_color = (self.screen.get_at(
        # event.pos)[0], self.screen.get_at(event.pos)[1], self.screen.get_at(event.pos)[2])
        # self.brush.fill(event.pos, clicked_color)

        image_data = pygame.surfarray.array3d(self.sub_screen)
        h, w = image_data.shape[:2]
        mask = np.zeros([h + 2, w + 2], np.uint8)
        print(pos)
        cv.floodFill(image_data, mask, (pos[1], pos[0]), self.brush.color, (1, 1, 1),
                     (1, 1, 1),
                     cv.FLOODFILL_FIXED_RANGE)
        image_quote = pygame.surfarray.pixels3d(self.sub_screen)
        # for x in range(image_data.shape[0]):
        #     for y in range(image_data.shape[1]):
        #         image_quote[x][y] = image_data[x][y]
        image_quote[:] = image_data[:]
        del image_quote
        self.sub_screen.unlock()

    def save_broad(self):
        root = Tk()
        filename = tkinter.filedialog.asksaveasfilename()
        root.destroy()
        print(filename)
        self.menu.doSave = False
        if '.jpg' in filename:
            pygame.image.save(self.sub_screen, filename)
        else:
            pygame.image.save(self.sub_screen, filename + '.jpg')

    def input_broad(self):
        self.menu.doInput = False
        root = Tk()
        filename = tkinter.filedialog.askopenfilename()
        root.destroy()
        if filename != '':
            input_img = pygame.image.load(filename).convert()
            self.sub_screen.blit(input_img, (0, 0))

    def out_photo(self):
        self.menu.doOut2DNN = False
        image_data = pygame.surfarray.array3d(self.sub_screen)  # 获取图像
        image_big = np.array(image_data)
        image256 = np.zeros((256, 256, 3))
        for i in range(256):  # Resize
            for j in range(256):
                image256[i][j][:] = image_big[int(i * self.BIGRATIO)][int(j * self.BIGRATIO)][:]
        label_img = color_2_label(image256)  # 转换为label
        im = Image.fromarray(label_img.astype(np.uint8))
        im.save('new_lena.png')
        label_img_path = './new_lena.png'
        image_path = './new_lena.png'
        img = generate_one(label_img_path, image_path)
        plt.imshow(img)
        plt.show()
        print('Out2DNN')

    def run(self):
        self.screen.fill((255, 255, 255))
        while True:
            self.clock.tick(50)
            for event in pygame.event.get():
                if event.type == QUIT:
                    return
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:  # 按esc清空画板
                        self.screen.fill((255, 255, 255))

                elif event.type == MOUSEBUTTONDOWN:
                    if event.pos[0] <= self.MENU_WIDTH and self.menu.click_button(event.pos):  # 点在按钮上了
                        # 点到按钮需要立即执行的
                        if self.menu.doSave:  # 保存
                            self.save_broad()
                        elif self.menu.doInput:  # 导入
                            self.input_broad()
                        elif self.menu.doOut2DNN:  # 导出到DNN
                            self.out_photo()
                    # added by zhh
                    # 点到按钮需要下次执行的
                    elif self.menu.doFill:
                        sub_pos = (event.pos[0] - self.MENU_WIDTH, event.pos[1])  # 转换为subsurface中pos
                        self.fill_by_cv2(sub_pos)
                    else:
                        sub_pos = (event.pos[0] - self.MENU_WIDTH, event.pos[1])  # 转换为subsurface中pos
                        self.brush.start_draw(sub_pos)  # subsurface适配
                elif event.type == MOUSEMOTION:
                    sub_pos = (event.pos[0] - self.MENU_WIDTH, event.pos[1])  # 转换为subsurface中pos
                    self.brush.draw(sub_pos)  # subsurface适配
                elif event.type == MOUSEBUTTONUP:
                    self.brush.end_draw()

            self.menu.draw()
            pygame.display.update()


def main():
    app = Painter()
    app.run()


if __name__ == '__main__':

    # 加载模型
    model = Pix2PixModel(TestOptions().parse())
    model.eval()
    main()
