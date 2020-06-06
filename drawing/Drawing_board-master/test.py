#泛洪填充(彩色图像填充)
'''
 floodFill(image, mask, seedPoint, newVal[, loDiff[, upDiff[, flags]]]) -> retval, image, mask, rect
image:表示输入/输出1或3通道，8位或浮点图像
mask参数表示掩码，该掩码是单通道8位图像，比image的高度多2个像素，宽度多2个像素.填充时不能穿过输入掩码中的非零像素。
seedPoint参数表示泛洪算法(漫水填充算法)的起始点。
newVal参数表示在重绘区域像素的新值。
loDiff参数表示当前观察像素值与其部件邻域像素值或待加入该组件的种子像素之间的亮度或颜色之负差的最大值.
upDiff参数表示当前观察像素值与其部件邻域像素值或待加入该组件的种子像素之间的亮度或颜色之正差的最大值。
flags参数：操作标志符，包含三部分：（参考https://www.cnblogs.com/little-monkey/p/7598529.html）

　　　　低八位（0~7位）：用于控制算法的连通性，可取4（默认）或8。

　　　　中间八位（8~15位）：用于指定掩码图像的值，但是如果中间八位为0则掩码用1来填充。

　　　　高八位（16~32位）：可以为0或者如下两种标志符的组合：

　　　　FLOODFILL_FIXED_RANGE:表示此标志会考虑当前像素与种子像素之间的差，否则就考虑当前像素与相邻像素的差。FLOODFILL_MASK_ONLY:表示函数不会去填充改变原始图像,而是去填充掩码图像mask，mask指定的位置为零时才填充，不为零不填充
'''
import cv2 as cv
import  time
import pygame
import numpy as np


def fill_color_demo(image):
    copyImg = image.copy()
    h, w = image.shape[:2]
    mask = np.zeros([h+2, w+2], np.uint8)   #mask必须行和列都加2，且必须为uint8单通道阵列
    #为什么要加2可以这么理解：当从0行0列开始泛洪填充扫描时，mask多出来的2可以保证扫描的边界上的像素都会被处理
    t1 = time.clock()
    cv.floodFill(copyImg, mask, (146, 59), (0, 0, 0), (1, 1, 1), (1, 1, 1), cv.FLOODFILL_FIXED_RANGE)
    t2 = time.clock()
    print(t2-t1)
    cv.imshow("fill_color_demo", copyImg)


src = cv.imread('images/circle.png')
cv.namedWindow('input_image', cv.WINDOW_AUTOSIZE)
cv.imshow('input_image', src)
fill_color_demo(src)
cv.waitKey(0)
cv.destroyAllWindows()

screen = pygame.display.set_mode((800, 600))#设置屏幕尺寸
screen.fill((255, 255, 255))
