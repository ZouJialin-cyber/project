import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread(r"D:\bgi_project\demo\data\Center_86274.jpg")
    h, w = img.shape[:2]

    # 图像平移
    dx, dy = 30, 20    # dx, dy分别为x方向和y方向的平移像素距离, 根据需要求改
    MAT = np.float32([[1, 0, dx], [0, 1, dy]])    # 平移变换矩阵, 无需修改
    imgTrans1 = cv.warpAffine(img, MAT, (w, h))    # 平移函数, 第三个参数为w和h的元组, 平移后图像的宽高
    imgTrans2 = cv.warpAffine(img, MAT, (300, 200), borderValue=(255, 255, 255))    # 使用白色填充移动后空白区域，不适用默认黑色

    cv.imwrite(r"D:\bgi_project\demo\output\imgTrans1.jpg", imgTrans1)
    cv.imwrite(r"D:\bgi_project\demo\output\imgTrans2.jpg", imgTrans2)