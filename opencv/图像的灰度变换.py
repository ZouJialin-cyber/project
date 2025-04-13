import cv2
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def invert_image(image_path):
    img = cv.imread(image_path)
    gray_img = cv.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 直接利用数学运算实现反色
    result_math_image = 255 - img
    result_math_image_gray = 255 - gray_img

    # LUT快速查表法实现反色
    transTable = np.array([(255 - i) for i in range(256)]).astype(np.uint8)    # 构建查找表
    imgInv = cv.LUT(img, transTable)
    grayInv = cv.LUT(gray_img, transTable)

    cv.imwrite(r"D:\bgi_project\demo\output\result_math_image.jpg", result_math_image)
    cv.imwrite(r"D:\bgi_project\demo\output\result_math_image_gray.jpg", result_math_image_gray)
    cv.imwrite(r"D:\bgi_project\demo\output\imgInv.jpg", imgInv)
    cv.imwrite(r"D:\bgi_project\demo\output\grayInv.jpg", grayInv)

def normalizedGrayhist(src):
    iMax, iMin = np.max(src), np.min(src)
    oMax, oMin = 255, 0
    a = float((oMax - oMin) / (iMax - iMin))
    b = oMin - a * iMin
    dst = a * src - b
    return dst.astype(np.uint8)

def normalizedGrayhist_master(image_path):
    gray = cv.imread(image_path, flags=0)    # 以灰度图的方式读取图像

    # 直方图正规化
    gray = cv.add(cv.multiply(gray, 0.6), 36)
    grayNorm1 = normalizedGrayhist(gray)    # 直方图正规化子函数
    grayNorm2 = cv.normalize(gray, None, 0, 255, cv.NORM_MINMAX)     # Opencv 函数

    cv.imwrite(r"D:\bgi_project\demo\output\grayNorm1.jpg", grayNorm1)
    cv.imwrite(r"D:\bgi_project\demo\output\grayNorm2.jpg", grayNorm2)


if __name__ == '__main__':
    image_path = r"D:\bgi_project\demo\data\stomics_center.jpg"

    # 图像反色
    invert_image(image_path)

    # 直方图正规化（归一化）
    normalizedGrayhist_master(image_path)