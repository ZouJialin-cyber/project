import cv2 as cv
import numpy as np
import copy


def laplacian(image_path):
    img = cv.imread(image_path, flags=0)

    imgLaplacian = cv.Laplacian(img, cv.CV_32F, ksize=3)    # 输出图像类型为CV的32位浮点型数据
    absLaplacian = cv.convertScaleAbs(imgLaplacian)    # 数据拉伸到[0, 255]范围才能显示

    cv.imwrite(r"D:\bgi_project\demo\output\absLaplacian.jpg", absLaplacian)

def sobel(image_path):
    img = cv.imread(image_path)

    sobelX = cv.Sobel(img, cv.CV_64F, 1, 0)    # X方向
    sobelY = cv.Sobel(img, cv.CV_64F, 0, 1)    # Y方向
    # 图像格式转变到np.uint8
    absSobelX = cv.convertScaleAbs(sobelX)
    absSobelY = cv.convertScaleAbs(sobelY)
    absSobelXY = cv.add(absSobelX, absSobelY)    # 使用绝对值替代平方根，加快运算速度

    cv.imwrite(r"D:\bgi_project\demo\output\absSobelXY.jpg", absSobelXY)

def scharr(image_path):
    img = cv.imread(image_path)

    scharrX = cv.Scharr(img, cv.CV_64F, 1, 0)
    scharrY = cv.Scharr(img, cv.CV_64F, 0, 1)
    absScharrX = cv.convertScaleAbs(scharrX)
    absScharrY = cv.convertScaleAbs(scharrY)
    absScharrXY = cv.add(absScharrX, absScharrY)

    cv.imwrite(r"D:\bgi_project\demo\output\absScharrXY.jpg", absScharrXY)

def imagePyramid(image_path):
    img = cv.imread(r"D:\bgi_project\demo\data\Center_86274.jpg")

    # 图像降采样，构建高斯金字塔
    pyrG0 = copy.deepcopy(img)
    pyrG1 = cv.pyrDown(pyrG0)
    pyrG2 = cv.pyrDown(pyrG1)
    pyrG3 = cv.pyrDown(pyrG2)
    pyrG4 = cv.pyrDown(pyrG3)

    # 高斯金字塔上采样
    pyrU4 = cv.pyrUp(pyrG4)
    pyrU3 = cv.pyrUp(pyrU4)
    pyrU2 = cv.pyrUp(pyrU3)
    pyrU1 = cv.pyrUp(pyrU2)

    # 构建拉普拉斯金字塔
    pyrL0 = pyrG0 - cv.pyrUp(pyrG1)
    pyrL1 = pyrG1 - cv.pyrUp(pyrG2)
    pyrL2 = pyrG2 - cv.pyrUp(pyrG3)
    pyrL3 = pyrG3 - cv.pyrUp(pyrG4)

    # 上采样恢复高分辨率图像
    rebuildG3 = pyrL3 + cv.pyrUp(pyrG4)
    rebuildG2 = pyrL2 + cv.pyrUp(rebuildG3)
    rebuildG1 = pyrL1 + cv.pyrUp(rebuildG2)
    rebuildG0 = pyrL0 + cv.pyrUp(rebuildG1)

    cv.imwrite(r"D:\bgi_project\demo\output\pyrG4.jpg", pyrG4)
    cv.imwrite(r"D:\bgi_project\demo\output\pyrU1.jpg", pyrU1)
    cv.imwrite(r"D:\bgi_project\demo\output\pyrL3.jpg", pyrL3)
    cv.imwrite(r"D:\bgi_project\demo\output\rebuildG0.jpg", rebuildG0)


if __name__ == '__main__':
    image_path = r"D:\bgi_project\demo\data\stomics_center.jpg"

    # 拉普拉斯算子，主要用于图像的边缘检测
    laplacian(image_path)

    # Sobel算子实现图像的边缘检测
    sobel(image_path)

    # Scharr算子（优化的Sobel算子，中心权重更大，相当于方差较小的高斯分布）实现图像的边缘检测
    scharr(image_path)

    # 图像经过高斯模糊和降采样得到一系列低分辨率图，放到一起就是高斯金字塔，高斯金字塔上采样会丢失原有信息，拉普拉斯金字塔就是高斯金字塔的残差信息，上采样的时候加上拉普拉斯金字塔可以还原丢失信息
    imagePyramid(image_path)