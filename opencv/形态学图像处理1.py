import cv2 as cv
import numpy as np

def imageErodeDilate(image_path):
    img = cv.imread(image_path, flags=0)
    _, imgOSTU = cv.threshold(img, 128, 255, cv.THRESH_OTSU)

    # 图像腐蚀
    ksize = (3, 3)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, ksize)
    imgErode = cv.erode(imgOSTU, kernel=kernel)

    # 图像膨胀
    imgDilate = cv.dilate(imgErode, kernel=kernel)

    cv.imwrite(r"D:\bgi_project\demo\output\imgDilate.png", imgDilate)

def imageOpenClose(image_path):
    img = cv.imread(image_path, flags=0)
    _, imgOSTU = cv.threshold(img, 128, 255, cv.THRESH_OTSU)

    cv.imwrite(r"D:\bgi_project\demo\output\imgOSTU.png", imgOSTU)

    ksize = (5, 5)
    element = cv.getStructuringElement(cv.MORPH_RECT, ksize)

    imgOpen = cv.morphologyEx(imgOSTU, cv.MORPH_OPEN, kernel=element)

    imgClose = cv.morphologyEx(imgOSTU, cv.MORPH_CLOSE, kernel=element)

    cv.imwrite(r"D:\bgi_project\demo\output\imgOpen.png", imgOpen)
    cv.imwrite(r"D:\bgi_project\demo\output\imgClose.png", imgClose)

def imageGradient(image_path):
    img = cv.imread(image_path, flags=0)
    _, imgOSTU = cv.threshold(img, 128, 255, cv.THRESH_OTSU)

    ksize = (5, 5)
    element = cv.getStructuringElement(cv.MORPH_RECT, ksize)

    # 形态学梯度运算对噪声敏感，先进行开运算去除噪点，再进行梯度运算
    imgOpen = cv.morphologyEx(imgOSTU, cv.MORPH_OPEN, kernel=element)
    imgGradient = cv.morphologyEx(imgOpen, cv.MORPH_GRADIENT, kernel=element)

    cv.imwrite(r"D:\bgi_project\demo\output\imgGradient.png", imgGradient)

def imageHitMiss(image_path):
    img = cv.imread(image_path, flags=0)
    _, imgOSTU = cv.threshold(img, 128, 255, cv.THRESH_OTSU)

    ksize = (12, 12)
    element = cv.getStructuringElement(cv.MORPH_RECT, ksize)

    imgHitMiss = cv.morphologyEx(imgOSTU, cv.MORPH_HITMISS, kernel=element)

    cv.imwrite(r"D:\bgi_project\demo\output\imgHitMiss.png", imgHitMiss)


if __name__ == '__main__':
    image_path = r"D:\bgi_project\demo\data\stomics_center.jpg"

    # 图像的腐蚀和膨胀
    imageErodeDilate(image_path)

    # 图像的开运算与闭运算
    imageOpenClose(image_path)

    # 形态学梯度运算，膨胀与腐蚀之差，用于提取图像的边缘
    imageGradient(image_path)

    # 击中-击不中变换进行特征识别，提取符合要求的形状，实现图像细化
    imageHitMiss(image_path)