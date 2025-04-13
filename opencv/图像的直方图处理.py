import cv2
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def paint_hist_image(histCV):
    # histCV参数为直方图数据，由cv.calcHist计算而来；返回值为直方图的numpy数组
    hist_image_np = np.zeros((256, 256, 1), dtype=np.uint8)
    hist_max_np = np.max(histCV)
    for i, h in enumerate(histCV):
        h = h[0]
        cv.line(hist_image_np, (i, 256), (i, 256 - int(h * 256 / hist_max_np)), (255, 255, 255), 1)
    return hist_image_np

def presentHist(image_path):
    img = cv.imread(image_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Opencv方法计算灰度图的直方图
    histCV = cv.calcHist([gray], [0], None, [256], [0, 255])    # 所有参数都已列表的形式传入

    # Numpy方法计算灰度图的直方图
    histNP, bins = np.histogram(gray.flatten(), 256)

    # 绘制OpenCV计算的灰度图直方图并保存
    hist_image_cv = paint_hist_image(histCV)
    cv.imwrite(r"D:/bgi_project/demo/output/histCV.jpg", hist_image_cv)

    # 绘制Numpy计算的灰度图直方图并保存
    hist_image_np = np.zeros((256, 256, 1), dtype=np.uint8)
    hist_max_np = np.max(histNP)
    for i, h in enumerate(histNP):
        cv.line(hist_image_np, (i, 256), (i, 256 - int(h * 256 / hist_max_np)), (255, 255, 255), 1)
    cv.imwrite(r"D:/bgi_project/demo/output/histNP.jpg", hist_image_np)

    # 计算彩色图各个通道的直方图并保存
    for i in range(3):
        histCh = cv.calcHist([img], [i], None, [256], [0, 255])
        hist_image_ch = np.zeros((256, 256, 1), dtype=np.uint8)
        hist_max_ch = np.max(histCh)
        for j, h in enumerate(histCh):
            h = h[0]
            cv.line(hist_image_ch, (j, 256), (j, 256 - int(h * 256 / hist_max_ch)), (255, 255, 255), 1)
        cv.imwrite(f"D:/bgi_project/demo/output/histCh_{i}.jpg", hist_image_ch)


def equalize_hist(image_path):
    gray = cv.imread(image_path, flags=0)
    cv.imwrite(r"D:/bgi_project/demo/output/gray.jpg", gray)

    histSrc = cv.calcHist([gray], [0], None, [256], [0, 255])    # 原始直方图

    histSrc_image = paint_hist_image(histSrc)
    cv.imwrite(r"D:/bgi_project/demo/output/histSrc_image.jpg", histSrc_image)

    # 直方图均衡化
    grayEqualize = cv.equalizeHist(gray)
    histEqual = cv.calcHist([grayEqualize], [0], None, [256], [0, 255])    # 增强后的直方图
    cv.imwrite(r"D:/bgi_project/demo/output/grayEqualize.jpg", grayEqualize)

    histEqual_image = paint_hist_image(histEqual)
    cv.imwrite(r"D:/bgi_project/demo/output/histEqual_image.jpg", histEqual_image)

    # 对比直方图归一化
    grayNorm = cv.normalize(gray, None, 0, 255, cv.NORM_MINMAX)
    histNorm = cv.calcHist([grayNorm], [0], None, [256], [0, 255])
    cv.imwrite(r"D:/bgi_project/demo/output/grayNorm.jpg", grayNorm)

    histNorm_image = paint_hist_image(histNorm)
    cv.imwrite(r"D:/bgi_project/demo/output/histNorm_image.jpg", histNorm_image)

def clahe(image_path):
    gray = cv.imread(image_path, flags=0)

    # 创建CLAHE类的对象
    Clahe = cv.createCLAHE(clipLimit=100, tileGridSize=(16, 16))    # clipLimit为颜色对比度阈值，tileGridSize为分块网格数

    imgClahe = Clahe.apply(gray)
    cv.imwrite(r"D:/bgi_project/demo/output/imgClahe.jpg", imgClahe)


if __name__ == '__main__':
    image_path = r"D:\bgi_project\demo\data\stomics_center.jpg"

    # 灰度图与彩色图的直方图的展示
    presentHist(image_path)

    # 灰度图的直方图均衡化
    equalize_hist(image_path)

    # 限制对比度自适应局部直方图均衡化
    clahe(image_path)