import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import copy

def fixeThreshold():
    # 生成灰度图
    h, w = 512, 512
    img = np.zeros((h, w), np.uint8)    # 创建黑色图像
    cv.rectangle(img, (60, 60), (450, 320), (127, 127, 127), -1)    # 矩形填充
    cv.circle(img, (256, 256), 120, (205, 205, 205), -1)    # 圆形填充

    # 添加高斯噪声
    mu, sigma = 0.0, 20.0
    noiseGause = np.random.normal(mu, sigma, img.shape)
    imgNoise = np.add(img, noiseGause)
    imgNoise = np.uint8(cv.normalize(imgNoise, None, 0, 255, cv.NORM_MINMAX))

    # 阈值处理
    _, imgBin1 = cv.threshold(imgNoise, 63, 255, cv.THRESH_BINARY)
    _, imgBin2 = cv.threshold(imgNoise, 125, 255, cv.THRESH_BINARY)
    _, imgBin3 = cv.threshold(imgNoise, 175, 255, cv.THRESH_BINARY)

    cv.imwrite(r"D:\bgi_project\demo\output\img.jpg", img)
    cv.imwrite(r"D:\bgi_project\demo\output\imgNoise.jpg", imgNoise)
    cv.imwrite(r"D:\bgi_project\demo\output\imgBin1.jpg", imgBin1)
    cv.imwrite(r"D:\bgi_project\demo\output\imgBin2.jpg", imgBin2)
    cv.imwrite(r"D:\bgi_project\demo\output\imgBin3.jpg", imgBin3)

def globalThreshold(image_path):
    img = cv.imread(image_path, flags=0)

    deltaT = 1    # 预定义值
    histCV = cv.calcHist([img], [0], None, [256], [0, 255])
    grayScale = range(256)    # 灰度级[0, 255]
    totalPixels = img.shape[0] * img.shape[1]    # 像素总数
    totalGray = np.dot(histCV[:, 0], grayScale)    # 内积，总和灰度值
    T = round(totalGray / totalPixels)    # 平均灰度作为阈值初值

    while True:    # 迭代计算分割阈值
        numC1 = np.sum(histCV[:T, 0])    # C1像素数量
        sumC1 = np.sum(histCV[:T, 0] * range(T))    # C1灰度值总和
        numC2 = totalPixels - numC1    # C2像素数量
        sumC2 = totalGray - sumC1    # C2灰度值总和
        T1 = round(sumC1 / numC1)    # C1平均灰度
        T2 = round(sumC2 / numC2)  # C2平均灰度
        Tnew = round((T1 + T2) / 2)    # 新的阈值
        print(f"T={T}, m1={T1}, m2={T2}, Tnew={Tnew}")
        if abs(T - Tnew) < deltaT:
            break
        else:
            T = Tnew

    # 阈值处理
    ret, imgBin = cv.threshold(img, T, 255, cv.THRESH_BINARY)

    cv.imwrite(r"D:\bgi_project\demo\output\imgBin.jpg", imgBin)

def ostu(image_path):
    img = cv.imread(image_path, flags=0)

    ret, imgOSTU = cv.threshold(img, 128, 255, cv.THRESH_OTSU)    # 此时thresh参数不起作用
    cv.imwrite(r"D:\bgi_project\demo\output\imgOSTU.jpg", imgOSTU)


def doubleThreshold(img):
    histCV = cv.calcHist([img], [0], None, [256], [0, 255])
    grayScale = np.arange(0, 256, 1)    # 灰度级[0, 255]
    totalPixels = img.shape[0] * img.shape[1]    # 像素总数
    totalGray = np.dot(histCV[:, 0], grayScale)    # 内积，总和像素值
    mG = totalGray / totalPixels    # 平均灰度mean gray
    varG = sum(((i-mG)**2 * histCV[:, 0] / totalPixels) for i in range(256))

    T1, T2, varMax = 1, 2, 0.0
    for k1 in range(1, 254):    # k1: [1, 253], 1<=k1<k2<=254
        n1 = sum(histCV[:k1, 0])    # C1像素数量
        s1 = sum((i * histCV[i, 0] for i in range(k1)))
        P1 = n1 / totalPixels    # C1像素占比
        m1 = (s1 / n1) if n1 > 0 else 0    # C1平均灰度
        for k2 in range(k1 + 1, 256):    # k2: [2, 254], k2>k1
            n3 = sum(histCV[k2 + 1:, 0])    # C3像素数量
            s3 = sum((i * histCV[i, 0] for i in range(k2 + 1, 256)))
            P3 = n3 / totalPixels    # C3像素占比
            m3 = (s3 / n3) if n3 > 0 else 0    # C3平均灰度
            P2 = 1.0 - P1 - P3    # C2像素占比
            m2 = (mG - P1*m1 - P3*m3) / P2 if P2 > 1e-6 else 0    # C2平均灰度
            var = P1 * (m1-mG)**2 + P2 * (m2-mG)**2 + P3 * (m3-mG)**2
            if var > varMax:
                T1, T2, varMax = k1, k2, var

    epsT = varMax / varG    # 可分离测度
    print(totalPixels, mG, varG, varMax, epsT, T1, T2)
    return T1, T2, epsT

def multiOTSU(image_path):
    img = cv.imread(image_path)

    T1, T2, epsT = doubleThreshold(img)    # 多阈值处理子程序
    print("T1={}, T2={}, epsT={:.4f}".format(T1, T2, epsT))

    # 基于OSTU算法的最优阈值进行多阈值处理
    ret, imgOSTU = cv.threshold(img, 128, 256, cv.THRESH_OTSU)

    cv.imwrite(r"D:\bgi_project\demo\output\imgOSTU.jpg", imgOSTU)

def adaptThreshold(image_path):
    img = cv.imread(image_path, flags=0)

    # 自适应局部阈值处理
    binaryMean = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 7, 3)
    binaryGauss = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 7, 3)

    cv.imwrite(r"D:\bgi_project\demo\output\binaryMean.jpg", binaryMean)
    cv.imwrite(r"D:\bgi_project\demo\output\binaryGauss.jpg", binaryGauss)

def moveThreshold(img, n, b):
    imgFlip = img.copy()
    imgFlip[1:-1:2, :] = np.fliplr(img[1:-1:2, :])    # 向量翻转
    f = imgFlip.flatten()    # 展平为一维
    ret = np.cumsum(f)
    ret[n:] = ret[n:] - ret[:-n]
    m = ret / n    # 移动平均值
    g = np.array(f >= b * m).astype(int)
    g = g.reshape(img.shape)
    g[1:-1:2, :] = np.fliplr(g[1:-1:2, :])
    return g * 255

def moveAverageThreshold(image_path):
    img = cv.imread(image_path, flags=0)

    # 移动平均阈值处理，n=8，b=0.8
    imgMAthres1 = moveThreshold(img, 8, 0.8)

    cv.imwrite(r"D:\bgi_project\demo\output\imgMAthres1.jpg", imgMAthres1)



if __name__ == '__main__':
    image_path = r"D:\bgi_project\demo\data\stomics_center.jpg"

    # 固定阈值法
    fixeThreshold()

    # 全局阈值计算，选一个合适的固定阈值
    globalThreshold(image_path)

    # OSTU阈值算法
    ostu(image_path)

    # OTSU算法计算多阈值处理的阈值
    multiOTSU(image_path)

    # 自适应阈值处理
    adaptThreshold(image_path)

    # 移动平均阈值处理，适用于光照不均匀的图像
    moveAverageThreshold(image_path)