import cv2 as cv
import numpy as np
import time

def opencvDFT(image_path):
    img = cv.imread(image_path, flags=0)

    imgFloat = img.astype(np.float32)    # 转换为float格式
    dft = cv.dft(imgFloat, flags=cv.DFT_COMPLEX_OUTPUT)    # 傅里叶变换
    difShift = np.fft.fftshift(dft)    # 将低频分量移动到频谱中心

    iShift = np.fft.ifftshift(difShift)    # 将低频分量移回四角
    idft = cv.idft(iShift)    # 逆傅里叶变换
    idftAmp = cv.magnitude(idft[:, :, 0], idft[:, :, 1])    # 重建图像
    rebuild = np.uint8(cv.normalize(idftAmp, None, 0, 255, cv.NORM_MINMAX))    # 将数据范围归一化到[0, 255]

    cv.imwrite(r"D:\bgi_project\demo\output\rebuild.jpg", rebuild)

def numpyDFT(image_path):
    img = cv.imread(image_path, flags=0)

    fft = np.fft.fft2(img)
    fftShift = np.fft.fftshift(fft)    # 中心化，将低频分量移动到频谱中心

    iFftShift = np.fft.ifftshift(fftShift)    # 逆中心化，将低频分量移动到四角
    ifft = np.fft.ifft2(iFftShift)    # 傅里叶逆变换
    rebuild = np.abs(ifft)    # 重建图像，复数的模

    cv.imwrite(r"D:\bgi_project\demo\output\rebuild.jpg", rebuild)

def opencvFastDFT():
    img = cv.imread(r"D:\bgi_project\demo\data\stomics_center.jpg", flags=0)
    imgFloat = img.astype(np.float32)
    # img = np.zeros((1101, 1821), np.uint8)
    # cv.rectangle(img, (100, 100), (900, 900), 128, -1)
    # cv.circle(img, (500, 500), 306, 225, -1)

    height, width = img.shape[:2]
    hPad = cv.getOptimalDFTSize(height)    # 计算傅里叶变换的最优尺寸
    wPad = cv.getOptimalDFTSize(width)
    imgOpt = np.zeros((hPad, wPad), np.float32)    # 初始化扩充图像
    imgOpt[:height, :width] = imgFloat    # 原图的float类型填充最优尺寸图，下侧和右侧补0

    dft = cv.dft(imgOpt, flags=cv.DFT_COMPLEX_OUTPUT)    # 傅里叶变换
    dftShift = np.fft.fftshift(dft)    # 中心化

    iShift = np.fft.ifftshift(dftShift)    # 去中心化
    idft = cv.idft(iShift)    # 逆傅里叶变换
    idftAmpt = cv.magnitude(idft[:, :, 0], idft[:, :, 1])    # 重建图像
    rebuild = np.uint8(cv.normalize(idftAmpt, None, 0, 255, cv.NORM_MINMAX))

    cv.imwrite(r"D:\bgi_project\demo\output\rebuild.jpg", rebuild)


if __name__ == '__main__':
    image_path = r"D:\bgi_project\demo\data\stomics_center.jpg"

    # opencv实现傅里叶变换和逆傅里叶变换
    start_time_dft = time.time()

    opencvDFT(image_path)

    end_time_dft = time.time()

    time_dft = end_time_dft - start_time_dft
    print(f"傅里叶变换的运行时间为{time_dft}s")

    # numpy实现傅里叶变换和逆傅里叶变换
    # numpyDFT(image_path)

    # opencv实现快速傅里叶变换
    start_time_fastDft = time.time()

    opencvFastDFT()

    end_time_fastDft = time.time()

    time_fastDft = end_time_fastDft - start_time_fastDft
    print(f"快速傅里叶变换的运行时间为{time_fastDft}s")