import cv2 as cv
import numpy as np

def idealLPF(height, width, radius=10):    # 理想低通滤波器
    # 使用 np.linspace 生成均匀分布的坐标
    u = np.linspace(-1, 1, height)
    v = np.linspace(-1, 1, width)
    U, V = np.meshgrid(u, v)
    Dist = cv.magnitude(U.T, V.T)
    DO = radius / height    # 滤波器半径
    kernel = np.zeros((height, width), np.uint8)
    kernel[Dist <= DO] = 1
    return kernel

def idealHPF(height, width, radius=10):    # 理想高通滤波器
    # 使用 np.linspace 生成均匀分布的坐标
    u = np.linspace(-1, 1, height)
    v = np.linspace(-1, 1, width)
    U, V = np.meshgrid(u, v)
    Dist = cv.magnitude(U.T, V.T)
    DO = radius / height    # 滤波器半径
    kernel = np.ones((height, width), np.uint8)
    kernel[Dist <= DO] = 0
    return kernel

def gaussLPF(height, width, radius=10):    # 高斯低通滤波器
    # 使用 np.linspace 生成均匀分布的坐标
    u = np.linspace(-1, 1, height)
    v = np.linspace(-1, 1, width)
    U, V = np.meshgrid(u, v)
    Dist = cv.magnitude(U.T, V.T)
    DO = radius / height  # 滤波器半径
    kernel = np.exp(-(Dist**2)/(2*DO**2))
    return kernel

def gaussHPF(height, width, radius=10):    # 高斯高通滤波器
    # 使用 np.linspace 生成均匀分布的坐标
    u = np.linspace(-1, 1, height)
    v = np.linspace(-1, 1, width)
    U, V = np.meshgrid(u, v)
    Dist = cv.magnitude(U.T, V.T)
    DO = radius / height  # 滤波器半径
    kernel = 1 - np.exp(-(Dist**2)/(2*DO**2))
    return kernel

def butterWorthLPF(height, width, radius=10, n=2):    # 巴特沃斯低通滤波器
    # 使用 np.linspace 生成均匀分布的坐标
    u = np.linspace(-1, 1, height)
    v = np.linspace(-1, 1, width)
    U, V = np.meshgrid(u, v)
    Dist = cv.magnitude(U.T, V.T)
    DO = radius / height  # 滤波器半径
    kernel = 1.0 / (1.0 + np.power(Dist / DO, 2*n))
    return kernel

def butterWorthHPF(height, width, radius=10, n=2):    # 巴特沃斯高通滤波器
    # 使用 np.linspace 生成均匀分布的坐标
    u = np.linspace(-1, 1, height)
    v = np.linspace(-1, 1, width)
    U, V = np.meshgrid(u, v)
    Dist = cv.magnitude(U.T, V.T)
    DO = radius / height  # 滤波器半径
    epsilon = 1e-8
    kernel = 1.0 / (1.0 + np.power(DO / (Dist + epsilon), 2*n))
    return kernel

def LaplacianHPF(height, width):    # 拉普拉斯高通滤波器
    u, v = np.mgrid[-1:1:2.0/height, -1:1:2.0/width]
    D = np.sqrt(u**2 + v**2)
    kernel = -4.0 * np.pi**2 * D**2
    return kernel

def fre_dom_Filter(image_path):
    img = cv.imread(image_path, flags=0)
    height, width = img.shape[:2]

    # 对图像进行傅里叶变换，并将低频分量移动到中心
    imgFloat = img.astype(np.float32)
    dft = cv.dft(imgFloat, flags=cv.DFT_COMPLEX_OUTPUT)
    dftShift = np.fft.fftshift(dft)

    # 构造低通滤波器
    mask = idealLPF(height, width, 300)
    maskDual = cv.merge([mask, mask])    # 频域图是两个通道，mask滤波器也要两个通道

    # 修改傅里叶变换实现频域图像滤波
    dftMask = dftShift * maskDual

    # 逆中心化并进行傅里叶逆变换
    iShift = np.fft.ifftshift(dftMask)
    idft = cv.idft(iShift)
    dftAmp = cv.magnitude(idft[:, :, 0], idft[:, :, 1])
    rebuild = np.uint8(cv.normalize(dftAmp, None, 0, 255, cv.NORM_MINMAX))

    cv.imwrite(r"D:\bgi_project\demo\output\rebuild.jpg", rebuild)

def dftFilter(img, mask):
    # 傅里叶变换和频域滤波
    # :param img : 输入图像，numpy数组
    # :param mask : 滤波器
    imgFloat = img.astype(np.float32)
    dft = cv.dft(imgFloat, flags=cv.DFT_COMPLEX_OUTPUT)
    dftShift = np.fft.fftshift(dft)

    maskDual = cv.merge([mask, mask])

    dftMask = dftShift * maskDual

    iShift = np.fft.ifftshift(dftMask)
    idft = cv.idft(iShift)
    dftAmp = cv.magnitude(idft[:, :, 0], idft[:, :, 1])
    res = np.uint8(cv.normalize(dftAmp, None, 0, 255, cv.NORM_MINMAX))

    return res

def compareLPF(image_path):
    img = cv.imread(image_path, flags=0)
    height, width = img.shape[:2]

    maskIdeal = idealLPF(height, width, 200)
    resIdeal = dftFilter(img, maskIdeal)

    maskGauss = gaussLPF(height, width, 200)
    resGauss = dftFilter(img, maskGauss)

    maskButterWorth = butterWorthLPF(height, width, 200)
    resButterWorth = dftFilter(img, maskButterWorth)

    cv.imwrite(r"D:\bgi_project\demo\output\resIdeal.jpg", resIdeal)
    cv.imwrite(r"D:\bgi_project\demo\output\resGauss.jpg", resGauss)
    cv.imwrite(r"D:\bgi_project\demo\output\resButterWorth.jpg", resButterWorth)


def dftHPF(image_path, flag):
    img = cv.imread(image_path, flags=0)
    height, width = img.shape[:2]

    if flag == "idealHPF":
        mask = idealHPF(height, width, 200)
    elif flag == "gaussHPF":
        mask = gaussHPF(height, width, 200)
    elif flag == "butterWorthHPF":
        mask = butterWorthHPF(height, width, 200)
    elif flag == "LaplacianHPF":
        mask = LaplacianHPF(height, width)
    else:
        print(f"无{flag}高通滤波")

    res = dftFilter(img, mask)

    cv.imwrite(r"D:\bgi_project\demo\output\res.jpg", res)


if __name__ == '__main__':
    image_path = r"D:\bgi_project\demo\data\Center_86295.jpg"

    # 频域图像滤波的基本步骤
    fre_dom_Filter(image_path)

    # 理想低通滤波器、高斯低通滤波器和巴特沃斯低通滤波器的比较
    compareLPF(image_path)

    # 高通滤波器实现图像锐化,滤波函数=1-低通滤波器
    flag = "LaplacianHPF"
    dftHPF(image_path, flag)