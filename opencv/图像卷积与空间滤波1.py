import cv2 as cv
import numpy as np


def imageFilter(image_path):
    img = cv.imread(image_path, flags=0)
    cv.imwrite(r"D:\bgi_project\demo\output\img.jpg", img)

    # （1）不对称卷积核
    kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    imgCor = cv.filter2D(img, -1, kernel)    # 相关运算
    kernelFlip = cv.flip(kernel, -1)    # 翻转卷积核
    imgConv = cv.filter2D(img, -1, kernelFlip)    # 卷积运算

    cv.imwrite(r"D:\bgi_project\demo\output\imgCor.jpg", imgCor)
    cv.imwrite(r"D:\bgi_project\demo\output\imgConv.jpg", imgConv)

    # （2）对称卷积核
    kernSymm = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    imgCorSymm = cv.filter2D(img, -1, kernSymm)
    imgConvSymm = cv.filter2D(img, -1, kernSymm)

    cv.imwrite(r"D:\bgi_project\demo\output\imgCorSymm.jpg", imgCorSymm)
    cv.imwrite(r"D:\bgi_project\demo\output\imgConvSymm.jpg", imgConvSymm)

    # （3）可分离卷积核：KernelXY = KernelX * KernelY
    kernX = np.array([[-1, 2, -1]], np.float32)    # 水平卷积核(1, 3)
    kernY = np.transpose(kernX)    # 垂直卷积核(3, 1),transpose方法实现转置
    kernXY = kernX * kernY    # 二维卷积核(3, 3)
    imgConvXY = cv.filter2D(img, -1, kernXY)    # 直接使用二维卷积核
    imgConvSep = cv.sepFilter2D(img, -1, kernX, kernY)    # 分离卷积核依次卷积

    cv.imwrite(r"D:\bgi_project\demo\output\imgConvXY.jpg", imgConvXY)
    cv.imwrite(r"D:\bgi_project\demo\output\imgConvSep.jpg", imgConvSep)


def meanFilter(image_path):
    img = cv.imread(image_path)

    # cv.blur实现均值滤波
    conv1 = cv.blur(img, (5, 5))
    # cv.boxFilter实现均值滤波
    conv2 = cv.boxFilter(img, -1, (11, 11))

    cv.imwrite(r"D:\bgi_project\demo\output\conv1.jpg", conv1)
    cv.imwrite(r"D:\bgi_project\demo\output\conv2.jpg", conv2)


def gaussFilter(image_path):
    img = cv.imread(image_path)

    ksize = (11, 11)
    GaussBlur11 = cv.GaussianBlur(img, ksize, 0)    # sigma由ksize计算
    ksize = (43, 43)
    GaussBlur43 = cv.GaussianBlur(img, ksize, 0)

    cv.imwrite(r"D:\bgi_project\demo\output\GaussBlur11.jpg", GaussBlur11)
    cv.imwrite(r"D:\bgi_project\demo\output\GaussBlur43.jpg", GaussBlur43)

def medianFilter(image_path):
    img = cv.imread(image_path)

    imgMedianBlur = cv.medianBlur(img, 5)

    cv.imwrite(r"D:\bgi_project\demo\output\imgMedianBlur.jpg", imgMedianBlur)

def staticticsSortFilter(image_path):
    img = cv.imread(image_path, flags=0)

    hImg, wImg = img.shape[:2]

    # 边界填充
    m, n = 3, 3    # 统计排序滤波器尺寸
    hPad, wPad = int((m-1)/2), int((n-1)/2)
    imgPad = cv.copyMakeBorder(img, hPad, hPad, wPad, wPad, cv.BORDER_REFLECT)

    imgMaximumF = np.zeros(img.shape)    # 最大值滤波器
    imgMinimumF = np.zeros(img.shape)  # 最小值滤波器
    imgMiddleF = np.zeros(img.shape)  # 中点滤波器
    imgAlphaF = np.zeros(img.shape)    # 修正阿尔法滤波器
    for h in range(hImg):
        for w in range(wImg):
            # 当前像素的邻域
            neighborhood = imgPad[h:h+m, w:w+n]
            padMax = np.max(neighborhood)
            padMin = np.min(neighborhood)

            # 最大值滤波器
            imgMaximumF[h, w] = padMax

            # 最小值滤波器
            imgMinimumF[h, w] = padMin

            # 中点滤波器
            imgMiddleF[h, w] = int(padMax/2 + padMin/2)    # 分别除以二再相加，防止溢出

            # 修正阿尔法滤波器
            d = 2    # 修正值
            neighborSort = np.sort(neighborhood.flatten())    # 邻域像素按灰度值进行排序
            sumAlpha = np.sum(neighborSort[d:m*n-d-1])    #    删除d个最大值，d个最小值
            imgAlphaF[h, w] = sumAlpha / (m*n-2*d)    # 对剩余像素进行算术平均

    cv.imwrite(r"D:\bgi_project\demo\output\imgMaximumF.jpg", imgMaximumF)
    cv.imwrite(r"D:\bgi_project\demo\output\imgMinimumF.jpg", imgMinimumF)
    cv.imwrite(r"D:\bgi_project\demo\output\imgMiddleF.jpg", imgMiddleF)
    cv.imwrite(r"D:\bgi_project\demo\output\imgAlphaF.jpg", imgAlphaF)

def aotuPartFilter(image_path):
    img = cv.imread(image_path, flags=0)
    hImg, wImg = img.shape[:2]

    # 边界填充
    m, n = 5, 5    # 统计排序滤波器尺寸
    hPad, wPad = int((m-1)/2), int((n-1)/2)
    imgPad = cv.copyMakeBorder(img, hPad, hPad, wPad, wPad, cv.BORDER_REFLECT)
    # 估计原始图像的噪声方差
    mean, stddev = cv.meanStdDev(img)    # 图像均值，标准差
    varImg = stddev**2    # 方差
    # 自适应局部降噪
    epsilon = 1e-8
    imgAdaptLocal = np.zeros(img.shape)
    for h in range(hImg):
        for w in range(wImg):
            neighborhood = imgPad[h:h+m, w:w+n]
            meanSxy, stddevSxy = cv.meanStdDev(neighborhood)    # 邻域局部均值，标准差
            varSxy = stddevSxy**2    # 邻域局部方差
            ratioVar = min(varImg / (varSxy + epsilon), 1.0)    # 加性噪声
            imgAdaptLocal[h, w] = img[h, w] - ratioVar * (img[h, w] - meanSxy)

    cv.imwrite(r"D:\bgi_project\demo\output\imgAdaptLocal.jpg", imgAdaptLocal)

def autoMedianFilter(image_path):
    img = cv.imread(image_path, flags=0)
    hImg, wImg = img.shape[:2]

    # 边界填充
    smax = 7    # 允许最大的窗口尺寸
    m, n = smax, smax
    hPad, wPad = int((m-1)/2), int((n-1)/2)
    imgPad = cv.copyMakeBorder(img, hPad, hPad, wPad, wPad, cv.BORDER_REFLECT)

    imgAdaptMedianFilter = np.zeros(img.shape)    # 自适应中值滤波器
    for h in range(hPad, hPad+hImg):
        for w in range(wPad, wPad+wImg):
            ksize = 3    # 自适应邻域窗口初值
            zxy = img[h-hPad, w-wPad]
            while True:
                k = ksize // 2
                win = imgPad[h-k:h+k+1, w-k:w+k+1]
                zmin, zmed, zmax = np.min(win), np.median(win), np.max(win)
                if zmin < zmed < zmax:    # zmed不是噪声
                    if zmin < zxy < zmax:
                        imgAdaptMedianFilter[h-hPad, w-wPad] = zxy
                    else:
                        imgAdaptMedianFilter[h-hPad, w-wPad] = zmed
                    break
                else:
                    if ksize >= smax:    # 达到最大窗口
                        imgAdaptMedianFilter[h-hPad, w-wPad] = zmed
                        break
                    else:
                        ksize = ksize + 2    # 增大窗口尺寸

    cv.imwrite(r"D:\bgi_project\demo\output\imgAdaptMedianFilter.jpg", imgAdaptMedianFilter)

def bilateralFilter(image_path):
    img = cv.imread(image_path)
    hImg, wImg = img.shape[:2]

    imgBilateralFilter = cv.bilateralFilter(img, 9, sigmaColor=40, sigmaSpace=10)

    cv.imwrite(r"D:\bgi_project\demo\output\imgBilateralFilter.jpg", imgBilateralFilter)

def imgSharpen(img_path):
    img = cv.imread(img_path)

    # 对原始图像进行高斯平滑
    imgGaussBlur = cv.GaussianBlur(img, (11, 11), 0)

    # 掩蔽模板mask：原始图像减去平滑图像
    mask = cv.subtract(img, imgGaussBlur)

    # mask与img相加
    maskWeak = cv.multiply(mask, 0.5)    # k < 1，减弱钝化遮蔽，对图像锐化程度要求不高，且希望尽量保持图像原有风貌的场景
    imgWeak = cv.add(img, maskWeak)

    imgWeakAll = cv.add(img, mask)    # k = 1，钝化遮蔽（图像锐化），让图像的边缘和细节更加清晰，本质上是梯度算法

    maskEnhance = cv.multiply(mask, 2.0)    # k > 1，高提升滤波，能够更突出地增强图像的边缘和细节，可能会放大图像中的噪声
    imgEnhance = cv.add(img, maskEnhance)

    cv.imwrite(r"D:\bgi_project\demo\output\imgWeak.jpg", imgWeak)
    cv.imwrite(r"D:\bgi_project\demo\output\imgWeakAll.jpg", imgWeakAll)
    cv.imwrite(r"D:\bgi_project\demo\output\imgEnhance.jpg", imgEnhance)

if __name__ == '__main__':
    image_path = r"D:\bgi_project\demo\data\stomics_center.jpg"

    # 图像的卷积运算与相关运算
    imageFilter(image_path)

    # 空间滤波之盒式低通滤波器（均值滤波），即每个像素点的值等于邻域内的值的均值
    meanFilter(image_path)

    # 空间滤波之高斯低通滤波，图像边缘较模糊
    gaussFilter(image_path)

    # 空间滤波之中值滤波器
    medianFilter(image_path)

    # 统计排序滤波器：最大、最小值滤波器，中点滤波器，阿尔法修正滤波器
    staticticsSortFilter(image_path)

    # 自适应局部降噪滤波器
    aotuPartFilter(image_path)

    # 自适应中值滤波器，根据中值是否为噪声，适当调整窗口尺寸
    autoMedianFilter(image_path)

    # 双边滤波器，取决于邻域的像素值和当前像素与邻域像素的灰度差，既考虑了颜色空间，又考虑了坐标空间，保持图像边缘清晰
    bilateralFilter(image_path)

    # 图像的钝化掩蔽（图像锐化），锐化滤波，让图像的边缘和细节更清晰
    imgSharpen(image_path)