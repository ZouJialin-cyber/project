import cv2 as cv
import numpy as np

def distanceWaterShed(image_path):
    '''
    分水岭算法步骤：
    1. 数据预处理，对输入图像进行灰度化，降噪
    2. 计算梯度，一次获取图像中像素的灰度变化信息，梯度大的地方对应图像的边缘（会有灰度值的跳跃）
    3. 标记极小值点，在梯度图像中找到局部极小值点，被视为不同区域的种子点，代表不同的物体或区域
    4. 分水岭变换，从种子点开始按照一定的规则逐步扩展区域，直到所有的像素都分配到某个区域中，同时在不同区域边界形成分水岭
    :param image_path: img path
    :return: imgWatershed[numpy]
    '''
    img = cv.imread(image_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 阈值分割，将灰度图像分为黑白二值图像
    ret, thresh = cv.threshold(gray, 128, 255, cv.THRESH_OTSU)
    # 形态学操作，生成确定背景区域sureBG
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)    # 开运算，消除噪点
    sureBG = cv.dilate(opening, kernel, iterations=3)    # 膨胀操作，生成确定背景区域
    # 距离变换，生成确定前景区域
    distance = cv.distanceTransform(opening, cv.DIST_L2, 5)
    _, sureFG = cv.threshold(distance, 0.1 * distance.max(), 255, cv.THRESH_BINARY)
    sureFG = np.uint8(sureFG)
    # 连通域处理
    ret, component = cv.connectedComponents(sureFG, connectivity=8)
    markers = component + 1
    kinds = markers.max()
    maxKind = np.argmax(np.bincount(markers.flatten()))
    markersBGR = np.ones_like(img) * 255
    for i in range(kinds):
        if (i != maxKind):
            colorKind = np.random.randint(0, 255, size=(1, 3))
            markersBGR[markers==i] = colorKind
    # 去除连通域中背景区域部分
    unknown = cv.subtract(sureBG, sureFG)
    markers[unknown==255] = 0
    # 用分水岭算法标注目标的轮廓
    markers = cv.watershed(img, markers)

    # 把轮廓添加到原始图像上
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[markers==1] = 255
    mask = cv.dilate(mask, kernel=np.ones((3, 3)))
    imgWatershed = img.copy()
    imgWatershed[mask==255] = [255, 0, 0]

    cv.imwrite(r"D:\bgi_project\demo\output\imgWatershed.jpg", imgWatershed)

def contoursWaterShed(image_path):
    img = cv.imread(image_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 梯度处理
    imgGauss = cv.GaussianBlur(gray, (5, 5), -1)
    grad = cv.Canny(imgGauss, 50, 150)

    # 查找梯度图像和绘制图像轮廓
    contours, hierarchy = cv.findContours(grad, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    markers = np.zeros(img.shape[:2], np.int32)
    for index in range(len(contours)):
        markers = cv.drawContours(markers, contours, index, (index, index, index),1, 8, hierarchy)
    ContoursMarkers = np.zeros(img.shape[:2], np.uint8)
    ContoursMarkers[markers > 0] = 255

    # 分水岭算法
    markers = cv.watershed(img, markers)
    WatershedMarkers = cv.convertScaleAbs(markers)

    # 用随机颜色填充图像
    bgrMarkers = np.zeros_like(img)
    for i in range(len(contours)):
        colorKind = np.random.randint(0, 255, size=(1, 3))
        bgrMarkers[markers == i] = colorKind
    bgrFilled = cv.addWeighted(img, 0.67, bgrMarkers, 0.33, 0)

    cv.imwrite(r"D:\bgi_project\demo\output\bgrFilled.jpg", bgrFilled)


if __name__ == '__main__':
    image_path = r"D:\bgi_project\demo\data\stomics_center.jpg"

    # 基于距离变换的分水岭算法
    distanceWaterShed(image_path)

    # 基于轮廓标记（先验知识）的分水岭算法
    contoursWaterShed(image_path)