import cv2 as cv
import numpy as np

def getBorder(image_path):
    img = cv.imread(image_path, flags=0)
    _, imgOSTU = cv.threshold(img, 128, 255, cv.THRESH_OTSU)

    element = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    imgErode = cv.erode(imgOSTU, kernel=element)
    imgBorder1 = imgOSTU - imgErode

    imgDilate = cv.dilate(imgOSTU, element)
    imgBorder2 = imgDilate - imgOSTU

    cv.imwrite(r"D:\bgi_project\demo\output\imgBorder1.jpg", imgBorder1)
    cv.imwrite(r"D:\bgi_project\demo\output\imgBorder2.jpg", imgBorder2)

def floodFill(image_path):
    img = cv.imread(image_path, flags=0)
    _, imgOSTU = cv.threshold(img, 128, 255, cv.THRESH_OTSU)

    imgBin = cv.bitwise_not(imgOSTU)    # 二值图像的补集（黑色背景），填充基准
    h, w = img.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)    # mask比img宽两个像素、高两个像素
    imgFloodFill = imgBin.copy()
    cv.floodFill(imgFloodFill, mask, (0, 0), newVal=225)    # 从背景像素原点(0, 0)开始
    imgRebuild = cv.bitwise_and(imgOSTU, imgFloodFill)    # 孔洞填充结果图像

    cv.imwrite(r"D:\bgi_project\demo\output\imgRebuild.jpg", imgRebuild)


if __name__ == '__main__':
    image_path = r"D:\bgi_project\demo\data\stomics_center.jpg"

    # 边界提取：img-腐蚀=内边界，膨胀-img=外边界
    getBorder(image_path)

    # 使用泛洪算法进行孔洞填充
    floodFill(image_path)