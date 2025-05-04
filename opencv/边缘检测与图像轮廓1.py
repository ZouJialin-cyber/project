import cv2 as cv
import numpy as np

def compare(image_path):
    img = cv.imread(image_path, flags=0)

    # DoG：高斯差分，使用两个不同尺度的高斯核得到的结果做差
    GaussBlur1 = cv.GaussianBlur(img, (0, 0), sigmaX=1.0)
    GaussBlur2 = cv.GaussianBlur(img, (0, 0), sigmaX=2.0)
    imgDoG = cv.subtract(GaussBlur2, GaussBlur1)

    # LoG：高斯拉普拉斯，先使用高斯模糊，然后使用拉普拉斯提取边缘
    imgLoG = np.uint8(cv.Laplacian(GaussBlur1, cv.CV_32F, ksize=3))

    # Canny算子，抗噪性强，通过高低阈值确定边缘
    TL, TH = 50, 150
    imgCanny = cv.Canny(img, TL, TH)

    cv.imwrite(r"D:\bgi_project\demo\output\imgDoG.jpg", imgDoG)
    cv.imwrite(r"D:\bgi_project\demo\output\imgLoG.jpg", imgLoG)
    cv.imwrite(r"D:\bgi_project\demo\output\imgCanny.jpg", imgCanny)


if __name__ == '__main__':
    image_path = r"D:\bgi_project\demo\data\stomics_center.jpg"

    # DoG、LoG和Canny算子
    compare(image_path)