import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import copy

def onMouseAction(event, x, y, flags, param):    # 鼠标交互（单机选点，右击完成）
    setpoint = (x, y)
    if event == cv.EVENT_LBUTTONDOWN:    # 单击
        pts.append(setpoint)
        print("选择顶点{}：{}".format(len(pts), setpoint))

def xieqie(image_path):
    img = cv.imread(image_path)
    h, w = img.shape[:2]

    angle = 20 * np.pi/180    # 斜切角度

    # 水平斜切
    MAS = np.float32([[1, np.tan(angle), 0], [0, 1, 0]])    # 斜切变换矩阵
    wShear = w + int(h * abs(np.tan(angle)))    # 调整宽度
    imgShearH = cv.warpAffine(img, MAS, (wShear, h))

    cv.imwrite(r"D:\bgi_project\demo\output\imgShearH.jpg", imgShearH)

    # 垂直斜切
    MAS = np.float32([[1, 0, 0], [np.tan(angle), 1, 0]])  # 斜切变换矩阵
    hShear = h + int(w * abs(np.tan(angle)))  # 调整宽度
    imgShearW = cv.warpAffine(img, MAS, (w, hShear))

    cv.imwrite(r"D:\bgi_project\demo\output\imgShearW.jpg", imgShearW)

def touying_bianhuan(image_path):
    image = cv.imread(image_path)
    imgCopy = copy.deepcopy(image)
    # 鼠标交互从输入图像选择4个顶点
    print("单击左键选择4个顶点（左上-左下-右下-右上）：")
    pts = []  # 初始化ROI坐标顶点集合
    status = True  # 进入绘图状态
    cv.namedWindow('origin')  # 创建图像显示窗口
    cv.setMouseCallback('origin', onMouseAction, status)  # 绑定回调函数
    while True:
        if len(pts) > 0:
            cv.circle(imgCopy, pts[-1], 5, (0, 0, 255), -1)  # 绘制最近的一个顶点
        if len(pts) > 1:
            cv.line(imgCopy, pts[-1], pts[-2], (255, 0, 0), 2)  # 绘制最近的一段线段
        if len(pts) == 4:  # 已有4个顶点，结束绘制
            cv.line(imgCopy, pts[0], pts[-1], (255, 0, 0), 2)  # 绘制最后的一段线段
            cv.imshow('origin', imgCopy)
            cv.waitKey(1000)
            break
        cv.imshow('origin', imgCopy)
        cv.waitKey(100)
    cv.destroyAllWindows()
    ptsSrc = np.array(pts)  # 列表转换为（4，2）numpy数组
    print(ptsSrc)

    # 计算投影变换矩阵MP
    ptsSrc = np.float32(pts)
    x1, y1, x2, y2 = int(0.1 * w), int(0.1 * h), int(0.9 * w), int(0.9 * h)
    ptsDst = np.float32([[x1, y1], [x1, y2], [x2, y2], [x2, y1]])  # 投影变换后的4个顶点坐标
    MP = cv.getPerspectiveTransform(ptsSrc, ptsDst)

    # 投影变换
    dsize = (256, 256)  # 输出图像尺寸(w, h)
    perspect = cv.warpPerspective(image, MP, dsize, borderValue=(255, 255, 255))
    print(image.shape, ptsSrc.shape, ptsDst.shape)

    cv.imwrite(r"D:\bgi_project\demo\output\perspect.jpg", perspect)

def remap_basic(image_path):
    image = cv.imread(image_path)
    height, width = image.shape[:2]
    # 初始化
    mapx = np.zeros((height, width), np.float32)  # mapx是对x的映射，表示像素点在目标图像的列号
    mapy = np.zeros((height, width), np.float32)  # mapy是对y的映射，表示像素点在目标图像的行号
    for h in range(height):
        for w in range(width):
            mapx[h, w] = w  # 水平方向不变
            mapy[h, w] = h  # 垂直方向不变
    dst1 = cv.remap(image, mapx, mapy, cv.INTER_LINEAR)

    mapx = np.array([[i * 1.5 for i in range(width)] for j in range(height)], dtype=np.float32)
    mapy = np.array([[j * 1.5 for i in range(width)] for j in range(height)], dtype=np.float32)

    dst2 = cv.remap(image, mapx, mapy, cv.INTER_LINEAR)  # 尺寸缩放

    cv.imwrite(r"D:\bgi_project\demo\output\copy_remap.jpg", dst1)
    cv.imwrite(r"D:\bgi_project\demo\output\resize_remap.jpg", dst2)

def updateMapXY(img, mapx, mapy, s):
    h, w = img.shape[:2]
    scale = 0.1 + 0.9 * s / 100
    padx = 0.5 * w * (1 - scale)    # 左右填充
    pady = 0.5 * h * (1 - scale)    # 上下填充
    mapx = np.array([[((j-padx)/scale) for j in range(w)] for i in range(h)], np.float32)
    mapy = np.array([[((i - pady) / scale) for j in range(w)] for i in range(h)], np.float32)
    return mapx, mapy

def remap_dynamic(image_path):
    img = cv.imread(image_path)
    h, w = img.shape[:2]
    mapx = np.zeros(img.shape[:2], np.float32)
    mapy = np.zeros(img.shape[:2], np.float32)
    borderColor = img[-1, -1, :].tolist()
    dst = np.zeros(img.shape, np.uint8)
    print(img.shape, dst.shape)

    for s in range(100):
        key = 0xFF & cv.waitKey(10)    # 按Esc退出
        if key == 27:    #esc to exit
            break
        mapx, mapy = updateMapXY(img, mapx, mapy, s)
        dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR, borderValue=borderColor)
        cv.imshow('RemapWin', dst)
    cv.destroyAllWindows()

    plt.figure(figsize=(9, 3.5))
    sList = [20, 50, 80]
    for i in range(len(sList)):
        mapx, mapy = updateMapXY(img, mapx, mapy, s=sList[i])
        dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR, borderValue=borderColor)
        plt.subplot(1, 3, i+1), plt.title("Dynamic (t={})".format(sList[i]))
        plt.axis('off'), plt.imshow(cv.cvtColor(dst, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()

def main():
    # 图像的斜切（扭变）
    image_path = r"D:\bgi_project\demo\data\stomics_center.jpg"
    xieqie(image_path)

    # 基于投影变换实现图像校正，简易理解为按照顺序点4个点，输出为点内区域；实现功能为：如果图像中的物体原始是歪着的，那么可以通过截取四个顶点，矫正为正着的
    touying_bianhuan(image_path)

    # 图像的重映射方法实现图像的复制和缩放
    remap_basic(image_path)

    # 基于图像重映射实现动画播放效果
    remap_dynamic(image_path)


if __name__ == '__main__':
    main()