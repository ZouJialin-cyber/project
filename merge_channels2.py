"""
    根据track点坐标绘制配准图（by 冯柠）
"""
import cv2 as cv
import tifffile as tif
import numpy as np


def f_ij_16_to_8(img, chunk_size=1000):
    if img.dtype == 'uint8':
        return img
    dst = np.zeros(img.shape, np.uint8)
    p_max = np.max(img)
    p_min = np.min(img)
    scale = 256.0 / (p_max - p_min + 1)
    for idx in range(img.shape[0] // chunk_size + 1):
        sl = slice(idx * chunk_size, (idx + 1) * chunk_size)
        win_img = np.copy(img[sl])
        win_img = np.int16(win_img)
        win_img = (win_img & 0xffff)
        win_img = win_img - p_min
        win_img[win_img < 0] = 0
        win_img = win_img * scale + 0.5
        win_img[win_img > 255] = 255
        dst[sl] = np.array(win_img).astype(np.uint8)
    return dst

mark = ['ssDNA', 'HE', 'DAPI']

def enhance(img, mode, thresh):
    data = img.ravel()
    min_v = np.min(data)
    data_ = data[np.where(data <= thresh)]
    if len(data_) == 0:
        return 0, 0
    if mode == 'median':
        var_ = np.std(data_)
        thr = np.median(data_)
        max_v = thr + var_
    elif mode == 'hist':
        freq_count, bins = np.histogram(data_, range(min_v, int(thresh + 1)))
        count = np.sum(freq_count)
        freq = freq_count / count
        thr = bins[np.argmax(freq)]
        max_v = thr + (thr - min_v)

    return min_v, max_v

def encode(img, min_v, max_v):
    if min_v >= max_v:
        img = img.astype(np.uint8)
        return img
    mat = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    v_w = max_v - min_v
    mat[img < min_v] = 0
    mat[img > max_v] = 255
    pos = (img >= min_v) & (img <= max_v)
    mat[pos] = (img[pos] - min_v) * (255 / v_w)
    return mat

def f_gray2bgr(img):
    img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    return img

def dapi_enhance(img, depth):
    th = int(1 << depth) * (1 - 0.618)
    min_v, max_v = enhance(img, mode='hist', thresh=th)
    enhance_img = encode(img, min_v, max_v)
    bgr_img = f_gray2bgr(enhance_img)

    return bgr_img


def he_enhance(img):
    if img.ndim == 3:
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    if img.dtype == 'uint16':
        img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    MAX_RANGE = np.power(2, 8)
    img_invert = (MAX_RANGE - img).astype(np.uint8)
    img_invert = cv.equalizeHist(img_invert)
    bgr_arr = f_gray2bgr(img_invert)
    return bgr_arr

def pts_on_img(img, pts, radius=5, color=(255, 0, 0), thickness=2):
    """
    原函数功能是在图像上根据给定的点坐标绘制圆，现在在此基础上添加绘制水平和垂直的线，
    每条线长度为21像素（上下左右各10像素 + 中间点所在像素），线宽为1像素。
    """
    for pt in pts:
        pos = (int(pt[0]), int(pt[1]))
        # 绘制水平方向的线
        start_x = pos[0] - 10
        end_x = pos[0] + 10
        cv.line(img, (start_x, pos[1]), (end_x, pos[1]), color, 1)
        # 绘制垂直方向的线
        start_y = pos[1] - 10
        end_y = pos[1] + 10
        cv.line(img, (pos[0], start_y), (pos[0], end_y), color, 1)
        # 绘制原来的圆
        cv.circle(img, pos, radius, color, thickness)

    return img

def read_template_file(template_path):
    coords = []
    with open(template_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()  # 去除每行的空白字符（如换行符、空格等）
            if line:  # 跳过空行（如果有的话）
                values = list(map(float, line.split()))  # 将每行字符串按空格分割后转换为浮点数列表
                coords.append(values)  # 将坐标值列表添加到总的坐标列表中
    return np.array(coords)

def create_track_point(img_path, mark, template):
    """

    :param img_path: 图像路径
    :param mark: 染色类型
    :param template: track点坐标
    :return:
    """
    scale = 2
    rect = 750
    inward_distance = 2000

    image = cv.imread(img_path, -1)
    image = cv.resize(image, (int(image.shape[1] / scale), int(image.shape[0] / scale)))
    if image.ndim == 3:
        _image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        _image = image

    _image = f_ij_16_to_8(_image)
    _, thresh = cv.threshold(_image, 1, 255, cv.THRESH_BINARY)
    thresh = cv.medianBlur(thresh, 21)
    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)[0]
    _rect = cv.minAreaRect(contours)
    box = cv.boxPoints(_rect)
    box = np.intp(box)

    img = np.zeros((image.shape[0], image.shape[1], 3), dtype="uint8")

    hw = img.shape[:2]

    center_x = np.mean(box[:, 0])
    center_y = np.mean(box[:, 1])

    for i in range(len(box)):
        vector_x = box[i][0] - center_x
        vector_y = box[i][1] - center_y
        norm = np.sqrt(vector_x ** 2 + vector_y ** 2)
        if norm != 0:
            vector_x = vector_x / norm * inward_distance
            vector_y = vector_y / norm * inward_distance
            box[i][0] = int(box[i][0] - vector_x)
            box[i][1] = int(box[i][1] - vector_y)

    ir_0_0_y = box[0][1] if (box[0][1] - rect > 0 and box[0][1] < hw[1]) else rect
    ir_0_0_x = box[0][0] if (box[0][0] - rect > 0 and box[0][0] < hw[0]) else rect
    ir_0_1_y = box[1][1] if (box[1][1] - rect > 0 and box[1][1] < hw[1]) else rect
    ir_0_1_x = box[1][0] if (box[1][0] - rect > 0 and box[1][0] < hw[0]) else rect
    ir_1_0_y = box[2][1] if (box[2][1] - rect > 0 and box[2][1] < hw[1]) else rect
    ir_1_0_x = box[2][0] if (box[2][0] - rect > 0 and box[2][0] < hw[0]) else rect
    ir_1_1_y = box[3][1] if (box[3][1] - rect > 0 and box[3][1] < hw[1]) else rect
    ir_1_1_x = box[3][0] if (box[3][0] - rect > 0 and box[3][0] < hw[0]) else rect

    ir_0_0 = image[ir_0_0_y - rect:rect + ir_0_0_y, ir_0_0_x - rect:rect + ir_0_0_x]
    ir_0_1 = image[ir_0_1_y - rect:rect + ir_0_1_y, ir_0_1_x - rect:rect + ir_0_1_x]
    ir_1_0 = image[ir_1_0_y - rect:rect + ir_1_0_y, ir_1_0_x - rect:rect + ir_1_0_x]
    ir_1_1 = image[ir_1_1_y - rect:rect + ir_1_1_y, ir_1_1_x - rect:rect + ir_1_1_x]

    if mark == 'DAPI' or mark == 'ssDNA':
        template[:, :2] = template[:, :2] / scale
        enhance_arr_00 = dapi_enhance(ir_0_0, image.depth)
        enhance_arr_01 = dapi_enhance(ir_0_1, image.depth)
        enhance_arr_10 = dapi_enhance(ir_1_0, image.depth)
        enhance_arr_11 = dapi_enhance(ir_1_1, image.depth)
    elif mark == 'HE':
        template[:, :2] = template[:, :2] / scale
        enhance_arr_00 = he_enhance(ir_0_0)
        enhance_arr_01 = he_enhance(ir_0_1)
        enhance_arr_10 = he_enhance(ir_1_0)
        enhance_arr_11 = he_enhance(ir_1_1)
    else:
        print('error!check the stain')
        # error_reason = "check the stain"
        return None, None, None, None

    point_img = pts_on_img(img, template, radius=int(5 / scale))
    point_img_00 = point_img[ir_0_0_y - rect:rect + ir_0_0_y, ir_0_0_x - rect:rect + ir_0_0_x]
    point_img_01 = point_img[ir_0_1_y - rect:rect + ir_0_1_y, ir_0_1_x - rect:rect + ir_0_1_x]
    point_img_10 = point_img[ir_1_0_y - rect:rect + ir_1_0_y, ir_1_0_x - rect:rect + ir_1_0_x]
    point_img_11 = point_img[ir_1_1_y - rect:rect + ir_1_1_y, ir_1_1_x - rect:rect + ir_1_1_x]

    img_00 = cv.addWeighted(enhance_arr_00, 1, point_img_00, 0.5, 0)
    img_01 = cv.addWeighted(enhance_arr_01, 1, point_img_01, 0.5, 0)
    img_10 = cv.addWeighted(enhance_arr_10, 1, point_img_10, 0.5, 0)
    img_11 = cv.addWeighted(enhance_arr_11, 1, point_img_11, 0.5, 0)

    return img_00, img_01, img_10, img_11

img_path = r"C:\Users\zoujialin\Desktop\gold\demo\C03834C6\image\C03834C6_regist_img.tif"

template_path = r"C:\Users\zoujialin\Desktop\gold\demo\C03834C6\transformed_point.txt"

template = read_template_file(template_path)

img_00, img_01, img_10, img_11 = create_track_point(img_path, mark[1], template)

tif.imwrite(r"C:\Users\zoujialin\Desktop\gold\demo\C03834C6\tmp\img_00.tif", img_00)
tif.imwrite(r"C:\Users\zoujialin\Desktop\gold\demo\C03834C6\tmp\img_01.tif", img_01)
tif.imwrite(r"C:\Users\zoujialin\Desktop\gold\demo\C03834C6\tmp\img_10.tif", img_10)
tif.imwrite(r"C:\Users\zoujialin\Desktop\gold\demo\C03834C6\tmp\img_11.tif", img_11)