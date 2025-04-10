import cv2 as cv
import numpy as np
import tifffile as tif

image = tif.imread(r"C:\Users\zoujialin\Desktop\gold\demo\B04272D4\matrix\B04272D4_matrix.tif")

image = cv.applyColorMap(image, cv.COLORMAP_JET)

tif.imwrite(r"C:\Users\zoujialin\Desktop\gold\demo\B04272D4\matrix\B04272D4_colormap.tif", image, compression='zlib')