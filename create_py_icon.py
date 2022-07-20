import cv2
import numpy as np

raw = cv2.imread("data/img/raw.png", -1)
extension = np.zeros((80, 5, 4), dtype=np.uint8)
py_icon_hr = np.concatenate((extension, raw, extension), axis=1)
py_icon = cv2.resize(py_icon_hr, (32, 32), interpolation=cv2.INTER_AREA)
cv2.imwrite("data/img/py_icon.png", py_icon)
