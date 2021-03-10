import cv2
import torch
import numpy as np
## 彩色图片
img=cv2.imread("./datasets/lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg")
print(img.shape)
(b,g,r)=img[100,120]
print(b,g,r)
cv2.imshow("111",img)
cv2.waitKey(0)
cv2.destroyWindow()

## 灰度