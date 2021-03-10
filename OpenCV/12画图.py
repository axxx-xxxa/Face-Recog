# 1 导入库
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 2 定义颜色 字典
colors={'blue':(255,0,0),
        'green':(0,255,0),
        'red':(0,0,255),
        'white':(255,255,255)}

# 3 方法：显示图片
def show_image(image,title):
    img_RGB = image[:,:,::-1]
    plt.title(title)
    plt.imshow(img_RGB)
    plt.show()

# 4 画布
canvas = np.zeros((400,400,3),np.uint8) #默认背景黑色
canvas[:] = colors['white']
# show_image(canvas,"background")

# 5 画直线
cv2.line(canvas,(0,0),(400,400),colors['red'],5)
# show_image(canvas,"line")

# 6 长方形
canvas = np.zeros((400,400,3),np.uint8) #默认背景黑色
canvas[:] = colors['white']
cv2.rectangle(canvas,(10,10),(50,50),colors['red'],5)
# show_image(canvas,"rectangle")

# 7 圆形
canvas = np.zeros((400,400,3),np.uint8) #默认背景黑色
canvas[:] = colors['white']
cv2.circle(canvas,(200,200),100,colors['red'],5)
# show_image(canvas,"circle")

# 8 折线
canvas = np.zeros((400,400,3),np.uint8) #默认背景黑色
canvas[:] = colors['white']
pts = np.array([[250,5],[220,80],[280,80]],np.int32)
pts = pts.reshape((-1,1,2))
cv2.polylines(canvas,[pts],True,colors['red'],5)
show_image(canvas,"polyline")








