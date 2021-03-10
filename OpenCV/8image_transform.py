# 1 导入库
import cv2
import matplotlib.pyplot as plt
import numpy as np

# 2 加载图像
img=cv2.imread("./datasets/lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg")
plt.imshow(img)

# 3 高度，宽度，通道
height, width, channel = img.shape
print(height,width,channel)

# 4 图片放大，缩小
# cv2.resize()
resized_image = cv2.resize(img,(width*2,height*2),interpolation=cv2.INTER_LINEAR)
plt.imshow(resized_image)

height1, width1, channel1 = resized_image.shape
print(height1,width1,channel1)

small_img = cv2.resize(img,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_LINEAR)

height2, width2, channel2 = small_img.shape
print(height2,width2,channel2)


# 5 图像平移 cv2.warpAffine() M 变换矩阵
height, width = img.shape[:2]
M1 = np.float32([[1,0,100],[0,1,50]])#向右200 向下50
move_img = cv2.warpAffine(img,M1,(width,height))
plt.imshow(move_img)

# 6 图像旋转 M 中心点
height, width = img.shape[:2]
center = (width // 2.0,height //2.0)
M3 = cv2.getRotationMatrix2D(center,180,1) # 1表示没有缩放
rotation_img = cv2.warpAffine(img,M3,(width,height))
plt.imshow(rotation_img)

# 7 图片放射
# cv2.getAffineTransform(p1,p2) p1 p2 映射关系
# M4 计算变换矩阵
p1 = np.float32([[120,35],[215,45],[135,120]])
p2 = np.float32([[135,45],[300,110],[130,230]])
M4 = cv2.getAffineTransform(p1,p2)
trans_img = cv2.warpAffine(img,M4,(width,height))
plt.imshow(trans_img)

# 8 图片裁剪
crop_img = img[20:300,150:230]
plt.imshow(crop_img)

# 9 图像的位运算
# 长方形
rectangle = np.zeros((300,300),dtype='uint8')
#300*300画布
rect_img = cv2.rectangle(rectangle,(25,25),(275,275),255,-1)
#(25,25)(275,275)两个小正方形坐标 255颜色 -1边框粗细
plt.imshow(rect_img)
# 圆形
rectangle = np.zeros((300,300),dtype='uint8')
circle_img = cv2.circle(rectangle,(150,150),120,255,-1)
plt.imshow(circle_img)

# 与运算 cv2.bitwise_and()
and_img = cv2.bitwise_and(rect_img,circle_img)
plt.imshow(and_img)

# 或运算 cv2.bitwise_or()
or_img = cv2.bitwise_or(rect_img,circle_img)
plt.imshow(or_img)

# 10 图像的分离和融合
# 分离BGR split()
(B,G,R) = cv2.split(img) # 分离RGB
print(B.shape,G.shape,R.shape)

# 融合RGB merge
zeros = np.zeros(img.shape[:2],dtype='uint8')
plt.imshow(cv2.merge([zeros,zeros,R]))
plt.show()

# 11 颜色空间
#灰度
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#HSV (色度，饱和度，纯度)
hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
#