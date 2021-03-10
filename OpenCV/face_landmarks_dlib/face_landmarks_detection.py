# 1 导入库
import cv2
import matplotlib.pyplot as plt
import numpy as np
import dlib
# 2 读取一张照片
image = cv2.imread("../face_landmarks_face_recognition/family.jpg")
# 3 调用人脸检测器
detector = dlib.get_frontal_face_detector()
# 4 加载预测关键点模型（68个关键点）
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# 5 灰度转换
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# 6 人脸检测
faces = detector(gray,1)
print(faces)
# 7 循环，遍历每一张人脸，给人脸绘制矩形框
for face in faces:
    # 8 绘制矩形框
    print(face.left())
    cv2.rectangle(image,(face.left(),face.top()),(face.right(),face.bottom()),(255,0,0),3)
    # 9 预测关键点
    shape = predictor(image,face)
    print(type(shape))
    # 10获取关键点坐标
    for pt in shape.parts():
        # 获取横纵坐标
        pt_position = (pt.x,pt.y)
        # 11绘制关键点坐标
        cv2.circle(image,pt_position,2,(255,255,255),-1)
#12 显示整个效果图
plt.imshow(image)
plt.axis("off")
plt.show()