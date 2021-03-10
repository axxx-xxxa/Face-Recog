# 1 导入库
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 2 方法：显示图片
def show_image(image,title,pos):
    img_RGB = image[:,:,::-1]
    plt.subplot(2,2,pos)
    plt.title(title)
    plt.imshow(img_RGB)


# 3 方法：绘制图片中检测到的人脸
def plot_rectangle(image,faces):
    #faces会返回四个值 ： 坐标(x,y),宽高(width,height)
    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),3)
    return image

# 4 主函数
def main():

    # 5 读取素材图片
    image = cv2.imread("girls.jpg")

    # 6 转化为灰度图片
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    # 7 基于OpenCV自带的方法 cv2.CascadeClassfier()加载级联分类器
    face_alt2 = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")

    # 8 通过分类器对人脸进行检测
     #针对gray进行人脸检测返回结果
    face_alt2_detect = face_alt2.detectMultiScale(gray)

    # 9 调用方法绘制图片中检测到的人脸
    face_alt2_result = plot_rectangle(image.copy(),face_alt2_detect)

    # 10 创建画布显示
    plt.figure(figsize=(9,6))
    plt.suptitle("Face detection with Haar Cascade",fontsize=14,fontweight="bold")

    # 11 显示整个检测效果
    show_image(face_alt2_result,"face_alt2",1)
    plt.show()
# 12 主程序入口
if __name__ == '__main__':
    main()