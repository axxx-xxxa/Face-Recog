# 1 导入库
import cv2
import matplotlib.pyplot as plt
import numpy as np

# 2 方法：显示图片
def show_image(image,title,pos):
    # 顺序转换 BGR to RGB
    image_RGB = image[:, :, ::-1]#(height,width,channel(倒叙))
    # 显示标题
    plt.title(title)
    # 指定位置
    plt.subplot(2,3,pos)
    plt.imshow(image_RGB)

# 3 方法：显示图片的灰度直方图
def show_histogram(hist,title,pos,color):
    #显示标题
    plt.title(title)
    plt.subplot(2,3,pos)
    plt.xlabel("Bins")
    plt.ylabel("Pixels")
    plt.plot(hist,color=color)

# 4 主函数调用
def main():
    # 5 创建画布
    plt.figure(figsize=(15,6))# 画布大小
    plt.suptitle("灰度直方图",fontsize=14,fontweight="bold")

    # 6 加载图片
    img=cv2.imread("./datasets/lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg")

    # 7 灰度转换
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # 8 计算hist
    hist_image = cv2.calcHist(img_gray,[0],None,[256],[0,256])

    # 9 展示灰度直方图
    # 灰度图转换
    img_BGR = cv2.cvtColor(img_gray,cv2.COLOR_GRAY2BGR)
    show_image(img_BGR, "BGR image",1)
    show_histogram(hist_image,"gray image histogram ",4,"blue") 

    # 10 每个像素值加50
    M = np.ones(img_gray.shape,np.uint8) * 50

    added_img = cv2.add(img_gray,M)
    add_img_hist = cv2.calcHist([added_img],[0],None,[256],[0,256])
    added_img_BGR = cv2.cvtColor(added_img,cv2.COLOR_GRAY2BGR)
    show_image(added_img_BGR, "added image",2)
    show_histogram(add_img_hist,"added hist image",5,'red')

    # 11 每个像素值减50
    M = np.ones(img_gray.shape, np.uint8) * 50

    subtract_img = cv2.subtract(img_gray, M)
    subtract_img_hist = cv2.calcHist([subtract_img], [0], None, [256], [0, 256])
    subtract_img_BGR = cv2.cvtColor(subtract_img, cv2.COLOR_GRAY2BGR)
    show_image(subtract_img_BGR, "subtract image", 3)
    show_histogram(subtract_img_hist, "subtract hist image", 6, 'yellow')
    plt.show()


if __name__ == '__main__':
    main()



