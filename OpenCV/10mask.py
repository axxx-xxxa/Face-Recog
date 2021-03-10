# 1 导入库
import numpy as np
import matplotlib.pyplot as plt
import cv2

# 2 方法：显示图片
def show_image(image,title,pos):
    img_RGB = image[:, :, ::-1]
    plt.title(title)
    plt.subplot(2,2,pos)
    plt.imshow(img_RGB)

# 3 方法：显示灰度直方图
def show_histogram(hist,title,pos,color):
    plt.subplot(2,2,pos)
    plt.title(title)
    plt.xlim([0,256])
    plt.plot(hist,color=color)

# 4 main
def main():
    # 5 创建画布
    plt.figure(figsize=(12,7))
    plt.suptitle("Grayimg and Histogram with mask")

    # 6 读取图片并灰度转换,计算直方图,显示
    img_gray = img=cv2.imread("./datasets/lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg"
                              ,cv2.COLOR_BGR2GRAY)
    img_gray_hist = cv2.calcHist([img_gray],[0],None,[256],[0,256])
    show_image(img_gray,"img gray",1)
    show_histogram(img_gray_hist,"gray hist",3,"red")

    # 7 mask , 计算位图 , 直方图
    mask = np.zeros(img_gray.shape[:2],np.uint8)
    mask[25:175,55:185] = 255 # 获取mask并赋予颜色
    img_mask_hist = cv2.calcHist([img_gray],[0],mask,[256],[0,256])

    # 8 通过位运算（与运算）计算带有mask的灰度图片
    mask_img = cv2.bitwise_and(img_gray,img_gray,mask = mask)

    # 9 显示带有mask的图片和直方图
    show_image(mask_img,"mask img",2)
    show_histogram(img_gray_hist,"mask hist",4,"blue")
    plt.show()



if __name__ == '__main__':
    main()