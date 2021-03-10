# 1 导入库
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 2 方法：显示图片
def show_image(image,title,pos):
    plt.subplot(3,2,pos)
    plt.title(title)
    image_RGB = image[:,:,::-1]
    plt.imshow(image_RGB)
    plt.axis("off")
# 3 方法：显示彩色直方图
def show_histogram(hist,title,pos,color):
    plt.subplot(3,2,pos)
    plt.title(title)
    plt.xlim([0,256])
    for h,c in zip(hist,color):#color传进来三个颜色的元组
        plt.plot(h,color=c)
# 4 方法：计算直方图
def calc_color_his(image):
    # b,g,r
    hist = []
    hist.append(cv2.calcHist([image],[0],None,[256],[0,256]))
    hist.append(cv2.calcHist([image], [1], None, [256], [0, 256]))
    hist.append(cv2.calcHist([image], [2], None, [256], [0, 256]))
    return hist

# 5 主函数
def main():
    # 5.1 画布
    plt.figure(figsize=(12,8))
    plt.suptitle("Color Hist",fontsize=4,fontweight="bold")
    # 5.2 读取
    img = cv2.imread("./datasets/lfw/Abel_Pacheco\Abel_Pacheco_0004.jpg")

    # 5.3 计算直方图
    img_hist = calc_color_his(img)

    # 5.4 显示
    show_image(img,"RGB img",1)
    show_histogram(img_hist,"RGB img hist",2,("b","g","r"))

    # 5.5 像素值+50
    M = np.ones(img.shape,dtype='uint8')*50
    added_img = cv2.add(img,M)
    added_img_hist =calc_color_his(added_img)
    show_image(added_img,"added",3)
    show_histogram(added_img_hist,"added hist",4,('b','g','r'))
    # 5.6 像素值-50
    subtract_img = cv2.subtract(img,M)
    subtract_img_his = calc_color_his(subtract_img)
    show_image(subtract_img,"sub",5)
    show_histogram(subtract_img_his,"sub hist",6,('b','g','r'))



    plt.show()
if __name__ == '__main__':
    main()