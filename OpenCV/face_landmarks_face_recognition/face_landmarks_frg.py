# 1 加入库
import cv2
import face_recognition
import face_landmarks_face_recognition
import matplotlib.pyplot as plt

# 2 方法：显示图片
def show_image(image,title):
    plt.title(title)
    plt.imshow(image)
    plt.axis("off")
# 3 方法：绘制landmars关键点
def show_landmarks(image,landmarks):
    for landmarks_dict in landmarks:
        for landmarks_key in landmarks_dict.keys():
            for point in landmarks_dict[landmarks_key]:
                cv2.circle(image,point,3,(0,0,70),1,-1)
    return image
# 4 主函数
def main():
    # 5 读取图片
    image = cv2.imread("family.jpg")
    # 6 灰度转换
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # 7 调用方法（face_rec）face_landmarks()
    face_marks = face_recognition.face_landmarks(gray,None,"large")

    # 8 调用绘制关键点方法
    img_result = show_landmarks(image,face_marks)
    # 9 创建画布
    plt.figure(figsize=(12,7))
    plt.suptitle("fneariofhnweioafiwea",fontsize=14)

    # 10 显示整体效果
    plt.imshow(img_result)
    plt.show()
if __name__ == '__main__':
    main()