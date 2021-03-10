# 1 导入库
import dlib
import cv2
import matplotlib.pyplot as plt
# 2 方法：人脸绘制矩形框
def plot_rectangle(image,faces):
    for face in faces:
        cv2.rectangle(image,(face.left(),face.top()),(face.right(),face.bottom()),(255,0,0),3)
    return image
# 3 主函数
def main():
    # 4 打开摄像头，读取视频
    image=[]
    captrue = cv2.VideoCapture(0)
    # 5 判断摄像头是否工作
    if captrue.isOpened() is False:
        print("Camera Error")
    # 6 读取每一帧
    while True:
        ret,frame = captrue.read()
        if ret:
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            # 7 获取检测器(dlib)
            detector = dlib.get_frontal_face_detector()
            dets_result = detector(gray, 1)
            # 8 绘制结果
            dets_image = plot_rectangle(frame,dets_result)
            # 9 显示结果
            cv2.imshow("face detection with dlib",dets_image)
            # 10 按键退出
            if cv2.waitKey(1) == 27:
                image=dets_image
                break

    # 11 释放资源
    captrue.release()
    cv2.destroyAllWindows()
    show_image(dets_image,'dsb')
    plt.show()
def show_image(image,title):
    img_RGB = image[:,:,::-1]
    plt.title(title)
    plt.imshow(img_RGB)
    plt.axis("off")
if __name__ == '__main__':
    main()