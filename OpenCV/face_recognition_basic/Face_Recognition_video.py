# 1 导入库
import cv2

# 3 方法：绘制图片中检测到的人脸
def plot_rectangle(image,faces):
    #faces会返回四个值 ： 坐标(x,y),宽高(width,height)
    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),3)
    return image

# 4 主函数
def main():
    # 5 读取摄像头
    capture = cv2.VideoCapture(0)

    # 7 基于OpenCV自带的方法 cv2.CascadeClassfier()加载级联分类器
    face_alt2 = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")

    # 判断摄像头是否正常工作
    if capture.isOpened()is False:
        print("Camera Error")

    while True:
        # 获取每一帧
        ret, frame = capture.read()
        # 灰度转换
        if ret:
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            # 8 通过分类器对人脸进行检测
             #针对gray进行人脸检测返回结果
            face_alt2_detect = face_alt2.detectMultiScale(gray)

            # 9 调用方法绘制图片中检测到的人脸
            face_alt2_result = plot_rectangle(frame,face_alt2_detect)

            cv2.imshow("face detection",face_alt2_result)

            if cv2.waitKey() & 0xFF == ord('q'):
                break
    capture.release()
    cv2.destroyAllWindows()

# 12 主程序入口
if __name__ == '__main__':
    main()