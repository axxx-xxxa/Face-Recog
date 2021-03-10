# 1 导入库
import cv2
import dlib
# 2 打开摄像头
capture = cv2.VideoCapture(0)
# 3 获取人脸检测器
detector = dlib.get_frontal_face_detector()
# 4 获取人脸关键点检测模型
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
while True:
    # 5 读取视频流
    ret,frame = capture.read()
    # 6 灰度转换
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # 7 人脸检测
    faces = detector(gray,1)
    # 8 绘制矩形框关键点
    for face in faces:
        # 8.1 绘制矩形框
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (255, 0, 0), 3)
        # 8.2 检测到关键点
        shape = predictor(gray,face)
        # 8.3 获取关键点坐标
        for pt in shape.parts():
            pt_position = (pt.x,pt.y)
            # 8.4 绘制关键点
            cv2.circle(frame,pt_position,2,(255,255,255),-1)
    if cv2.waitKey(1) == 27:
         break
    cv2.imshow("face decet",frame)
# 11 释放资源
capture.release()
cv2.destroyAllWindows()


