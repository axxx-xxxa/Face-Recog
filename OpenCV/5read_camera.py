# 1 导入库
import cv2
import argparse

# 2 获取参数
parser = argparse.ArgumentParser()

# 3 添加参数
parser.add_argument("index_camera",help="the camera ID",type=int)

# 4 解析参数
args = parser.parse_args()
print("the camera index:",args.index_camera)

# 5 捕获摄像头的视频
capture = cv2.VideoCapture(args.index_camera)

# 6 获取帧的宽度,高度,FPS
frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
#每秒钟闪过多少照片
fps = capture.get(cv2.CAP_PROP_FPS)
print("帧的宽度:{}".format(frame_width))
print("帧的高度:{}".format(frame_height))
print("FPS:{}".format(fps))

# 7 判断摄像头是否打开
if capture.isOpened() is False:
    print("Camera Error")

# 8 从摄像头读取视频直到关闭
while capture.isOpened():
    # 9 通过摄像头，捕获帧
    ret, frame = capture.read()
    # 10 转化为灰度帧
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # 11 显示每一帧(视频流)
    cv2.imshow("frame",frame)
    cv2.imshow("gray frame",gray_frame)
    # 12 键盘输入,关闭摄像头
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# 13 释放资源
capture.release()
# 14
cv2.destroyWindow()
