# 1 加载库
import cv2
import argparse

# 2 获取参数
parser = argparse.ArgumentParser()

# 3 添加参数
parser.add_argument("video_path",help="the path to the video file")

# 4 解析参数
args = parser.parse_args()

# 5 加载视频文件
capture = cv2.VideoCapture(args.video_path)

# 6 读取视频
ret ,frame = capture.read()
gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#ret 是否读取到了帧（图片）(一张)
while ret: #（继续读取帧）
    cv2.imshow("video",frame)
    cv2.imshow("gray_video",gray_frame)
    # 帧不断改变
    ret,frame = capture.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()