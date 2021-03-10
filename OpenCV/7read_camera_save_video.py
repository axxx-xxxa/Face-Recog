# 1 导入库
import cv2
import argparse

# 2 获取参数
parser = argparse.ArgumentParser()

# 3 添加参数
parser.add_argument("video_output",help="the path to the output video")

# 4 解析参数
args = parser.parse_args()

# 5 启动摄像头 （捕获摄像头）
capture = cv2.VideoCapture(0) # 0 1 2...共有几个摄像头

# 6 是否打开了摄像头
if capture.isOpened() is False:
    print("Camera Error")

# 7 获取帧的属性：宽,高,fps
frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = capture.get(cv2.CAP_PROP_FPS)

# 8 对视频进行编码
fourcc = cv2.VideoWriter_fourcc(*"XIVD")
# 1.路径 2.编码方式 3.帧 4.帧的属性 5.保存形式False 灰 True 彩
output_gray = cv2.VideoWriter(args.video_output,fourcc,int(fps),(int(frame_width),int(frame_height)),False)

# 9 读取摄像头
while capture.isOpened():
    ret, frame = capture.read()
    if ret is True:
        # 10 转成灰度保存
        gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        # 11 写入视频文件中
        output_gray.write(gray_frame)
        # 12 imshow显示
        cv2.imshow("gray",gray_frame)
        # 13
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
# 14
capture.release()
output_gray.release()
cv2.destroyAllWindows()