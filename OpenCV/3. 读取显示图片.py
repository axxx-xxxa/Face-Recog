# 1 导入库
import cv2
import argparse

# 2 获取参数
parser = argparse.ArgumentParser()

# 3 添加参数
parser.add_argument("path_image",help="path to input the image")

# 4 解析参数
args = parser.parse_args()

# 5 加载图片, 方式一
img = cv2.imread(args.path_image)
cv2.imshow("1",img)

# 6 加载图片, 方式二
args_dict = vars(parser.parse_args())
#  （键）“path_image" : （值）"images/logo.png"
img2 = cv2.imread(args_dict['path_image'])
cv2.imshow("2",img2)

# 7
cv2.waitKey(0)
# 8
cv2.destroyWindow()