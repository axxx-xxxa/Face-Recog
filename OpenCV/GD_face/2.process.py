# 1 导入库
import cv2
import os
import matplotlib.pyplot as plt
import dlib
import numpy as np
import face_recognition
# 2 方法：图片文件转化字典
def  read_directory(directory_name):
    file_list = [filename for filename in os.listdir(directory_name)]
    img_dict = {}
    for i in range(len(file_list)):
        img_dict[i] = os.listdir(directory_name + "/" + file_list[i])
    return file_list,img_dict
# 3 方法：显示图片
def show_image(image,title):
    img_RGB = image[:,:,::-1]
    plt.title(title)
    plt.imshow(img_RGB)
    plt.axis("off")
# 4 定义：关键点128D编码
def encoder_face(image,detector,predictor,encoder,unsample=1,jet=1):
    # 2.1 检测人脸
    faces = detector(image,unsample)
    # 2.2 每张人脸关键点检测
    faces_keypoints = [predictor(image,face)for face in faces]
    return [np.array(encoder.compute_face_descriptor(image,faces_keypoint,jet))for faces_keypoint in faces_keypoints]
        #通过这个方法直接把关键点编码
# 5 定义：人脸是否属于同一个人/计算欧氏距离
def compare_faces(face_encoding,test_encoding):
    return list(np.linalg.norm(np.array(face_encoding)-np.array(test_encoding),axis=1))

# 6 定义：比较之后输出对应名称
def compare_faces_order(face_encoding,test_encoding,names):
    distance = list(np.linalg.norm(np.array(face_encoding)-np.array(test_encoding),axis=1))
    return zip(*sorted(zip(distance,names)))

# 7 BGR2RGB
def BGR2RGB(img):
    img = img[:, :, ::-1]
    return img

def main():
    img_name, img_dict = read_directory("./lfw")
    print(img_name)
    print(img_dict)
    # 3 加载人脸检测器
    detector = dlib.get_frontal_face_detector()
    # 4 加载关键点的检测器
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    # 5 加载人脸特征编码模型
    encoder = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")





if __name__ == '__main__':
    main()