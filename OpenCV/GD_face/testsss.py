# 1 库
import cv2
import os
import dlib
import matplotlib.pyplot as plt
import numpy as np
import time
from face_recognition_dlib.face_rg_dlib import encoder_face

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def  read_directory(directory_name):
    file_list = [filename for filename in os.listdir(directory_name)]
    #建立字典
    img_dict = {}
    #读取所有值
    for i in range(len(file_list)):
        img_dict[i] = os.listdir(directory_name + "/" + file_list[i])
    #键值对对应
    img_dict_num = img_dict.copy()
    for i in range(len(file_list)):
        img_dict.update({file_list[i]:img_dict.pop(i)})
    #检查
    #for i in img_dict.keys():
    #     print(i,img_dict[i])
    return file_list,img_dict,img_dict_num

# 3 方法：用于存储所有图片
def get_all_image(file_list,img_dict,img_dict_num,detector, predictor, encoder):
    #存放图片
    image_all=[]
    for i in range(len(img_dict)):
        for j in range(len(img_dict_num[i])):
            # 3.1 读取image
            image = cv2.imread("./lfw/"+file_list[i]+"/"+"".join(img_dict[file_list[i]][j]))
            image = image[:, :, ::-1]
            image = encoder_face(image, detector, predictor, encoder)
            # 3.2 存进image_all
            image_all.append(image)
            print(i,j)

    return image_all


# 4 方法：显示图片
def show_image(image,title):
    img_RGB = image[:,:,::-1]
    plt.title(title)
    plt.imshow(img_RGB)
    plt.axis("off")

# 5 方法：128编码
def encoder_face(image,detector,predictor,encoder,unsample=1,jet=1):
    # 检测人脸
    faces = detector(image,unsample)
     # 每张人脸关键点检测
    faces_keypoints = [predictor(image,face)for face in faces]

    return [np.array(encoder.compute_face_descriptor(image,faces_keypoint,jet))for faces_keypoint in faces_keypoints]
        #通过这个方法直接把关键点编码
# 图片对应名字
def image_name_define(img_dict_num):
    img_name=[]
    for value in img_dict_num.values():
        for i in range(len(value)):
            img_name.append(value[i])
    return img_name

def contrast(image1,image_all,image_name):
    print("目标图片",image_name[34])
    for i in range(len(image_all)):
        # print(np.sqrt(np.sum(np.square(np.array(image1) - np.array(image_all[i])))))
        if(np.sqrt(np.sum(np.square(np.array(image1) - np.array(image_all[i]))))<0.4):
            print("相似图片",image_name[i])
def main():
    start1 = time.time()
#---------------------------------------------------------------------
    # 获取文件字典
    file_list,img_dict,img_dict_num= read_directory("./lfw")
    # 3 加载人脸检测器
    detector = dlib.get_frontal_face_detector()
    # 4 加载关键点的检测器
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    # 5 加载人脸特征编码模型
    encoder = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
    image_all = get_all_image(file_list,img_dict,img_dict_num,detector, predictor, encoder)
    image_name=image_name_define(img_dict_num)
    contrast(image_all[34],image_all,image_name)
#---------------------------------------------------------------------
    end1=time.time()
    print(end1-start1)
if __name__ == '__main__':
    main()
