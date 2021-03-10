# 1 导入库
import cv2
import dlib
import numpy as np

# 2 定义：关键点128D编码
def encoder_face(image,detector,predictor,encoder,unsample=1,jet=1):
    # 2.1 检测人脸
    faces = detector(image,unsample)
    # 2.2 每张人脸关键点检测
    faces_keypoints = [predictor(image,face)for face in faces]
    return [np.array(encoder.compute_face_descriptor(image,faces_keypoint,jet))for faces_keypoint in faces_keypoints]
        #通过这个方法直接把关键点编码

# 3 定义：人脸是否属于同一个人/计算欧氏距离
def compare_faces(face_encoding,test_encoding):
    return list(np.linalg.norm(np.array(face_encoding)-np.array(test_encoding),axis=1))

# 4 定义：比较之后输出对应名称
def compare_faces_order(face_encoding,test_encoding,names):
    distance = list(np.linalg.norm(np.array(face_encoding)-np.array(test_encoding),axis=1))
    return zip(*sorted(zip(distance,names)))


def main():
    # 2 读取4张图片
    img1 = cv2.imread("guo.jpg")
    img2 = cv2.imread("liu1.jpg")
    img3 = cv2.imread("liu2.jpg")
    img4 = cv2.imread("liu3.jpg")
    test = cv2.imread("liu4.jpg")
    img1 = img1[:, :, ::-1]
    img2 = img2[:, :, ::-1]
    img3 = img3[:, :, ::-1]
    img4 = img4[:, :, ::-1]
    test = test[:, :, ::-1]

    img_names = ["guo.jpg","liu1.jpg","liu2.jpg","liu3.jpg"]
    # 3 加载人脸检测器
    detector = dlib.get_frontal_face_detector()
    # 4 加载关键点的检测器
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    # 5 加载人脸特征编码模型
    encoder = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
    # 6 调用编码方法：输出128D#
    img1_128D = encoder_face(img1,detector,predictor,encoder)[0]
    img2_128D = encoder_face(img2,detector,predictor,encoder)[0]
    img3_128D = encoder_face(img3,detector,predictor,encoder)[0]
    img4_128D = encoder_face(img4,detector,predictor,encoder)[0]
    test_128D = encoder_face(test,detector,predictor,encoder)[0]

    four_image_128D = [img1_128D,img2_128D,img3_128D,img4_128D]
    # 7 调用比较方法：计算距离，判断
    distance = compare_faces(four_image_128D,test_128D)
    # 8 调用方法 输出结果
    distance,name = compare_faces_order(four_image_128D,test_128D,img_names)
    print(len(img_names))
    print("distance:",distance)
    print("names:",name)
if __name__ == '__main__':
    main()