# 1 库
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

def main():
    a=np.array([1,2,3,4]);
    print(a)
    b=np.array([2,3,4,6]);
    print(type(a-b))
    print(np.sum(a-b))
if __name__ == '__main__':
    main()