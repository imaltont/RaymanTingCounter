import cv2
import numpy as np
from matplotlib import pyplot as plt

def binary_tings(col_img):
    th1 = (col_img[:,:,0] > 200) & (col_img[:,:,0] < 255)
    th2 = (col_img[:,:,1] > 100) & (col_img[:,:,1] < 165)
    th3 = (col_img[:,:,2] > 170) & (col_img[:,:,2] < 235) 
    th4 = th1 & th2 & th3
    return th4

def get_NN(file_name = ""):
    raise NotImplementedError

def save_NN(file_name = ""):
    raise NotImplementedError

def train_NN():
    raise NotImplementedError

def inference():
    raise NotImplementedError
if __name__=='__main__':
    im = cv2.imread('Example3.jpg')
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = binary_tings(im)
    im.dtype="uint8"
    cv2.imshow("Testing",im*255)
    cv2.waitKey(0)
