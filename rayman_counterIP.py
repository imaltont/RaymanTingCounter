import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('Example2.jpg') 

#img = cv2.medianBlur(img,5)

print(img[:,:,1].shape)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


#ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
#th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
th1 = cv2.adaptiveThreshold(img[:,:,0],255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
th2 = cv2.adaptiveThreshold(img[:,:,1],255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
#th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
th3_1 = (img[:,:,0] > 200) & (img[:,:,0] < 255)#img[:,:,0] > 200
th3_2 = (img[:,:,1] > 110) & (img[:,:,1] < 180)
th3_3 = (img[:,:,2] > 120) & (img[:,:,2] < 250) 
th3 = th3_1 & th3_2 & th3_3
#th3 = th3 == th3_3

titles = ['All channels', 'R', 'G', 'B']
images = [th3, th3_1, th3_2, th3_3]

for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i])
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()
