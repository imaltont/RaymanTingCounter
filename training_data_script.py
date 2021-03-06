"""
Script for loading in a bunch of images and having human classify them. 
Used for generating training data for the neural network that will be used to count tings.
Requires:
    numpy
    opencv
"""

import numpy as np
import glob
import cv2
import csv
import sys, getopt
from pathlib import Path
from rayman_counterIP import binary_tings


def label_video(filename="", ting_number=(0,0), ting_location_y=(0,0), life_number=(0,0), life_location_y=(0,0)):
    cap = cv2.VideoCapture(filename)

    image_data = read_csv()
    print(image_data)
    image_ID = int(image_data[-1][0].strip(".png")) if image_data else 0
    print(image_ID)
    while(cap.isOpened()):
        ret, frame = cap.read()
        #do cropping here
        if frame is None:
            break
        numbers = crop_and_resize(frame)
        for x in numbers:
            #show image
            if x is None:
                cap.release()
                break
            cv2.imshow("Current_num",x)
            n = cv2.waitKey(0)
            if n-48 in range(0,10):
                value = n-48
            else:
                value = -1
            cv2.destroyAllWindows()
            cv2.waitKey(1)

            #Get value of number
            #value = input("Which number was it: ")

            #save image
            img_filename = "".join([str(image_ID), '.png'])
            image_data.append(("".join([img_filename]), value))
            img = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
            img = binary_tings(img)
            img.dtype='uint8'
            cv2.imwrite(str(Path('./training_data/training')/img_filename), img*255)
            image_ID = image_ID + 1
        if x is None:
            break
    save_csv(csv_data=image_data)
def crop_and_resize(image=None, ting_x=(285,298), ting_y=(10,10), life_x=(46, 63), life_y=(10,10)):
    image = cv2.resize(image, (320,240))
    image_list = [1 for x in range(4)]
    number_height = 23
    number_width = 15

    image_list[0] = image[ting_y[0]:ting_y[0]+number_height, ting_x[0]:ting_x[0]+number_width]
    image_list[1] = image[ting_y[1]:ting_y[1]+number_height, ting_x[1]:ting_x[1]+number_width]
    image_list[2] = image[life_y[0]:life_y[0]+number_height, life_x[0]:life_x[0]+number_width]
    image_list[3] = image[life_y[1]:life_y[1]+number_height, life_x[1]:life_x[1]+number_width]

    return image_list
def read_csv(filename="training_data/training_data.csv", csv_data=[]):
    image_list = []
    with open(filename, newline='') as csvfile:
        img_reader = csv.reader(csvfile,delimiter=',')
        for row in img_reader:
            image_list.append((row[0], int(row[1])))
    return image_list
def save_csv(filename="training_data.csv", csv_data=[]):
    with open(str(Path('./training_data') / filename), 'w', newline='') as csvfile:
        img_writer = csv.writer(csvfile, delimiter=',')
        for x in csv_data:
            img_writer.writerow(x)

if __name__=="__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:", ["help", "input="])
    except:
        print('training_data_script.py -i <inputfile>')
        sys.exit(2)
    for opt, argv in opts:
        if opt in ('-h', '--help'):
            print('training_data_script.py -i <inputfile>')
            sys.exit()
        elif opt in ('-i', '--input'):
            label_video(argv)
