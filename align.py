import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import cv2

# # import defined functions
# from utils import mp_detector, plot_landmark

# image_dir = "..//image//yeji-itzy.jpg"
# image = cv2.imread(image_dir)
# plot_landmark(image_dir)

# rotation_matrix = np.array([])

def align_image(ax, ay, image):
    rotate_image = image.copy()
    x1, y1 = ax[1], ay[1]
    x2, y2 = ax[9], ay[9]
    m = (y1-y2)/(x1-x2)
    c = (x1*y2 - x2*y1)/(x1-x2)
    y = 0
    # line equation y = m*x + b
    x = (y - c)/m
    a = np.sqrt((ay[1]-0)**2)
    b = np.sqrt((ax[1]-abs(int(x)))**2 + (ay[1]-0)**2)
    angle = np.sin(a/b)*57.2958 - 45
    
    rows,cols, _= rotate_image.shape 
    M = cv2.getRotationMatrix2D((cols/2,rows/2), int(-angle), 1) 
    result = cv2.warpAffine(rotate_image, M, (cols,rows)) 
    return result

def align_two_image(ax, ay, bx, by, image):
    rotate_image = image.copy()

    x1, y1 = bx[1], by[1]
    x2, y2 = bx[9], by[9]
    m = (y1-y2)/(x1-x2)
    c = (x1*y2 - x2*y1)/(x1-x2)
    y = 0
    # line equation y = m*x + b
    x = (y - c)/m
    a = np.sqrt((by[1]-0)**2)
    b = np.sqrt((bx[1]-abs(int(x)))**2 + (by[1]-0)**2)
    angle_b = np.sin(a/b)*57.2958 
    
    # template angle
    x1, y1 = ax[1], ay[1]
    x2, y2 = ax[9], ay[9]
    m = (y1-y2)/(x1-x2)
    c = (x1*y2 - x2*y1)/(x1-x2)
    y = 0
    # line equation y = m*x + b
    x = (y - c)/m
    a = np.sqrt((ay[1]-0)**2)
    b = np.sqrt((ax[1]-abs(int(x)))**2 + (ay[1]-0)**2)
    angle = np.sin(a/b)*57.2958 - angle_b
    
    rows,cols, _= rotate_image.shape 
    M = cv2.getRotationMatrix2D((cols/2,rows/2), int(-angle), 1) 
    result = cv2.warpAffine(rotate_image, M, (cols,rows)) 
    return result