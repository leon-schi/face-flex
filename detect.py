import numpy as np
from scipy import ndimage

import math
import os
import cv2
import dlib
import imutils

from imutils import face_utils

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from raster import rasterize_polygon

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./data/shape_predictor_68_face_landmarks.dat')

def plot_face_and_landmarks(image, landmarks):
    plt.imshow(image)
    plt.scatter(*landmarks.T, marker='.', color='g')
    plt.show()

def plot_face(image):
    plt.imshow(image)
    plt.show()

def calculate_angle(landmarks):
    leftEye = [36, 37, 38, 39, 40, 41]
    rightEye = [42, 43, 44, 45, 46, 47]
    vec = landmarks[rightEye].mean(axis=0) - landmarks[leftEye].mean(axis=0)
    return -math.degrees(math.atan(vec[1] / vec[0]))

def rotate_landmarks_and_image(landmarks, image, angle):
    theta = np.radians(-angle)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c,-s), (s, c)))

    width, height = image.shape[1], image.shape[0]
    rotated = imutils.rotate(image, angle)

    vec = np.array([width/2, height/2], dtype=np.int32)
    landmarks -= vec
    landmarks = R.dot(landmarks.T).T
    landmarks += vec
    return landmarks, image

def cut_image(landmarks, image, padding=5):
    left = int(landmarks[0][0] - padding)
    right = int(landmarks[16][0] + padding)
    top = int(landmarks[24][1] - padding)
    bottom = int(landmarks[8][1] + padding)
    
    width, height = right - left, bottom - top
    size = max(width, height)
    d = abs(width - height) / 2

    if width > height:
        top = int(top-d)
    else:
        left = int(left-d)
    
    image = image[top:top+size, left:left+size]
    landmarks -= np.array([left, top])

    image = cv2.resize(image, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
    landmarks = landmarks * 128 / size

    return landmarks, image

def mask_image(landmarks, image):
    width, height = image.shape[1], image.shape[0]

    start, end = landmarks[0][0], landmarks[16][0]
    landmarks = [[start, 0]] + landmarks[:17].tolist() + [[end, 0]]
    mask = rasterize_polygon(landmarks, width, height) * 255
    mask = ndimage.gaussian_filter(mask, 3) / 255
    
    image = image / 255
    image = (image.T * mask.T).T
    return image

def align_image(image, landmarks):
    angle = calculate_angle(landmarks)
    landmarks, image = rotate_landmarks_and_image(landmarks, image, angle)
    landmarks, image = cut_image(landmarks, image)
    image = mask_image(landmarks, image)

    plot_face(image)

def detect(filename):
    image = cv2.imread(os.path.join('../data/img_align_celeba/img_align_celeba', filename))
    #image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 1)

    for rect in rects:
        shape = predictor(gray, rect)
        landmarks = imutils.face_utils.shape_to_np(shape)

        align_image(image, landmarks)

if __name__ == "__main__":
    directory = '../data/img_align_celeba/img_align_celeba'
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    for filename in files:
        detect(filename)