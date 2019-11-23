import dlib
import cv2
import os
import imutils
import math
import numpy as np
from PIL import Image

from imutils import face_utils
from scipy import ndimage
import matplotlib.pyplot as plt

import torch
from torchvision import transforms

from raster import rasterize_polygon
from image import FaceImage

from modifier import Modifier

from vae import model

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./data/shape_predictor_68_face_landmarks.dat')

class FaceBoundingBox:
    def __init__(self, top, left, size, angle):
        self.top = top
        self.left = left
        self.size = size
        self.angle = angle

class AlignedFace:
    def __init__(self, face, mask, face_bounding_box): 
        self.face = face
        self.mask = mask
        self.face_bounding_box = face_bounding_box

    def apply_mask(self):
        image = self.face.image 
        image = (image.T * self.mask.T).T
        return image

    def plot(self, plt):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

        ax1.imshow(self.face.image)
        ax1.scatter(*self.face.landmarks.T, marker='.', color='r')
        
        ax2.imshow(self.mask, cmap='gray')
        
        ax3.imshow(self.apply_mask())

class FaceTransformer:
    def __init__(self):
        self.faces = []

    def detect_faces(self, filename):
        self.image = cv2.imread(filename)
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        rects = detector(gray, 1)

        for rect in rects:
            shape = predictor(gray, rect)
            landmarks = imutils.face_utils.shape_to_np(shape)

            face = FaceImage(self.image, landmarks)
            self.faces.append(self.align_face(face))

    def align_face(self, face):
        angle = face.calculate_align_angle()
        face.rotate(angle)
        top, left, size = face.get_face_bounding_box()
        face.cut_to_square(top, left, size)
        bounding_box = FaceBoundingBox(top, left, size, angle)
        mask = face.get_morphing_mask(bounding_box.size)

        return AlignedFace(face, mask, bounding_box)

    def morph_image(self, aligned_face):
        height, width, _ = self.image.shape
        face = aligned_face.face.copy()
        
        face.realign_into_bounding_box(aligned_face.face_bounding_box, width, height)
        return face.morph_with(self.image, aligned_face.face_bounding_box.size)

    def plot(self, plt):
        plt.imshow(self.image)

        for face in self.faces:
            plt.scatter(*face.landmarks.T, marker='.')
            face.aligned_face.plot(plt)

if __name__ == "__main__":
    transformer = FaceTransformer()
    transformer.detect_faces('./images/000055.jpg')

    aligned_face = transformer.faces[0]

    modifier = Modifier(aligned_face.face, attribute='Smiling', i=5)
    modifier.addExpression(3)
    aligned_face.face = modifier.decode_face()

    aligned_face.plot(plt)
    plt.show()

    img = transformer.morph_image(aligned_face)

    image = Image.fromarray(np.uint8(img * 255))
    image.show()