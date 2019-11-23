import numpy as np
import math
import imutils
import cv2

from raster import rasterize_polygon
from scipy import ndimage

""""
All images in this class shall be represented as float arrays of shape (height, width, 3) with a range of 0..1
"""

class FaceImage:
    def __init__(self, image, landmarks):        
        self.image = np.copy(image)
        self.landmarks = np.copy(landmarks)
        self.norm_image()

    def plot(self, plt, show_landmarks=True):
        plt.imshow(self.image)
        if show_landmarks:
            plt.scatter(*self.landmarks.T, marker='.', color='r')

    def copy(self):
        return FaceImage(np.copy(self.image), np.copy(self.landmarks))

    def width(self): return self.image.shape[1]
    def height(self): return self.image.shape[0]

    def calculate_align_angle(self):
        leftEye = [36, 37, 38, 39, 40, 41]
        rightEye = [42, 43, 44, 45, 46, 47]
        vec = self.landmarks[rightEye].mean(axis=0) - self.landmarks[leftEye].mean(axis=0)
        return -math.degrees(math.atan(vec[1] / vec[0]))

    def norm_image(self):
        self.image = self.image / self.image.max()

    def resize(self, new_size):
        self.landmarks = self.landmarks * np.array([new_size[0] / self.width(),
                                                    new_size[1] / self.height()])
        self.image = cv2.resize(self.image, dsize=new_size, interpolation=cv2.INTER_CUBIC)
        self.norm_image()

    def pad(self, left, right, top, bottom):
        self.image = np.pad(self.image, ((top, bottom), (left, right), (0, 0)), mode='constant')
        self.landmarks = self.landmarks + np.array([left, top])

    def realign_into_bounding_box(self, bounding_box, width, height):
        self.resize((bounding_box.size, bounding_box.size))

        # pad image + landmarks 
        left_pad, right_pad = bounding_box.left, width - bounding_box.left - bounding_box.size
        top_pad, bottom_pad = bounding_box.top, height - bounding_box.top - bounding_box.size
        self.pad(left_pad, right_pad, top_pad, bottom_pad)

        # rotate
        self.rotate(bounding_box.angle)

    def rotate(self, angle):
        theta = np.radians(-angle)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c,-s), (s, c)))

        width, height = self.image.shape[1], self.image.shape[0]
        rotated = imutils.rotate(self.image, angle)

        vec = np.array([width/2, height/2], dtype=np.int32)
        self.landmarks -= vec
        self.landmarks = R.dot(self.landmarks.T).T
        self.landmarks += vec

    def get_face_bounding_box(self):
        left = int(self.landmarks[0][0])
        right = int(self.landmarks[16][0])
        top = int(self.landmarks[24][1])
        bottom = int(self.landmarks[8][1])
        padding = int(0.3 * max(right - left, bottom - top))
        
        left -= padding
        right += padding
        top -= padding
        bottom += padding

        width, height = right - left, bottom - top
        size = max(width, height)
        d = abs(width - height) / 2

        if width > height:
            top = int(top-d)
        else:
            left = int(left-d)

        top -= int(0.15 * size)
        top, left = max(0, top), max(0, left)
        return top, left, size

    def cut_to_square(self, top, left, size):
        self.image = self.image[top:top+size, left:left+size]
        self.landmarks -= np.array([left, top])

        self.resize((128, 128))

    def get_mask(self):
        width, height = self.image.shape[1], self.image.shape[0]

        start, end = self.landmarks[0][0], self.landmarks[16][0]
        landmarks = [[start, 0]] + self.landmarks[:17].tolist() + [[end, 0]]
        mask = rasterize_polygon(landmarks, height, width) * 255
        return ndimage.gaussian_filter(mask, 3) / 255

    def get_morphing_mask(self, original_size):
        eyeline = np.flip(self.landmarks[17:27], axis=0) + np.array([0, -original_size*0.17])
        points = self.landmarks[:17].tolist() + eyeline.tolist()
        mask = rasterize_polygon(points, self.height(), self.width()) * 255
        return ndimage.gaussian_filter(mask, 5) / 255

    def morph_with(self, image, original_size):
        mask = self.get_morphing_mask(original_size)
        mask = mask.reshape((*mask.shape, 1))
        inv_mask = 1 - mask
        result = self.image * mask + image / image.max() * inv_mask
        return result