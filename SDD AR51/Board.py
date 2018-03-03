import cv2
import numpy


class Board:
    def __init__(self,image,row,column):
        self._board = [[0 for r in range(row)] for c in range(column)]
        if type(image) != numpy.ndarray:
            raise TypeError('image is not numpy.ndarray')
        self._image = image
        self._orig = image.copy()
        self._warp = None


    def set_image(self,new_image):
        self._image = image

    def set_warp(self,new_warp):
        self._warp = new_warp


    def get_image(self):
        return self._image

    def get_warp(self):
        return self._warp