# -*- coding: utf-8 -*-
from PIL import Image
import os
import pyocr
import sys
import cv2
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod
from typing import Tuple
import numpy as np


def get_img_uri(img_filename: str) -> str:
    filepath = os.path.split(__file__)[0]
    return os.path.join(filepath, img_filename)


def load_image(img_uri: str) -> Tuple[np.ndarray, int, int]:
    # load image
    img = cv2.imread(img_uri, 0)
    height, width = img.shape[:2]
    return (img, height, width)


class Recogition(metaclass=ABCMeta):
    def __init__(self):
        tools = pyocr.get_available_tools()
        if len(tools) == 0:
            print("No OCR tool found")
            sys.exit(1)

        # initialize class member
        # select pyocr.tesseract
        self.tool = tools[0]

    @abstractmethod
    def get_threshold_value(self):
        pass

    @abstractmethod
    def get_img_removed_noise(self) -> np.ndarray:
        pass

    def set_img(self, img: np.ndarray) -> None:
        self.img = img
        self.height, self.width = img.shape[:2]

    def get_analyzed_txt(self) -> str:
        img = self.get_img_removed_noise()
        thre_val = self.get_threshold_value()

        rs_img = cv2.resize(img, (self.width*10, self.height*10))
        # show threshold binary
        ret, thresh1 = cv2.threshold(rs_img, thre_val, 255, cv2.THRESH_BINARY)
        plt.imshow(thresh1, cmap='gray')
        plt.show()

        # convert
        txt = self.tool.image_to_string(
            Image.fromarray(thresh1),
            builder=pyocr.builders.TextBuilder(tesseract_layout=6)
        )

        return txt

    def show(self) -> None:
        img = self.get_img_removed_noise()
        thre_val = self.get_threshold_value()

        rs_img = cv2.resize(img, (self.width*10, self.height*10))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # show threshold binary
        ret, thresh1 = cv2.threshold(gray, thre_val, 255, cv2.THRESH_BINARY)
        plt.imshow(thresh1, cmap='gray')
        plt.show()


class TimeRecogition(Recogition):
    def __init__(self):
        super().__init__()

    def get_threshold_value(self):
        return 20

    def get_img_removed_noise(self) -> np.ndarray:
        # reverse color
        return cv2.bitwise_not(self.img)


class DamageRecogition(Recogition):
    def __init__(self):
        super().__init__()

    def get_threshold_value(self):
        return 200

    def get_img_removed_noise(self) -> np.ndarray:
        # reverse color
        return cv2.bitwise_not(self.img)


class CharacterNameRecogition(Recogition):
    def __init__(self):
        super().__init__()

    def get_threshold_value(self):
        return 100

    def get_img_removed_noise(self) -> np.ndarray:
        # reverse color
        return cv2.bitwise_not(self.img)
