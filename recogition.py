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

# type define
Images = Tuple[np.ndarray, np.ndarray]
Texts = Tuple[str, str]


class ImageArea:
    def __init__(self, *args):
        self.x = args[0]
        self.y = args[1]
        self.width = args[2]
        self.height = args[3]

    def get_area(self) -> Tuple[int, int, int, int]:
        x_start = self.x
        y_start = self.y
        x_end = self.x + self.width
        y_end = self.y + self.height
        return (x_start, y_start, x_end, y_end)


def get_img_uri(img_filename: str) -> str:
    filepath = os.path.split(__file__)[0]
    return os.path.join(filepath, img_filename)


def load_image(img_uri: str) -> Tuple[np.ndarray, int, int]:
    # load image
    img = cv2.imread(img_uri, 0)
    height, width = img.shape[:2]
    return (img, height, width)


def get_clipped_image(img: np.ndarray, image_area: ImageArea) -> np.ndarray:
    # initialize values
    x_start, y_start, x_end, y_end = image_area.get_area()
    # clipping
    clipped_img = img[y_start: y_end, x_start: x_end]
    return clipped_img


class Recogition(metaclass=ABCMeta):
    def __init__(self):
        tools = pyocr.get_available_tools()
        if len(tools) == 0:
            print("No OCR tool found")
            sys.exit(1)

        # initialize class member
        # select pyocr.tesseract
        self.tool = tools[0]
        self.player1_img = np.ndarray
        self.player2_img = np.ndarray
        self.width = 0
        self.height = 0

    @abstractmethod
    def get_threshold_value(self):
        pass

    @abstractmethod
    def get_img_removed_noise(self) -> Images:
        pass

    @abstractmethod
    def set_img(self, img: np.ndarray) -> None:
        pass

    def get_analyzed_txts(self) -> Texts:
        img_p1, img_p2 = self.get_img_removed_noise()
        thre_val = self.get_threshold_value()

        scaled_img_p1 = cv2.resize(img_p1, (self.width*10, self.height*10))
        scaled_img_p2 = cv2.resize(img_p2, (self.width*10, self.height*10))

        gray_p1 = cv2.cvtColor(scaled_img_p1, cv2.COLOR_RGB2GRAY)
        gray_p2 = cv2.cvtColor(scaled_img_p2, cv2.COLOR_RGB2GRAY)

        # show threshold binary
        ret_p1, thresh1_p1 = cv2.threshold(gray_p1, thre_val, 255, cv2.THRESH_BINARY)
        ret_p2, thresh1_p2 = cv2.threshold(gray_p2, thre_val, 255, cv2.THRESH_BINARY)

        plt.imshow(thresh1_p1, cmap='gray')
        plt.show()
        plt.imshow(thresh1_p2, cmap='gray')
        plt.show()

        # convert
        txt_p1 = self.tool.image_to_string(
            Image.fromarray(thresh1_p1),
            builder=pyocr.builders.TextBuilder(tesseract_layout=6)
        )

        txt_p2 = self.tool.image_to_string(
            Image.fromarray(thresh1_p2),
            builder=pyocr.builders.TextBuilder(tesseract_layout=6)
        )

        return (txt_p1, txt_p2)

    def show(self) -> None:
        img_p1, img_p2 = self.get_img_removed_noise()
        thre_val = self.get_threshold_value()

        scaled_img_p1 = cv2.resize(img_p1, (self.width*10, self.height*10))
        scaled_img_p2 = cv2.resize(img_p2, (self.width*10, self.height*10))

        gray_p1 = cv2.cvtColor(scaled_img_p1, cv2.COLOR_RGB2GRAY)
        gray_p2 = cv2.cvtColor(scaled_img_p2, cv2.COLOR_RGB2GRAY)

        # show threshold binary
        ret_p1, thresh1_p1 = cv2.threshold(gray_p1, thre_val, 255, cv2.THRESH_BINARY)
        ret_p2, thresh1_p2 = cv2.threshold(gray_p2, thre_val, 255, cv2.THRESH_BINARY)

        plt.imshow(thresh1_p1, cmap='gray')
        plt.show()
        plt.imshow(thresh1_p2, cmap='gray')
        plt.show()


class TimeRecogition(Recogition):
    def __init__(self):
        super().__init__()

    def get_threshold_value(self):
        return 20

    def get_img_removed_noise(self) -> np.ndarray:
        # reverse color
        return cv2.bitwise_not(self.img)

    def set_img(self, img: np.ndarray) -> None:
        self.img = img
        self.height, self.width = img.shape[:2]


class DamageRecogition(Recogition):
    def __init__(self):
        super().__init__()

    def get_threshold_value(self):
        return 200

    def get_img_removed_noise(self) -> np.ndarray:
        # reverse color
        return cv2.bitwise_not(self.img)

    def set_img(self, img: np.ndarray) -> None:
        self.img = img
        self.height, self.width = img.shape[:2]


class CharacterNameRecogition(Recogition):
    PLAYER1_CHARACTER_NAME_START_POSITION = (350, 670)
    PLAYER2_CHARACTER_NAME_START_POSITION = (840, 670)
    CHARACTER_NAME_AREA_SIZE = (125, 15)

    def __init__(self):
        super().__init__()

    def get_threshold_value(self):
        return 100

    def get_img_removed_noise(self) -> Images:
        # reverse color
        img_p1 = cv2.bitwise_not(self.player1_img)
        img_p2 = cv2.bitwise_not(self.player2_img)

        return (img_p1, img_p2)

    def __get_image_area_info(self, x: int, y: int, w_scale: float, h_scale: float) -> Tuple[int, int, int, int]:
        x_scaled = round(x * w_scale)
        y_scaled = round(y * h_scale)
        width = round(self.CHARACTER_NAME_AREA_SIZE[0] * w_scale)
        height = round(self.CHARACTER_NAME_AREA_SIZE[1] * h_scale)

        return (x_scaled, y_scaled, width, height)

    def set_img(self, img: np.ndarray) -> None:
        h, w = img.shape[:2]

        w_scale = w/1280
        h_scale = h/720

        x_p1 = self.PLAYER1_CHARACTER_NAME_START_POSITION[0]
        y_p1 = self.PLAYER1_CHARACTER_NAME_START_POSITION[1]
        x_p2 = self.PLAYER2_CHARACTER_NAME_START_POSITION[0]
        y_p2 = self.PLAYER2_CHARACTER_NAME_START_POSITION[1]

        image_area_info_p1 = self.__get_image_area_info(x_p1, y_p1, w_scale, h_scale)
        image_area_p1 = ImageArea(*image_area_info_p1)

        image_area_info_p2 = self.__get_image_area_info(x_p2, y_p2, w_scale, h_scale)
        image_area_p2 = ImageArea(*image_area_info_p2)

        clipped_img_p1 = get_clipped_image(img, image_area_p1)
        clipped_img_p2 = get_clipped_image(img, image_area_p2)

        self.player1_img = clipped_img_p1
        self.player2_img = clipped_img_p2
        self.width, self.height = image_area_info_p1[2:]
