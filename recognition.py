# -*- coding: utf-8 -*-
from PIL import Image
import os
import pyocr
import sys
import cv2
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod
from typing import Tuple, List
import numpy as np
import pickle


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


class Recogition(metaclass=ABCMeta):
    def __init__(self):
        tools = pyocr.get_available_tools()
        if len(tools) == 0:
            print("No OCR tool found")
            sys.exit(1)

        # initialize class member
        # select pyocr.tesseract
        self.tool = tools[0]
        self.p1_img_list = []
        self.p2_img_list = []
        self.player1_img = np.ndarray
        self.player2_img = np.ndarray
        self.width = 0
        self.height = 0

    @abstractmethod
    def set_img(self, img: np.ndarray) -> None:
        pass

    @abstractmethod
    def recognize(self, image: np.ndarray) -> str:
        pass

    @abstractmethod
    def make_result_text(self, str_list: List[str]) -> str:
        pass

    @abstractmethod
    def get_image_for_model(self, image: np.ndarray) -> Images:
        pass

    def get_clipped_image(self, img: np.ndarray, img_area: ImageArea) -> np.ndarray:
        # initialize values
        x_start, y_start, x_end, y_end = img_area.get_area()
        # clipping
        clipped_img = img[y_start: y_end, x_start: x_end]
        return clipped_img

    def __template_recognize(self, image_list: List[np.ndarray]) -> str:
        string_list = []

        for image in image_list:
            result_image = self.get_image_for_model(image)
            string = self.recognize(result_image)
            string_list.append(string)

        result_text = self.make_result_text(string_list)
        return result_text

    def get_recognized_texts(self) -> Texts:
        result_text_p1 = self.__template_recognize(self.p1_img_list)
        result_text_p2 = self.__template_recognize(self.p2_img_list)

        return (result_text_p1, result_text_p2)

    def __template_show(self, image_list: List[np.ndarray]) -> None:
        for image in image_list:
            result_image = self.get_image_for_model(image)
            plt.imshow(result_image, cmap='gray')
            plt.show()

    def show(self) -> None:
        self.__template_show(self.p1_img_list)
        self.__template_show(self.p2_img_list)

    def __template_write(self, image_list: List[np.ndarray], filepath: str) -> None:
        path, ext = filepath.split('.')
        for i, image in enumerate(image_list):
            tmp_path = path + '_{}'.format(str(10**i).zfill(3))
            result_image = self.get_image_for_model(image)
            cv2.imwrite('.'.join([tmp_path, ext]), result_image)

    def write(self, filepath: Tuple[str, str]) -> None:
        # filepath's file name must not contain '.'
        self.__template_write(self.p1_img_list, filepath[0])
        self.__template_write(self.p2_img_list, filepath[1])


class CharacterNameRecogition(Recogition):
    PLAYER1_CHARACTER_NAME_START_POSITION = (350, 670)
    PLAYER2_CHARACTER_NAME_START_POSITION = (840, 670)
    CHARACTER_NAME_AREA_SIZE = (125, 15)

    def __init__(self):
        super().__init__()
        tools = pyocr.get_available_tools()
        if len(tools) == 0:
            print("No OCR tool found")
            sys.exit(1)

        # select pyocr.tesseract
        self.tool = tools[0]

    def get_image_for_model(self, image: np.ndarray) -> np.ndarray:
        grayed = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        thre_image = cv2.threshold(grayed, 180, 255, cv2.THRESH_BINARY)[1]
        result_image = cv2.bitwise_not(thre_image)
        return result_image

    def make_result_text(self, str_list: List[str]) -> str:
        # str_list is 1d array
        return str_list[0]

    def recognize(self, image: np.ndarray) -> str:
        character_name = self.tool.image_to_string(
            Image.fromarray(image),
            builder=pyocr.builders.TextBuilder(tesseract_layout=6)
        )

        return character_name

    def __get_image_area_info(self, x: int, y: int, w_scale: float, h_scale: float) -> Tuple[int, int, int, int]:
        x_scaled = round(x * w_scale)
        y_scaled = round(y * h_scale)
        width = round(self.CHARACTER_NAME_AREA_SIZE[0] * w_scale)
        height = round(self.CHARACTER_NAME_AREA_SIZE[1] * h_scale)

        return (x_scaled, y_scaled, width, height)

    def set_img(self, img: np.ndarray) -> None:
        self.p1_img_list = []
        self.p2_img_list = []

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

        clipped_img_p1 = self.get_clipped_image(img, image_area_p1)
        clipped_img_p2 = self.get_clipped_image(img, image_area_p2)

        self.p1_img_list.append(clipped_img_p1)
        self.p2_img_list.append(clipped_img_p2)


class DamageRecogition(Recogition):
    WIDTH_OFFSET = 8
    ITALIC_OFFSET = 4
    PLAYER1_DAMAGE_START_POSITION = (330, 610)
    PLAYER2_DAMAGE_START_POSITION = (820, 610)
    DAMAGE_AREA_SIZE = (36+WIDTH_OFFSET, 50)
    MODEL_PICKLE = 'model3.pickle'

    def __init__(self):
        root_dir = os.path.split(__file__)[0]
        pickle_uri = os.path.join(root_dir, self.MODEL_PICKLE)

        # initialize instance member
        self.p1_img_list = []
        self.p2_img_list = []
        with open(pickle_uri, mode='rb') as fp:
            self.model = pickle.load(fp)

    def __get_image_area_info(self, x: int, y: int, w_scale: float, h_scale: float) -> Tuple[int, int, int, int]:
        x_scaled = round(x * w_scale)
        y_scaled = round(y * h_scale)
        width = round(self.DAMAGE_AREA_SIZE[0] * w_scale)
        height = round(self.DAMAGE_AREA_SIZE[1] * h_scale)

        return (x_scaled, y_scaled, width, height)

    def __get_corrected_image(self, img, x, y, w_scale, h_scale):
        # create ImageArea instance
        image_area_info = self.__get_image_area_info(x, y, w_scale, h_scale)
        image_area = ImageArea(*image_area_info)

        # get damage area by clipping
        clipped_img = self.get_clipped_image(img, image_area)

        x_size = clipped_img.shape[0]
        y_size = clipped_img.shape[1]

        offset = self.ITALIC_OFFSET
        p_original = np.float32([[offset, 0], [x_size-offset, 0], [0, y_size], [x_size-(offset*2), y_size]])
        p_trans = np.float32([[0, 0], [x_size, 0], [0, y_size], [x_size, y_size]])

        M = cv2.getPerspectiveTransform(p_original, p_trans)

        i_trans = cv2.warpPerspective(clipped_img, M, (y_size, x_size))

        return i_trans

    def set_img(self, img: np.ndarray) -> None:
        h, w = img.shape[:2]

        self.p1_img_list = []
        self.p2_img_list = []

        # if receive resized video, calculate scale
        w_scale = w/1280
        h_scale = h/720

        x_p1, y_p1 = self.PLAYER1_DAMAGE_START_POSITION
        x_p2, y_p2 = self.PLAYER2_DAMAGE_START_POSITION

        # number is 3 digits so divide into 3
        for i in range(3):
            # Because the numbers are italicized, apply corrections
            # shift x position(width of DAMAGE_AREA_SIZE has buffer size so need offset)
            shift_size = i * (self.DAMAGE_AREA_SIZE[0] - self.WIDTH_OFFSET)
            img_p1 = self.__get_corrected_image(img, x_p1+shift_size, y_p1, w_scale, h_scale)
            img_p2 = self.__get_corrected_image(img, x_p2+shift_size, y_p2, w_scale, h_scale)

            self.p1_img_list.append(img_p1)
            self.p2_img_list.append(img_p2)

    def get_image_for_model(self, image: np.ndarray) -> np.ndarray:
        image_cp = image.copy()

        # pick up number(by remove blue)
        grn = image_cp[:, :, 1]
        red = image_cp[:, :, 2]
        red_grn = cv2.addWeighted(red, 0.5, grn, 0.5, 0)
        th_red_grn = cv2.adaptiveThreshold(red_grn, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 10)

        # another pick up number(by gray scale threshold)
        # grayed = cv2.cvtColor(image_cp, cv2.COLOR_RGB2GRAY)
        grayed = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        ret, th_gr = cv2.threshold(grayed, 10, 255, cv2.THRESH_BINARY)

        # merge
        merged = np.minimum(th_red_grn, th_gr)

        return merged

    def recognize(self, image: np.ndarray) -> str:
        # convert to 1d array
        image_1d = image.reshape(1, -1)
        # predict number
        number_str = str(self.model.predict(image_1d)[0])  # [0] -> 1d array to number

        return number_str

    def make_result_text(self, str_list: List[str]) -> str:
        damage = -1
        place_1 = int(str_list[2])    # ones place
        place_10 = int(str_list[1])   # tens place
        place_100 = int(str_list[0])  # hundreds place
        if(place_1 != 10):
            # (value % 10) -> 10 to 0
            damage = (place_100 % 10)*100 + (place_10 % 10)*10 + place_1

        return str(damage)

    # def show(self) -> None:
    #     img_p1, img_p2 = self.get_img_removed_noise()
    #     #thre_val = self.get_threshold_value()
    #
    #     scaled_img_p1 = cv2.resize(img_p1, (self.width*10, self.height*10))
    #     scaled_img_p2 = cv2.resize(img_p2, (self.width*10, self.height*10))
    #
    #     p1 = scaled_img_p1.copy()
    #     p2 = scaled_img_p2.copy()
    #
    #     grn_p1 = p1[:, :, 1]
    #     grn_p2 = p2[:, :, 1]
    #
    #     red_p1 = p1[:, :, 2]
    #     red_p2 = p2[:, :, 2]
    #
    #     redGreen_p1 = cv2.addWeighted(red_p1, 0.5, grn_p1, 0.5, 0)
    #     redGreen_p2 = cv2.addWeighted(red_p2, 0.5, grn_p2, 0.5, 0)
    #
    #     th_red_p1 = cv2.adaptiveThreshold(redGreen_p1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 10)
    #     th_red_p2 = cv2.adaptiveThreshold(redGreen_p2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 10)
    #
    #     gray_p1 = cv2.cvtColor(scaled_img_p1, cv2.COLOR_RGB2GRAY)
    #     gray_p2 = cv2.cvtColor(scaled_img_p2, cv2.COLOR_RGB2GRAY)
    #
    #     # show threshold binary
    #     # ret_p1, thresh1_p1 = cv2.threshold(gray_p1, 15, 255, cv2.THRESH_BINARY)
    #     # ret_p2, thresh1_p2 = cv2.threshold(gray_p2, 15, 255, cv2.THRESH_BINARY)
    #
    #     ret_p1, th_p1 = cv2.threshold(gray_p1, 10, 255, cv2.THRESH_BINARY)
    #     ret_p2, th_p2 = cv2.threshold(gray_p2, 10, 255, cv2.THRESH_BINARY)
    #
    #     # mix_p1 = cv2.addWeighted(th_red_p1, 0.7, th_p1, 1.0, 0)
    #     # mix_p2 = cv2.addWeighted(th_red_p2, 0.7, th_p2, 1.0, 0)
    #
    #     mix_p1 = np.minimum(th_red_p1, th_p1)
    #     mix_p2 = np.minimum(th_red_p2, th_p2)
    #     orgHeight, orgWidth = mix_p1.shape[:2]
    #     size = (int(orgWidth/4), int(orgHeight/4))
    #     mix_p1 = cv2.resize(mix_p1, size)
    #     mix_p2 = cv2.resize(mix_p2, size)
    #
    #     plt.imshow(self.p1_img_list, cmap='gray')
    #     plt.show()
    #     plt.imshow(self.p2_img_list, cmap='gray')
    #     plt.show()
    #
    #     plt.imshow(mix_p1, cmap='gray')
    #     plt.show()
    #     plt.imshow(mix_p2, cmap='gray')
    #     plt.show()
    #
    # def write(self, filepath):
    #     imgs_p1, imgs_p2 = (self.p1_img_list, self.p2_img_list)
    #
    #     i = 0
    #     for img_p1, img_p2 in zip(imgs_p1, imgs_p2):
    #         i = i + 1
    #         h_p1, w_p1 = img_p1.shape[:2]
    #         h_p2, w_p2 = img_p2.shape[:2]
    #
    #         mix_p1 = self.__get_data_for_model(img_p1)
    #         mix_p2 = self.__get_data_for_model(img_p2)
    #
    #         filepath_p1 = filepath + str(i) + '_p1.png'
    #         filepath_p2 = filepath + str(i) + '_p2.png'
    #         cv2.imwrite(filepath_p1, mix_p1, [cv2.IMWRITE_JPEG_QUALITY, 100])
    #         cv2.imwrite(filepath_p2, mix_p2, [cv2.IMWRITE_JPEG_QUALITY, 100])
