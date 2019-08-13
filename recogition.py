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

        # thresh1_p1 = cv2.adaptiveThreshold(gray_p1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 81, 25)
        # thresh1_p2 = cv2.adaptiveThreshold(gray_p2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 81, 25)

        # plt.imshow(thresh1_p1, cmap='gray')
        # plt.show()
        # plt.imshow(thresh1_p2, cmap='gray')
        # plt.show()

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
    PLAYER1_DAMAGE_START_POSITION = (365, 610)
    PLAYER2_DAMAGE_START_POSITION = (860, 610)
    CHARACTER_DAMAGE_AREA_SIZE = (35, 50)

    def __init__(self):
        super().__init__()

    def get_threshold_value(self):
        return 210

    def get_img_removed_noise(self) -> Images:
        # reverse color
        img_p1 = cv2.bitwise_not(self.player1_img)
        img_p2 = cv2.bitwise_not(self.player2_img)

        return (img_p1, img_p2)

    def __get_image_area_info(self, x: int, y: int, w_scale: float, h_scale: float) -> Tuple[int, int, int, int]:
        x_scaled = round(x * w_scale)
        y_scaled = round(y * h_scale)
        width = round(self.CHARACTER_DAMAGE_AREA_SIZE[0] * w_scale)
        height = round(self.CHARACTER_DAMAGE_AREA_SIZE[1] * h_scale)

        return (x_scaled, y_scaled, width, height)

    def set_img(self, img: np.ndarray) -> None:
        h, w = img.shape[:2]

        w_scale = w/1280
        h_scale = h/720

        x_p1 = self.PLAYER1_DAMAGE_START_POSITION[0]
        y_p1 = self.PLAYER1_DAMAGE_START_POSITION[1]
        x_p2 = self.PLAYER2_DAMAGE_START_POSITION[0]
        y_p2 = self.PLAYER2_DAMAGE_START_POSITION[1]

        image_area_info_p1 = self.__get_image_area_info(x_p1, y_p1, w_scale, h_scale)
        image_area_p1 = ImageArea(*image_area_info_p1)

        image_area_info_p2 = self.__get_image_area_info(x_p2, y_p2, w_scale, h_scale)
        image_area_p2 = ImageArea(*image_area_info_p2)

        clipped_img_p1 = get_clipped_image(img, image_area_p1)
        clipped_img_p2 = get_clipped_image(img, image_area_p2)

        x_size_p1 = clipped_img_p1.shape[0]
        x_size_p2 = clipped_img_p2.shape[0]
        y_size_p1 = clipped_img_p1.shape[1]
        y_size_p2 = clipped_img_p2.shape[1]

        offset = 2

        p_original_p1 = np.float32([[offset, 0], [x_size_p1-offset, 0], [0, y_size_p1], [x_size_p1-(offset*2), y_size_p1]])
        p_trans_p1 = np.float32([[0, 0], [x_size_p1, 0], [0, y_size_p1], [x_size_p1, y_size_p1]])

        p_original_p2 = np.float32([[offset, 0], [x_size_p2-offset, 0], [0, y_size_p2], [x_size_p2-(offset*2), y_size_p2]])
        p_trans_p2 = np.float32([[0, 0], [x_size_p2, 0], [0, y_size_p2], [x_size_p2, y_size_p2]])

        M_p1 = cv2.getPerspectiveTransform(p_original_p1, p_trans_p1)
        M_p2 = cv2.getPerspectiveTransform(p_original_p2, p_trans_p2)

        i_trans_p1 = cv2.warpPerspective(clipped_img_p1, M_p1, (y_size_p1, x_size_p1))
        i_trans_p2 = cv2.warpPerspective(clipped_img_p2, M_p2, (y_size_p2, x_size_p2))

        self.player1_img = i_trans_p1
        self.player2_img = i_trans_p2
        self.width, self.height = image_area_info_p1[2:]

    def write(self, filepath):
        img_p1, img_p2 = self.get_img_removed_noise()
        thre_val = self.get_threshold_value()

        scaled_img_p1 = cv2.resize(img_p1, (self.width*10, self.height*10))
        scaled_img_p2 = cv2.resize(img_p2, (self.width*10, self.height*10))

        gray_p1 = cv2.cvtColor(scaled_img_p1, cv2.COLOR_RGB2GRAY)
        gray_p2 = cv2.cvtColor(scaled_img_p2, cv2.COLOR_RGB2GRAY)

        # show threshold binary
        ret_p1, thresh1_p1 = cv2.threshold(gray_p1, thre_val, 255, cv2.THRESH_BINARY)
        ret_p2, thresh1_p2 = cv2.threshold(gray_p2, thre_val, 255, cv2.THRESH_BINARY)

        filepath_p1 = filepath + '_p1.png'
        filepath_p2 = filepath + '_p2.png'
        cv2.imwrite(filepath_p1, thresh1_p1, [cv2.IMWRITE_JPEG_QUALITY, 100])
        cv2.imwrite(filepath_p2, thresh1_p2, [cv2.IMWRITE_JPEG_QUALITY, 100])


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


class TestRecogition:
    WIDTH_OFFSET = 8
    PLAYER1_DAMAGE_START_POSITION = (330, 610)
    PLAYER2_DAMAGE_START_POSITION = (820, 610)
    CHARACTER_DAMAGE_AREA_SIZE = (36+WIDTH_OFFSET, 50)

    def __init__(self):
        tools = pyocr.get_available_tools()
        if len(tools) == 0:
            print("No OCR tool found")
            sys.exit(1)

        # initialize class member
        # select pyocr.tesseract
        self.tool = tools[0]
        self.player1_img = []
        self.player2_img = []

        fileName = 'model3.pickle'
        filepath = os.path.split(__file__)[0]
        uri_pickle = os.path.join(filepath, fileName)

        with open(uri_pickle, mode='rb') as fp:
            self.lr = pickle.load(fp)


    # def get_threshold_value(self):
    #     return 0

    # def get_img_removed_noise(self) -> Images:
    #     # reverse color
    #     # img_p1 = cv2.bitwise_not(self.player1_img)
    #     # img_p2 = cv2.bitwise_not(self.player2_img)
    #
    #     img_p1 = self.player1_img
    #     img_p2 = self.player2_img
    #
    #     return (img_p1, img_p2)

    def __get_image_area_info(self, x: int, y: int, w_scale: float, h_scale: float) -> Tuple[int, int, int, int]:
        x_scaled = round(x * w_scale)
        y_scaled = round(y * h_scale)
        width = round(self.CHARACTER_DAMAGE_AREA_SIZE[0] * w_scale)
        height = round(self.CHARACTER_DAMAGE_AREA_SIZE[1] * h_scale)

        # print('area_info {},{},{},{}'.format(x_scaled, y_scaled, width, height))
        return (x_scaled, y_scaled, width, height)

    def __get_corrected_image(self, img, x, y, w_scale, h_scale):
        image_area_info = self.__get_image_area_info(x, y, w_scale, h_scale)
        image_area = ImageArea(*image_area_info)

        clipped_img = get_clipped_image(img, image_area)

        x_size = clipped_img.shape[0]
        y_size = clipped_img.shape[1]

        offset = 4
        p_original = np.float32([[offset, 0], [x_size-offset, 0], [0, y_size], [x_size-(offset*2), y_size]])
        p_trans = np.float32([[0, 0], [x_size, 0], [0, y_size], [x_size, y_size]])

        M = cv2.getPerspectiveTransform(p_original, p_trans)

        i_trans = cv2.warpPerspective(clipped_img, M, (y_size, x_size))

        return i_trans

    def set_img(self, img: np.ndarray) -> None:
        h, w = img.shape[:2]

        self.player1_img = []
        self.player2_img = []

        w_scale = w/1280
        h_scale = h/720

        x_p1 = self.PLAYER1_DAMAGE_START_POSITION[0]
        y_p1 = self.PLAYER1_DAMAGE_START_POSITION[1]
        x_p2 = self.PLAYER2_DAMAGE_START_POSITION[0]
        y_p2 = self.PLAYER2_DAMAGE_START_POSITION[1]

        for i in range(3):
            # need offset...-4
            i_trans_p1 = self.__get_corrected_image(img, x_p1+i*(self.CHARACTER_DAMAGE_AREA_SIZE[0]-self.WIDTH_OFFSET), y_p1, w_scale, h_scale)
            i_trans_p2 = self.__get_corrected_image(img, x_p2+i*(self.CHARACTER_DAMAGE_AREA_SIZE[0]-self.WIDTH_OFFSET), y_p2, w_scale, h_scale)

            # print(img.shape)
            # print(i_trans_p1.shape)

        # image_area_info_p1 = self.__get_image_area_info(x_p1, y_p1, w_scale, h_scale)
        # image_area_p1 = ImageArea(*image_area_info_p1)
        #
        # image_area_info_p2 = self.__get_image_area_info(x_p2, y_p2, w_scale, h_scale)
        # image_area_p2 = ImageArea(*image_area_info_p2)
        #
        # clipped_img_p1 = get_clipped_image(img, image_area_p1)
        # clipped_img_p2 = get_clipped_image(img, image_area_p2)
        #
        # x_size_p1 = clipped_img_p1.shape[0]
        # x_size_p2 = clipped_img_p2.shape[0]
        # y_size_p1 = clipped_img_p1.shape[1]
        # y_size_p2 = clipped_img_p2.shape[1]
        #
        # offset = 2
        #
        # p_original_p1 = np.float32([[offset, 0], [x_size_p1-offset, 0], [0, y_size_p1], [x_size_p1-(offset*2), y_size_p1]])
        # p_trans_p1 = np.float32([[0, 0], [x_size_p1, 0], [0, y_size_p1], [x_size_p1, y_size_p1]])
        #
        # p_original_p2 = np.float32([[offset, 0], [x_size_p2-offset, 0], [0, y_size_p2], [x_size_p2-(offset*2), y_size_p2]])
        # p_trans_p2 = np.float32([[0, 0], [x_size_p2, 0], [0, y_size_p2], [x_size_p2, y_size_p2]])
        #
        # M_p1 = cv2.getPerspectiveTransform(p_original_p1, p_trans_p1)
        # M_p2 = cv2.getPerspectiveTransform(p_original_p2, p_trans_p2)
        #
        # i_trans_p1 = cv2.warpPerspective(clipped_img_p1, M_p1, (y_size_p1, x_size_p1))
        # i_trans_p2 = cv2.warpPerspective(clipped_img_p2, M_p2, (y_size_p2, x_size_p2))

            self.player1_img.append(i_trans_p1)
            self.player2_img.append(i_trans_p2)



    def __get_data_for_model(self, img):
        img_cp = img.copy()

        # remove blue
        grn = img_cp[:, :, 1]
        red = img_cp[:, :, 2]
        red_grn = cv2.addWeighted(red, 0.5, grn, 0.5, 0)

        # pick up number(remove background)
        th_red_grn = cv2.adaptiveThreshold(red_grn, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 10)

        # another pick up number(by gray scale)
        grayed = cv2.cvtColor(img_cp, cv2.COLOR_RGB2GRAY)
        ret, th_gr = cv2.threshold(grayed, 10, 255, cv2.THRESH_BINARY)

        # merge
        merged = np.minimum(th_red_grn, th_gr)

        # print('grayed')
        # plt.imshow(grayed, cmap='gray')
        # plt.show()
        #
        # print('th_gr')
        # print(th_gr)
        # plt.imshow(th_gr, cmap='gray')
        # plt.show()
        #
        # print('merged')
        # print(merged)
        # plt.imshow(merged, cmap='gray')
        # plt.show()
        # resize
        # height, width = img.shape[:2]
        # size = (int(width/10), int(height/10))
        #
        # resized = cv2.resize(merged, size)

        # return resized
        return merged

    def get_analyzed_txts(self) -> Texts:
        imgs_p1, imgs_p2 = (self.player1_img, self.player2_img)

        # scaled_img_p1 = cv2.resize(img_p1, (self.width*10, self.height*10))
        # scaled_img_p2 = cv2.resize(img_p2, (self.width*10, self.height*10))

        # p1 = scaled_img_p1.copy()
        # p2 = scaled_img_p2.copy()

        # grn_p1 = p1[:, :, 1]
        # grn_p2 = p2[:, :, 1]
        #
        # red_p1 = p1[:, :, 2]
        # red_p2 = p2[:, :, 2]

        # redGreen_p1 = cv2.addWeighted(red_p1, 0.5, grn_p1, 0.5, 0)
        # redGreen_p2 = cv2.addWeighted(red_p2, 0.5, grn_p2, 0.5, 0)
        #
        # th_red_p1 = cv2.adaptiveThreshold(redGreen_p1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 10)
        # th_red_p2 = cv2.adaptiveThreshold(redGreen_p2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 10)

        # gray_p1 = cv2.cvtColor(scaled_img_p1, cv2.COLOR_RGB2GRAY)
        # gray_p2 = cv2.cvtColor(scaled_img_p2, cv2.COLOR_RGB2GRAY)

        # show threshold binary
        # ret_p1, thresh1_p1 = cv2.threshold(gray_p1, 15, 255, cv2.THRESH_BINARY)
        # ret_p2, thresh1_p2 = cv2.threshold(gray_p2, 15, 255, cv2.THRESH_BINARY)

        # ret_p1, th_p1 = cv2.threshold(gray_p1, 10, 255, cv2.THRESH_BINARY)
        # ret_p2, th_p2 = cv2.threshold(gray_p2, 10, 255, cv2.THRESH_BINARY)

        # mix_p1 = cv2.addWeighted(th_red_p1, 0.7, th_p1, 1.0, 0)
        # mix_p2 = cv2.addWeighted(th_red_p2, 0.7, th_p2, 1.0, 0)

        # mix_p1 = np.minimum(th_red_p1, th_p1)
        # mix_p2 = np.minimum(th_red_p2, th_p2)
        # orgHeight, orgWidth = mix_p1.shape[:2]
        # size = (int(orgWidth/4), int(orgHeight/4))
        # mix_p1 = cv2.resize(mix_p1, size)
        # mix_p2 = cv2.resize(mix_p2, size)

        # convert
        # print(mix_p1.reshape(1,-1).shape)

        txts_p1 = []
        txts_p2 = []

        for img_p1, img_p2 in zip(imgs_p1, imgs_p2):
            h_p1, w_p1 = img_p1.shape[:2]
            h_p2, w_p2 = img_p2.shape[:2]
            # scaled_img_p1 = cv2.resize(img_p1, (w_p1*10, h_p1*10))
            # scaled_img_p2 = cv2.resize(img_p2, (w_p2*10, h_p2*10))

            # mix_p1 = self.__get_data_for_model(scaled_img_p1)
            # mix_p2 = self.__get_data_for_model(scaled_img_p2)

            mix_p1 = self.__get_data_for_model(img_p1)
            mix_p2 = self.__get_data_for_model(img_p2)

            txt_p1 = self.lr.predict(mix_p1.reshape(1,-1))[0]   # [0] -> 1d array to number
            txt_p2 = self.lr.predict(mix_p2.reshape(1,-1))[0]   # [0] -> 1d array to number

            if(txt_p1 == 10):
                txt_p1 = 0
            if(txt_p2 == 10):
                txt_p2 = 0

            txts_p1.append(txt_p1)
            txts_p2.append(txt_p2)

            # print(txt_p1)
            # plt.imshow(mix_p1, cmap='gray')
            # plt.show()
            #
            # # print(txt_p2)
            # plt.imshow(mix_p2, cmap='gray')
            # plt.show()

        damage_p1 = txts_p1[0]*100 + txts_p1[1]*10 + txts_p1[2]
        damage_p2 = txts_p2[0]*100 + txts_p2[1]*10 + txts_p2[2]


        # print((damage_p1, damage_p2))
        return (damage_p1, damage_p2)

    def show(self) -> None:
        img_p1, img_p2 = self.get_img_removed_noise()
        thre_val = self.get_threshold_value()

        scaled_img_p1 = cv2.resize(img_p1, (self.width*10, self.height*10))
        scaled_img_p2 = cv2.resize(img_p2, (self.width*10, self.height*10))

        p1 = scaled_img_p1.copy()
        p2 = scaled_img_p2.copy()

        grn_p1 = p1[:, :, 1]
        grn_p2 = p2[:, :, 1]

        red_p1 = p1[:, :, 2]
        red_p2 = p2[:, :, 2]

        redGreen_p1 = cv2.addWeighted(red_p1, 0.5, grn_p1, 0.5, 0)
        redGreen_p2 = cv2.addWeighted(red_p2, 0.5, grn_p2, 0.5, 0)

        th_red_p1 = cv2.adaptiveThreshold(redGreen_p1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 10)
        th_red_p2 = cv2.adaptiveThreshold(redGreen_p2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 10)

        gray_p1 = cv2.cvtColor(scaled_img_p1, cv2.COLOR_RGB2GRAY)
        gray_p2 = cv2.cvtColor(scaled_img_p2, cv2.COLOR_RGB2GRAY)

        # show threshold binary
        # ret_p1, thresh1_p1 = cv2.threshold(gray_p1, 15, 255, cv2.THRESH_BINARY)
        # ret_p2, thresh1_p2 = cv2.threshold(gray_p2, 15, 255, cv2.THRESH_BINARY)

        ret_p1, th_p1 = cv2.threshold(gray_p1, 10, 255, cv2.THRESH_BINARY)
        ret_p2, th_p2 = cv2.threshold(gray_p2, 10, 255, cv2.THRESH_BINARY)

        # mix_p1 = cv2.addWeighted(th_red_p1, 0.7, th_p1, 1.0, 0)
        # mix_p2 = cv2.addWeighted(th_red_p2, 0.7, th_p2, 1.0, 0)

        mix_p1 = np.minimum(th_red_p1, th_p1)
        mix_p2 = np.minimum(th_red_p2, th_p2)
        orgHeight, orgWidth = mix_p1.shape[:2]
        size = (int(orgWidth/4), int(orgHeight/4))
        mix_p1 = cv2.resize(mix_p1, size)
        mix_p2 = cv2.resize(mix_p2, size)

        plt.imshow(self.player1_img, cmap='gray')
        plt.show()
        plt.imshow(self.player2_img, cmap='gray')
        plt.show()

        plt.imshow(mix_p1, cmap='gray')
        plt.show()
        plt.imshow(mix_p2, cmap='gray')
        plt.show()



    def write(self, filepath):
        imgs_p1, imgs_p2 = (self.player1_img, self.player2_img)

        i = 0
        for img_p1, img_p2 in zip(imgs_p1, imgs_p2):
            i = i + 1
            h_p1, w_p1 = img_p1.shape[:2]
            h_p2, w_p2 = img_p2.shape[:2]

            mix_p1 = self.__get_data_for_model(img_p1)
            mix_p2 = self.__get_data_for_model(img_p2)

            filepath_p1 = filepath + str(i) + '_p1.png'
            filepath_p2 = filepath + str(i) + '_p2.png'
            cv2.imwrite(filepath_p1, mix_p1, [cv2.IMWRITE_JPEG_QUALITY, 100])
            cv2.imwrite(filepath_p2, mix_p2, [cv2.IMWRITE_JPEG_QUALITY, 100])



        # img_p1, img_p2 = self.get_img_removed_noise()
        # thre_val = self.get_threshold_value()
        #
        # scaled_img_p1 = cv2.resize(img_p1, (self.width*10, self.height*10))
        # scaled_img_p2 = cv2.resize(img_p2, (self.width*10, self.height*10))
        #
        # p1 = scaled_img_p1.copy()
        # p2 = scaled_img_p2.copy()
        #
        # grn_p1 = p1[:, :, 1]
        # grn_p2 = p2[:, :, 1]
        #
        # red_p1 = p1[:, :, 2]
        # red_p2 = p2[:, :, 2]
        #
        # redGreen_p1 = cv2.addWeighted(red_p1, 0.5, grn_p1, 0.5, 0)
        # redGreen_p2 = cv2.addWeighted(red_p2, 0.5, grn_p2, 0.5, 0)
        #
        # th_red_p1 = cv2.adaptiveThreshold(redGreen_p1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 10)
        # th_red_p2 = cv2.adaptiveThreshold(redGreen_p2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 10)
        #
        # gray_p1 = cv2.cvtColor(scaled_img_p1, cv2.COLOR_RGB2GRAY)
        # gray_p2 = cv2.cvtColor(scaled_img_p2, cv2.COLOR_RGB2GRAY)
        #
        # # show threshold binary
        # # ret_p1, thresh1_p1 = cv2.threshold(gray_p1, 15, 255, cv2.THRESH_BINARY)
        # # ret_p2, thresh1_p2 = cv2.threshold(gray_p2, 15, 255, cv2.THRESH_BINARY)
        #
        # ret_p1, th_p1 = cv2.threshold(gray_p1, 10, 255, cv2.THRESH_BINARY)
        # ret_p2, th_p2 = cv2.threshold(gray_p2, 10, 255, cv2.THRESH_BINARY)
        #
        # # mix_p1 = cv2.addWeighted(th_red_p1, 0.7, th_p1, 1.0, 0)
        # # mix_p2 = cv2.addWeighted(th_red_p2, 0.7, th_p2, 1.0, 0)
        #
        # mix_p1 = np.minimum(th_red_p1, th_p1)
        # mix_p2 = np.minimum(th_red_p2, th_p2)
        # orgHeight, orgWidth = mix_p1.shape[:2]
        # size = (int(orgWidth/4), int(orgHeight/4))
        # mix_p1 = cv2.resize(mix_p1, size)
        # mix_p2 = cv2.resize(mix_p2, size)
        #
        # filepath_p1 = filepath + '_p1.png'
        # filepath_p2 = filepath + '_p2.png'
        # cv2.imwrite(filepath_p1, mix_p1, [cv2.IMWRITE_JPEG_QUALITY, 100])
        # cv2.imwrite(filepath_p2, mix_p2, [cv2.IMWRITE_JPEG_QUALITY, 100])
