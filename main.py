import cv2
import numpy as np
from PIL import Image
import streamlit as st

DEFAULT_IMG = 'fruits.jpg'

def gray_scale(img):
    '''
    :param img: numpy array of RGB image
    :return: numpy array if gray scale image
    '''
    if img.shape[-1] == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return gray_img

    if img.shape[-1] == 1:
        return img


def threshold(img):
    '''
    :param img: grey sclae image as numpy array
    :param threshold: int threshold value
    :return: Binary gray scale image
    '''
    gray = gray_scale(img)
    threshold = st.slider('Thresold values',0,255,127,1)
    ret, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return binary

# img = Image.open(DEFAULT_IMG)

def run():
    img_file = st.file_uploader("Add a file", type=['.jpg', '.jpeg', '.png'])
    if img_file != None:
        img = np.array(Image.open(img_file))
    else:
        img = np.array(Image.open(DEFAULT_IMG))

    print(img.shape)
    st.image([img, gray_scale(img), threshold(img)],width=100)