import cv2
import streamlit as st
import numpy as np

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
    threshold = st.sidebar.slider('Thresold values',0,255,127,1)

    is_invert = st.sidebar.checkbox('Invert image', value=False, key='Invert',)
    is_otsu = st.sidebar.checkbox('Otsu Binarization', value=False, key='Otsu',)

    if is_otsu:
        ret, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        ret, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    if is_invert:
        binary = cv2.bitwise_not(binary)

    return binary

def edge_detection(image):
    # convert the image to RGB
    image = gray_scale(image)
    low_thres = st.sidebar.slider('Lower threshold for edge detection', min_value=0, max_value=240, value=80)
    high_thresh = st.sidebar.slider('High threshold for edge detection', min_value=10, max_value=240, value=100)
    if low_thres > high_thresh:
        high_thresh = low_thres + 5
    edged = cv2.Canny(image, low_thres, high_thresh)
    return edged

def color_thresolding_rgb(image):
    r_low = st.sidebar.slider('Red Value', min_value=0, max_value=240, value=40)
    g_low = st.sidebar.slider('Green Value', min_value=0, max_value=240, value=158)
    b_low = st.sidebar.slider('Blue Value', min_value=0, max_value=240, value=16)
    thresh = st.sidebar.slider("Threshold",min_value=0, max_value=255, value=40)
    minBGR = np.array([r_low - thresh, g_low - thresh, b_low - thresh])
    maxBGR = np.array([r_low + thresh, g_low + thresh, b_low + thresh])
    maskBGR = cv2.inRange(image, minBGR, maxBGR)
    resultBGR = cv2.bitwise_and(image, image, mask=maskBGR)
    return resultBGR

def color_thresholding_hsv(image):
    # https://learnopencv.com/color-spaces-in-opencv-cpp-python/
    thresh = st.sidebar.slider("Threshold",min_value=0, max_value=255, value=23)
    h_low = st.sidebar.slider('Hue Value', min_value=0, max_value=240, value=65)
    s_low = st.sidebar.slider('Saturation Value', min_value=0, max_value=240, value=229)
    v_low = st.sidebar.slider('Value Value', min_value=0, max_value=240, value=158)
    hsv_frame = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    minHSV = np.array([h_low - thresh, s_low - thresh, v_low - thresh])
    maxHSV = np.array([h_low + thresh, s_low + thresh, v_low + thresh])

    maskHSV = cv2.inRange(hsv_frame, minHSV, maxHSV)
    resultHSV = cv2.bitwise_and(hsv_frame, hsv_frame, mask=maskHSV)
    result = cv2.cvtColor(resultHSV, cv2.COLOR_HSV2RGB)
    return result

def conv2d(image):
    kernel = np.array([[1,2,1], [0,0,0], [-1,-2,-1]])
    size = 3
    size = st.sidebar.slider('size of kernel', 2, 15, 3)
    coeffecient = st.sidebar.number_input('Enter coeffecient',value=1.0, step=0.01)

    k = np.zeros((size,size),dtype=np.int32)
    for i in range(size):
        cols = st.columns(size)
        for j in range(size):
            k[i][j] = cols[j].number_input(f'x[{i}][{j}]', step=1)

    kernel = k * coeffecient

    col1, col2 = st.columns(2)
    with col1:
        st.write(f'Kernel shape is {kernel.shape}, below is the kernel')
        st.write(kernel)
    with col2:
        img_type = st.radio("grayscale/color image", ['grayscale', 'color'])

    if img_type == 'grayscale':
        image = gray_scale(image)
    filter = cv2.filter2D(image,-1, kernel)
    return filter

def rotate_img(image):
    theta = st.sidebar.slider("Angle of rotation", -180, 180, value=0)
    #get height and wigth of an image
    height, width = image.shape[:2]
    #get center (x,y) of given image
    center = (width/2, height/2)
    # Rotate image at an angle theta w.r.t to center
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=theta, scale=1)
    # warp the image with rotate_matrix
    return cv2.warpAffine(src=image, M=rotate_matrix, dsize=(width, height))