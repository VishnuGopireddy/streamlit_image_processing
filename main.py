import numpy as np
from PIL import Image
import streamlit as st
import img_processing_algos as ip_alogs


DEFAULT_IMG = 'cube.png'
# img = Image.open(DEFAULT_IMG)
img_file = st.file_uploader("Add a file", type=['.jpg', '.jpeg', '.png'])
if img_file != None:
    img = Image.open(img_file)
    img = img.convert('RGB')
    img = np.array(img)
else:
    img = Image.open(DEFAULT_IMG)
    img = img.convert('RGB')
    img = np.array(img)

target = img.copy()

algorithms = ['grayscale', 'grayscale_binary', 'edge_detection', 'RGB_thresolding', 'HSV_thresholding', 'Conv', 'Rotation', 'Morphologial']

algo_type = st.sidebar.radio("Choose an algorithm", options=algorithms)
if algo_type == algorithms[0]:
    target = ip_alogs.gray_scale(img)
if algo_type == algorithms[1]:
    target = ip_alogs.threshold(img)
if algo_type == algorithms[2]:
    target = ip_alogs.edge_detection(img)
if algo_type == algorithms[3]:
    target = ip_alogs.color_thresolding_rgb(img)
if algo_type == algorithms[4]:
    target = ip_alogs.color_thresholding_hsv(img)
if algo_type == algorithms[5]:
    target = ip_alogs.conv2d(img)
if algo_type == algorithms[6]:
    target = ip_alogs.rotate_img(img)
if algo_type == algorithms[7]:
    target = ip_alogs.morphological(img)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Original Image")
    st.image(img)

    dimensions = img.shape
    # height, width, number of channels in image
    height = img.shape[0]
    width = img.shape[1]
    channels = img.shape[2]

    info = f'Properties of Image:\nImage Dimension    :{dimensions} \n Image Height     :{height}\n Image Width      :{width}\nNumber of Channels  :{channels}\n'
    st.text(info)

with col2:
    st.subheader(f"Target Image")
    st.image(target)