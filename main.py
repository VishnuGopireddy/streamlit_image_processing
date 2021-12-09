import streamlit as st
from PIL import Image
import io
import numpy as np
import cv2

uploaded_files = None
uploaded_files = st.file_uploader("Add IMAGES", type=['.jpg', '.jpeg', '.png'], accept_multiple_files=True)

st.write(f'Total uploaded files are {len(uploaded_files)}')
if uploaded_files is not None:
    files = {}
    for uploaded_file in uploaded_files:
        img = Image.open(uploaded_file)
        # st.write(img.height)
        # print(uploaded_file)
        # print(type(uploaded_file))
        tag = img._getexif()
        if tag is not None:
            date = tag[36867]
            yyyymmdd = date.split(' ')[0]
            mmdd = yyyymmdd.split(':')[1:][::-1]
            ddmm = '_'.join(mmdd)
            if ddmm not in files.keys():
                files.update({ddmm : 0})
            else:
                files[ddmm] = files[ddmm] + 1
            #save file with {ddmm}_count
            img = np.array(img)
            is_success, im_buf_arr = cv2.imencode(".jpg", img)
            io_buf = io.BytesIO(im_buf_arr)
            byte_im = io_buf.getvalue()
            # im = Image.fromarray(target)
            # b = io.BytesIO()
            st.write(f'Date from file: {ddmm}')
            st.download_button(f'Download f{uploaded_file.name}', data=byte_im, file_name=f'{ddmm}_{files[ddmm]}')
            io_buf.seek(0)
        else:
            st.write(f'meta data for {uploaded_file.name} is not available')

        st.write('-----------------------------------------------------------')

    st.write("Count of available files")
    st.write(files)

