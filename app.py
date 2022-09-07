import os
import random
import cv2
import streamlit as st
from PIL import Image

# import library
from swap import swap_eyes, swap_eyebrows, swap_mouth, swap_nose
from utils import mp_detector, get_coordinate
from enhance import enhance_image
st.set_page_config(page_title="Face swapping", page_icon="🙂")
st.title("🙂 Face Swapping")

image_path = ".//image//"
image_list = os.listdir(image_path)

source_name = random.choice(image_list)
source_image = Image.open(os.path.join(image_path, source_name))
st.image(source_image, "Source image")

prototype_list = [i for i in image_list if i != source_name]

components = []
if st.button("Swap"):
    image_b = cv2.imread(os.path.join(image_path, source_name))
    landmark_b = mp_detector(image_b)
    bx, by, bz = get_coordinate(landmark_b, image_b)
    components = [swap_mouth, swap_nose, swap_eyebrows, swap_eyes]
    for i in range(len(components)):
        st.progress(i)
        random_image_name = random.choice(prototype_list)
        image_a = cv2.imread(os.path.join(image_path, random_image_name))
        landmark_a = mp_detector(image_a)
        ax, ay, az = get_coordinate(landmark_a, image_a)
        result = components[i](image_a, ax, ay, image_b, bx, by)
        image_b = result.copy()
    dim = (256, 256)
    result = cv2.resize(result, dim, interpolation = cv2.INTER_AREA)
    result = enhance_image(result)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    result = Image.fromarray(result)
    st.header("Result")
    st.image(result, "Result image")
