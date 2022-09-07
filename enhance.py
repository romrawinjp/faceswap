from gfpgan import GFPGANer

bg_upsampler = 'realesrgan'
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
bg_upsampler = RealESRGANer(
    scale=2,
    model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
    model=model,
    tile=400,
    tile_pad=10,
    pre_pad=0,
    half=False)  # need to set False in CPU mode


model_path = ".//GFPGAN//experiments//pretrained_models//GFPGANv1.3.pth"
upscale = 4
arch = 'clean'
channel_multiplier = 2
restorer = GFPGANer(
    model_path=model_path,
    upscale=upscale,
    arch=arch,
    channel_multiplier=channel_multiplier,
    bg_upsampler=bg_upsampler)

import cv2
import numpy
from utils import display
# input_img = cv2.imread("D://faceswap//image//restore_image.png")
# cropped_faces, restored_faces, restored_img = restorer.enhance(input_img, paste_back=True)

def enhance_image(img):
    _, _, restored_img = restorer.enhance(img, paste_back=True)
    return restored_img

# import os
# # D:\faceswap\faceswap\experiments\a01\swapped_image.png
# image_name = "swapped_image.png"
# path = "D://faceswap//faceswap//experiments"
# folder = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10"]
# for f in folder:
#     folder_path = os.path.join(path, f)
#     image = cv2.imread(os.path.join(folder_path, image_name))
#     dim = (256, 256)
#     image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
#     result = enhance_image(image)
#     cv2.imwrite(os.path.join(folder_path, "swapped_enhanced_image.png"), result)
