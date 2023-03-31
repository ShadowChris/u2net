import os
from collections import defaultdict
from glob import glob

import PIL
import numpy as np
import streamlit as st
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, CenterCrop
import torchvision.transforms.functional as F
from pymatting import *

from lib import U2NET_full
from lib.utils.oom import free_up_memory
import cv2


## 打印图片作为测试
def cv2plt(img):
    cv2.imshow("img", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # print(new_image.astype(np.float32))
    #  add below code
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def create_ui(samples):
    st.sidebar.title('u2net - segmentation')

    st.sidebar.title('Select a model')
    model_select = st.sidebar.selectbox('', [
        'u2net_human_seg',
        'checkpoint.pth'
    ], index=1)

    st.sidebar.title('Select a sample')
    sample_select = st.sidebar.selectbox('', samples)

    return model_select, sample_select


def load_samples(folder_path='./dataset/demo'):
    assert os.path.isdir(folder_path), f'Unable to open {folder_path}'
    samples = glob(os.path.join(folder_path, f'*.jpg'))
    print(os.getcwd())
    return samples


device = 'cuda'
samples = load_samples()
model_select, sample_select = create_ui(samples)


def square_pad(image, fill=255):
    w, h = image.size
    max_wh = np.max([w, h])
    hp = int((max_wh - w) / 2)
    vp = int((max_wh - h) / 2)
    padding = (hp, vp, hp, vp)
    return F.pad(image, padding, fill, 'constant')


def get_transform():
    transforms = []
    # transforms.append(Resize(440)) # TBD: keep aspect ratio
    transforms.append(ToTensor())
    transforms.append(Normalize(mean=[.5, .5, .5],
                                std=[.5, .5, .5]))

    return Compose(transforms)


device = 'cpu'
checkpoint = torch.load(f'./checkpoints/{model_select}', map_location=device)
model = U2NET_full().to(device=device)

if 'model' in checkpoint:
    model.load_state_dict(checkpoint['model'])
else:
    model.load_state_dict(checkpoint)

image = Image.open(sample_select).convert('RGB')
# image = square_pad(image, 0)
print(image.size)
image = image.resize((image.size[0], image.size[1]), Image.ANTIALIAS)
st.image(image, width=800)

transforms = get_transform()

model.eval()
with torch.no_grad():
    x = transforms(image)
    x = x.to(device=device).unsqueeze(dim=0)
    y_hat, _ = model(x)

    alpha_image = y_hat.mul(255)
    alpha_image = Image.fromarray(alpha_image.squeeze().cpu().detach().numpy()).convert('L')
    st.image(alpha_image, width=800)

image = np.asarray(image)
# background = np.zeros(image.shape)
# background[:, :] = [0, 177 / 255, 64 / 255]
##Option2: 透明背景
h, w, c = image.shape
background = np.zeros((h, w, 3))

alpha = y_hat.squeeze().cpu().detach()
alpha = np.asarray(alpha)
# alpha = (alpha * 255).astype(np.uint8)
image = image.astype(np.float32) / 255

foreground = estimate_foreground_ml(
    image, alpha)  # , n_big_iterations=1, n_small_iterations=1, regularization=10e-10

new_image = blend(foreground, background, alpha).astype(np.float32)
st.image(new_image, width=800)
cv2plt(new_image)
del y_hat
free_up_memory()
