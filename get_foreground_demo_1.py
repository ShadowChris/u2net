import os
from glob import glob
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
import torchvision.transforms.functional as F
from pymatting import *

from lib import U2NET_full
from lib.utils.oom import free_up_memory
import cv2

input_path = "data/input"
img_name = "3.jpg"

mask_path = "data/mask"
output_path = "data/output"

ckpt_name = "checkpoints/checkpoint.pth"

# 图片采样大小
# define hyper-parameters
ref_size = 448

def load_samples(folder_path=input_path):
    assert os.path.isdir(folder_path), f'Unable to open {folder_path}'
    samples = glob(os.path.join(folder_path, f'*.jpg'))
    print(os.getcwd())
    return samples


device = 'cpu'
samples = load_samples()


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


## 打印图片作为测试
def cv2plt(img):
    cv2.imshow("img", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # print(new_image.astype(np.float32))
    #  add below code
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':

    device = 'cuda'
    checkpoint = torch.load(ckpt_name, map_location=device)
    model = U2NET_full().to(device=device)

    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    # -----------------------
    # 加载图片
    # for sample_select in samples:
    # sample_select = samples[0]
    print('Process image: {0}'.format(img_name))
    image = Image.open(os.path.join(input_path, img_name)).convert('RGB')
    transforms = get_transform()
    # image = transforms(image).unsqueeze(dim=0)

    # 图片尺寸缩小
    # resize image for input
    image = square_pad(image, 0)
    # image = image.resize((im_rw, im_rh), Image.ANTIALIAS)
    image = image.resize((ref_size, ref_size), Image.ANTIALIAS)

    alpha_image = None
    model.eval()
    with torch.no_grad():
        # x = image
        x = transforms(image).unsqueeze(dim=0)
        x = x.to(device=device)
        y_hat, _ = model(x)

        alpha_image = y_hat.mul(255)
        alpha_image = Image.fromarray(alpha_image.squeeze().cpu().detach().numpy()).convert('L')
        alpha_image.show()
        # st.image(alpha_image, width=800)

    image = np.asarray(image)

    # ##Option1: 绿背景
    # background = np.zeros(image.shape)
    # background[:, :] = [0, 177 / 255, 64 / 255]

    # ##Option2: 透明背景
    h, w, c = image.shape
    background = np.zeros((h, w, 3))
    alpha = y_hat.squeeze().cpu().detach()
    alpha = np.asarray(alpha)
    # alpha = (alpha * 255).astype(np.uint8)
    image = image.astype(np.float32) / 255

    foreground = estimate_foreground_ml(
        image, alpha)  # , n_big_iterations=1, n_small_iterations=1, regularization=10e-10

    new_image = blend(foreground, background, alpha).astype(np.float32)

    # cv2plt(new_image)
    ## Combine the new_image (foreground + transparent background) with the alpha channel
    alpha = alpha[..., np.newaxis]
    rgba_image = np.concatenate((new_image, alpha), axis=2)
    rgba_image = (rgba_image * 255).astype(np.uint8)
    rgba_pil_image = Image.fromarray(rgba_image, mode="RGBA")

    ## Save the result as a PNG file
    rgba_pil_image.save(os.path.join(output_path, img_name.split(".")[0] + ".png"))

    del y_hat
    free_up_memory()
    print("Done")
