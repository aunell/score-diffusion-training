import torchvision
import numpy as np
from PIL import Image

def normalize_0_to_1(image):
    "normalize numpy array"
    image_02perc = np.min(image)
    image_98perc = np.max(image)
    image_normalized = (image - image_02perc) / (image_98perc - image_02perc)
    image_normalized=np.clip(image_normalized, 0, 1)
    return image_normalized

def visualizeTensor(img, path, index):
    "torch tensor -->save image"
    torchvision.utils.save_image(img, path+index, nrow=int(img.shape[0] ** 0.5))

def visualizeArray(img, path, index):
    "numpy array -->save image"
    imgNormalized=normalize_0_to_1(img)*255
    im = Image.fromarray(imgNormalized)
    imgFinal = im.convert('L')
    imgFinal.save(path+index)