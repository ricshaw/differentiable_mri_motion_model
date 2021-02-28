import numpy as np
import nibabel as nib
import os
import cv2
import torch
import matplotlib.pyplot as plt

def normalise_image(image, use_torch=True):
    if use_torch:
        image = torch.abs(image)
    else:
        image = np.abs(image)
    if (image.max() - image.min()) < 1e-5:
        return image - image.min() + 1e-5
    else:
        return (image - image.min()) / (image.max() - image.min())

def load_nii_image(filename):
    image = nib.load(filename)
    image = np.asanyarray(image.dataobj).astype(np.float32)
    return normalise_image(image, use_torch=False)

def load_png(filename):
    image = cv2.imread(filename, cv2.IMREAD_ANYDEPTH).astype(np.float32)
    return normalise_image(image, use_torch=False)
