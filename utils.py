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

def rotation_matrix_2d(ang, use_torch=True, device=None):
    """2D rotation matrix."""
    if use_torch:
        ang = torch.deg2rad(ang)
        return torch.tensor([[torch.cos(ang), -torch.sin(ang)],
                             [torch.sin(ang), torch.cos(ang)]], device=device)
    else:
        ang = np.deg2rad(ang)
        return np.array([[np.cos(ang), -np.sin(ang)],
                         [np.sin(ang), np.cos(ang)]])

def rotation_matrix_3d(angles, use_torch=True, device=None):
    """3D rotation matrix."""
    if use_torch:
        angles = torch.deg2rad(angles)
        ax, ay, az = angles[0], angles[1], angles[2]
        Rx = torch.tensor([[1, 0, 0],
                           [0, torch.cos(ax), -torch.sin(ax)],
                           [0, torch.sin(ax), torch.cos(ax)]], device=device)
        Ry = torch.tensor([[torch.cos(ay), 0, torch.sin(ay)],
                           [0, 1, 0],
                           [-torch.sin(ay), 0, torch.cos(ay)]], device=device)
        Rz = torch.tensor([[torch.cos(az), -torch.sin(az), 0],
                           [torch.sin(az),  torch.cos(az), 0],
                           [0,0, 1]], device=device)
        return torch.matmul(Rz,torch.matmul(Ry,Rx))
    else:
        angles = np.deg2rad(angles)
        ax, ay, az = angles[0], angles[1], angles[2]
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(ax), -np.sin(ax)],
                       [0, np.sin(ax), np.cos(ax)]])
        Ry = np.array([[np.cos(ay), 0, np.sin(ay)],
                       [0, 1, 0],
                       [-np.sin(ay), 0, np.cos(ay)]])
        Rz = np.array([[np.cos(az), -np.sin(az), 0],
                       [np.sin(az),  np.cos(az), 0],
                       [0, 0, 1]])
        return np.matmul(Rz,np.matmul(Ry,Rx))
