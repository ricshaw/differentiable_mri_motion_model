import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import cv2
import nibabel as nib
import torch
import torch.nn as nn
import torchkbnufft as tkbn
import utils
from skimage.data import shepp_logan_phantom
from scipy.linalg import logm, expm
from scipy.ndimage import zoom
from piq import ssim, SSIMLoss, MultiScaleSSIMLoss, VSILoss
import matplotlib
matplotlib.use("Agg")

from pytorch3d.transforms.so3 import (
    so3_exponential_map,
    so3_relative_angle,
)

dtype = torch.complex64
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device)


def check(arr1, arr2):
    if torch.is_tensor(arr1):
        arr1 = arr1.detach().numpy()
    if torch.is_tensor(arr2):
        arr2 = arr2.detach().numpy()
    arr1 = arr1.astype(np.float32)
    arr2 = arr2.astype(np.float32)
    print('arr1', arr1)
    print('arr2', arr2)
    print('equal', np.array_equal(arr1, arr2))

def image_loss(target, image):
    target = torch.abs(target)
    image = torch.abs(image)
    target = (target - torch.mean(target)) / torch.std(target)
    image = (image - torch.mean(image)) / torch.std(image)
    return torch.sum( (target - image)**2 ) + SSIMLoss(data_range=1.)

def kspace_loss(F_target, F):
    return torch.sum( torch.abs(F_target.real - F.real) ) + torch.sum( torch.abs(F_target.imag - F.imag) )

def plot_kdata(kdata, ndims=2):
    if ndims == 2:
        plt.figure()
        plt.imshow(np.log10(np.abs(kdata)), cmap='gray')
        plt.tight_layout()
        plt.title('k-space data, log10 scale')
    if ndims == 3:
        fig, axs = plt.subplots(1,3)
        axs[0].imshow(np.log10(np.abs(kdata)[...,int(kdata.shape[2]//2)]), cmap='gray')
        axs[1].imshow(np.log10(np.abs(kdata)[:,int(kdata.shape[1]//2),:]), cmap='gray')
        axs[2].imshow(np.log10(np.abs(kdata)[int(kdata.shape[0]//2),...]), cmap='gray')
        plt.tight_layout()
        plt.suptitle('k-space data, log10 scale')
    plt.show()

def plot_ktraj(kx, ky, kz=None):
    if kz is None:
        kx_np = kx.detach().cpu().numpy()
        ky_np = ky.detach().cpu().numpy()
        plt.figure()
        plt.plot(kx_np[:,:].T, -ky_np[:,:].T)
        plt.axis('equal')
        plt.title('k-space trajectory')
        plt.tight_layout()
    else:
        kx_np = kx.detach().cpu().numpy()
        ky_np = ky.detach().cpu().numpy()
        kz_np = kz.detach().cpu().numpy()
        fig, axs = plt.subplots(1,3)
        axs[0].plot(kx_np[:,:,int(kx_np.shape[2]//2)].T, -ky_np[:,:,int(ky_np.shape[2]//2)].T)
        axs[1].plot(kx_np[:,int(kx_np.shape[1]//2),:].T, -kz_np[:,int(kz_np.shape[1]//2),:].T)
        axs[2].plot(ky_np[int(ky_np.shape[0]//2),...].T, -kz_np[int(kz_np.shape[0]//2),...].T)
        plt.suptitle('k-space trajectory')
        plt.tight_layout()

def plot_ktraj_image(kx, ky, kz=None):
    if kz is None:
        kx_np = kx.detach().cpu().numpy()
        ky_np = ky.detach().cpu().numpy()
        fig, axs = plt.subplots(1,2)
        axs[0].imshow(kx_np)
        axs[0].set_title('kx')
        axs[1].imshow(ky_np)
        axs[1].set_title('ky')
        plt.suptitle('k-space trajectory')
        plt.tight_layout()
    else:
        kx_np = kx.detach().cpu().numpy()
        ky_np = ky.detach().cpu().numpy()
        kz_np = kz.detach().cpu().numpy()
        fig, axs = plt.subplots(1,3)
        axs[0].imshow(kx_np[:,:,int(kx_np.shape[2]//2)])
        axs[1].imshow(ky_np[:,int(ky_np.shape[1]//2),:])
        axs[2].imshow(kz_np[int(kz_np.shape[0]//2),...])
        axs[0].set_title('kx')
        axs[1].set_title('ky')
        axs[2].set_title('kz')
        plt.suptitle('k-space trajectory')
        plt.tight_layout()
    plt.show()

def rotation_matrix_2d(ang):
    """2D rotation matrix."""
    ang = torch.deg2rad(ang)
    return torch.tensor([[torch.cos(ang), -torch.sin(ang)],
                         [torch.sin(ang), torch.cos(ang)]], device=device)

def rotation_matrix_2d_np(ang):
    ang = np.deg2rad(ang)
    return np.array([[np.cos(ang), -np.sin(ang)],
                     [np.sin(ang), np.cos(ang)]])

def rotation_matrix_3d(angles):
    """3D rotation matrix."""
    angles = torch.deg2rad(angles)
    ax, ay, az = angles[0], angles[1], angles[2]
    Rx = torch.tensor([[1, 0, 0],
                       [0, torch.cos(ax), -torch.sin(ax)],
                       [0, torch.sin(ax), torch.cos(ax)]])
    Ry = torch.tensor([[torch.cos(ay), 0, torch.sin(ay)],
                       [0, 1, 0],
                       [-torch.sin(ay), 0, torch.cos(ay)]])
    Rz = torch.tensor([[torch.cos(az), -torch.sin(az), 0],
                       [torch.sin(az),  torch.cos(az), 0],
                       [0,0, 1]])
    return torch.matmul(Rz,torch.matmul(Ry,Rx))

def rotate(ktraj, R):
    return torch.matmul(R, ktraj)

def rotate_np(ktraj, R):
    return np.matmul(R, ktraj)

def translate(F, ktraj, t):
    shape = F.shape
    phase = torch.matmul(t, ktraj)
    shift = torch.exp(1j*phase)
    F = shift * F.flatten()
    return torch.reshape(F, shape)

def translate_opt(F, ktraj, t):
    shape = F.shape
    phase = torch.matmul(t.to(torch.float32).to(device), ktraj.to(device))
    shift_real = torch.cos(phase)
    shift_imag = torch.sin(phase)
    shift = torch.complex(shift_real, shift_imag).to(device)
    F = shift * F.flatten()
    return torch.reshape(F, shape)

def translate_np(F, ktraj, t):
    shape = F.shape
    phase = np.matmul(t, ktraj)
    shift = np.exp(1j*phase)
    F = shift * F.flatten()
    return np.reshape(F, shape)

def sample_movements(n_movements, ndims=2):
    affines = []
    if ndims == 2:
        angles = torch.FloatTensor(n_movements+1,).uniform_(-15.0, 15.0).to(device)
        trans = torch.FloatTensor(n_movements+1,2).uniform_(-10.0, 10.0).to(device)
    if ndims == 3:
        angles = torch.FloatTensor(n_movements+1,ndims).uniform_(-15.0, 15.0).to(device)
        trans = torch.FloatTensor(n_movements+1,ndims).uniform_(-10.0, 10.0).to(device)
        #trans[0,0] = 20.
        #trans[0,1] = 0.
        #trans[0,2] = 0.
    for i in range(n_movements+1):
        ang = angles[i]
        t = trans[i,:]
        A = torch.eye(ndims+1).to(device)
        if ndims == 2:
            R = rotation_matrix_2d(ang).to(device)
        if ndims == 3:
            R = rotation_matrix_3d(ang).to(device)
        A[:ndims,:ndims] = R
        A[:ndims,ndims] = t.to(device)
        print(A)
        affines.append(A)
    return affines

def sample_movements_np(n_movements):
    affines = []
    angles = np.random.uniform(-10.0, 10.0, (n_movements,))
    trans = np.random.uniform(-15.0, 15.0, (n_movements,2))
    affines.append(np.eye(3))
    for i in range(n_movements):
        ang = angles[i]
        t = trans[i,:]
        A = np.eye(3)
        R = rotation_matrix_np(ang)
        A[:2,:2] = R
        A[:2,2] = t
        affines.append(A)
    return affines

def sample_movements_log(n_movements):
    log_R = 0.05*torch.randn(n_movements+1, 3, dtype=torch.float32, device=device)
    t = 10.0*torch.randn(n_movements+1, 3, dtype=torch.float32, device=device)
    log_R[0,:] = 0.
    t[0,:] = 0.
    R = so3_exponential_map(log_R).to(device)
    affines = []
    for i in range(n_movements+1):
        A = torch.eye(4)
        A[:3,:3] = R[i,:]
        A[:3,3] = t[i,:]
        print(A)
        affines.append(A)
    return affines

def gen_movements_log(n_movements, t):
    #R = so3_exponential_map(log_R)
    #log_R[0,:] = 0.
    #t[0,:] = 0.
    affines = []
    for i in range(n_movements+1):
        A = torch.eye(4)
        A[:3,:3] = R[i,:]
        A[:3,3] = t[i,:]
        affines.append(A)
    return affines

def combine_affines(affines):
    combinedAffines = []
    Aprev = affines[0]
    for i in range(len(affines)):
        A = affines[i]
        #combinedA = torch.matmul(Aprev,A)
        combinedA = torch.matrix_exp( logm(A) + logm(Aprev) )
        combinedAffines.append(combinedA)
        Aprev = combinedA
    return combinedAffines

def gen_masks(n_movements, locs, grid_size):
    masks = []
    if n_movements > 0:
        mask = torch.zeros(grid_size, device=device)
        mask[0:locs[0],:] = 1
        masks.append(mask)
        for i in range(1,n_movements):
            mask = torch.zeros(grid_size, device=device)
            mask[locs[i-1]:locs[i],:] = 1
            masks.append(mask)
        mask = torch.zeros(grid_size, device=device)
        mask[locs[-1]::,:] = 1
        masks.append(mask)
    else:
        masks.append(torch.ones(grid_size, device=device))
    return masks

def gen_masks_np(n_movements, nlines, klen):
    masks = []
    if n_movements > 0:
        lines = sorted(np.random.choice(nlines, n_movements))
        mask = np.zeros((nlines, klen))
        mask[0:lines[0],:] = 1
        masks.append(mask)
        for i in range(1,n_movements):
            mask = np.zeros((nlines, klen))
            mask[lines[i-1]:lines[i],:] = 1
            masks.append(mask)
        mask = np.zeros((nlines, klen))
        mask[lines[-1]::,:] = 1
        masks.append(mask)
    else:
        masks.append(np.ones((nlines, klen)))
    return masks

def gen_ktraj(nlines, klen, kdepth=None):
    if kdepth is None:
        kx = torch.linspace(-np.pi, np.pi, klen)
        ky = torch.linspace(-np.pi, np.pi, nlines)
        kx, ky = torch.meshgrid(kx, ky)
        kx = kx.T.to(device)
        ky = ky.T.to(device)
        return kx, ky
    else:
        kx = torch.linspace(-np.pi, np.pi, klen)
        ky = torch.linspace(-np.pi, np.pi, nlines)
        kz = torch.linspace(-np.pi, np.pi, kdepth)
        kx, ky, kz = torch.meshgrid(kx, ky, kz)
        kx = kx.permute(1,0,2).to(device)
        ky = ky.permute(1,0,2).to(device)
        kz = kz.permute(1,0,2).to(device)
        return kx, ky, kz

def gen_ktraj_np(nlines, klen):
    kx = np.linspace(-np.pi, np.pi, klen)
    ky = np.linspace(-np.pi, np.pi, nlines)
    kx, ky = np.meshgrid(kx, ky)
    return kx, ky

def to_1d(kx, ky, kz=None):
    if kz is None:
        return torch.stack((ky.flatten(), kx.flatten()))
    else:
        return torch.stack((ky.flatten(), kx.flatten(), kz.flatten()))

def to_1d_np(kx, ky):
    return np.stack((ky.flatten(), kx.flatten()))

def to_2d(ktraj, grid_size):
    ndims = len(grid_size)
    kx = torch.reshape(ktraj[1,...], grid_size)
    ky = torch.reshape(ktraj[0,...], grid_size)
    if ndims == 2:
        return kx, ky
    if ndims == 3:
        kz = torch.reshape(ktraj[2,...], grid_size)
        return kx, ky, kz


def to_2d_np(ktraj, nlines, klen):
    kx = ktraj[1,...]
    ky = ktraj[0,...]
    kx = np.reshape(kx, (nlines, klen))
    ky = np.reshape(ky, (nlines, klen))
    return kx, ky

def gen_movement(image, kx, ky, kz=None, grid_size=None, n_movements=None, locs=None, debug=False):

    # Convert image to tensor and unsqueeze coil and batch dimension
    ndims = len(image.shape)
    im_size = image.shape
    image = torch.tensor(image).to(dtype).unsqueeze(0).unsqueeze(0).to(device)
    print('image shape: {}'.format(image.shape))

    # Build ktraj
    ktraj = to_1d(kx, ky, kz).to(device)
    korig = ktraj.clone()

    affines = sample_movements(n_movements, ndims)
    #affines = combine_affines(affines)

    # Generate k-space masks
    masks = gen_masks(n_movements, locs, grid_size)
    if debug:
        fig = plt.figure()
        nplots = np.minimum(10,len(masks))
        for i in range(nplots):
            m = masks[i].detach().cpu().numpy()
            if ndims == 2:
                ax = fig.add_subplot(1,nplots,i+1)
                plt.imshow(m)
            if ndims == 3:
                ax1 = fig.add_subplot(3,nplots,i+1)
                plt.imshow(m[...,int(m.shape[2]//2)])
                ax2 = fig.add_subplot(3,nplots,i+1+nplots)
                plt.imshow(m[:,int(m.shape[1]//2),:])
                ax3 = fig.add_subplot(3,nplots,i+1+2*nplots)
                plt.imshow(m[int(m.shape[0]//2),...])

    # Apply rotation component
    ktrajs = []
    if ndims == 2:
        kx_new = torch.zeros_like(kx, device=device)
        ky_new = torch.zeros_like(ky, device=device)
        kz_new = None
        for i in range(len(affines)):
            R = affines[i][:ndims,:ndims].to(device)
            ktraji = rotate(ktraj, R)
            ktrajs.append(ktraji)
            kxi, kyi = to_2d(ktraji, grid_size)
            kx_new += masks[i] * kxi
            ky_new += masks[i] * kyi
    if ndims == 3:
        kx_new = torch.zeros_like(kx, device=device)
        ky_new = torch.zeros_like(ky, device=device)
        kz_new = torch.zeros_like(kz, device=device)
        for i in range(len(affines)):
            R = affines[i][:ndims,:ndims].to(device)
            ktraji = rotate(ktraj, R)
            ktrajs.append(ktraji)
            kxi, kyi, kzi = to_2d(ktraji, grid_size)
            kx_new += masks[i] * kxi
            ky_new += masks[i] * kyi
            kz_new += masks[i] * kzi

    mid = kx_new.shape[0]//2
    b = int(kx_new.shape[0] * 3/100.0)
    b = 0
    print('k-space centre:', b)
    kx_new[mid-b:mid+b,:] = kx[mid-b:mid+b,:]
    ky_new[mid-b:mid+b,:] = ky[mid-b:mid+b,:]
    ktraj = to_1d(kx_new, ky_new, kz_new)

    # Plot ktraj
    if debug:
        plot_ktraj(kx_new, ky_new, kz_new)
        plot_ktraj_image(kx_new, ky_new, kz_new)

    ktraj = torch.tensor(ktraj).to(torch.float)
    print('ktraj shape: {}'.format(ktraj.shape))

    # create NUFFT objects, use 'ortho' for orthogonal FFTs
    nufft_ob = tkbn.KbNufft(
        im_size=im_size,
        grid_size=grid_size,
        ).to(dtype).to(device)
    adjnufft_ob = tkbn.KbNufftAdjoint(
        im_size=im_size,
        grid_size=grid_size,
        ).to(image).to(device)

    # Calculate k-space data
    kdata = nufft_ob(image, korig).to(device)
    print('kdata', kdata.shape)

    # Apply translational component
    print('Applying translational component')
    kdata = torch.reshape(kdata, grid_size)
    kdata_new = torch.zeros_like(kdata)
    for i in range(len(masks)):
        t = affines[i][:ndims,ndims]
        print(i,t)
        kdata_i = translate_opt(kdata, ktraj, t)
        kdata_new += masks[i] * kdata_i
    kdata_new[mid-b:mid+b,:] = kdata[mid-b:mid+b,:]
    kdata = kdata_new.flatten().unsqueeze(0).unsqueeze(0)

    # Plot the k-space data on log-scale
    if debug:
        kdata_numpy = np.reshape(kdata.detach().cpu().numpy(), grid_size)
        plot_kdata(kdata_numpy, ndims)

    # adjnufft back
    image_out = adjnufft_ob(kdata, ktraj)
    return image_out, kdata, kx_new, ky_new, kz_new


def gen_movement_opt(image, ndims,
                     n_movements, locs, ts, angles,
                     kdata, korig, kx, ky, kz=None,
                     grid_size=None, im_size=None,
                     adjnufft_ob=None,
                     masks=None):

    # Apply rotation component
    if ndims == 2:
        kx_new = torch.zeros_like(kx, device=device)
        ky_new = torch.zeros_like(ky, device=device)
        kz_new = None
        for i in range(len(masks)):
            ang = torch.deg2rad(angles[i])
            kyi = torch.cos(ang)*ky - torch.sin(ang)*kx
            kxi = torch.sin(ang)*ky + torch.cos(ang)*kx
            kx_new += masks[i] * kxi
            ky_new += masks[i] * kyi

    if ndims == 3:
        kx_new = torch.zeros_like(kx, device=device)
        ky_new = torch.zeros_like(ky, device=device)
        kz_new = torch.zeros_like(kz, device=device)
        for i in range(len(masks)):
            ang = torch.deg2rad(angles[i])
            ax, ay, az = ang[0], ang[1], ang[2]
            cax, sax = torch.cos(ax), torch.sin(ax)
            cay, say = torch.cos(ay), torch.sin(ay)
            caz, saz = torch.cos(az), torch.sin(az)

            kyi = (cax*cay)*ky + (cax*say*saz - sax*caz)*kx + (cax*say*caz + sax*saz)*kz
            kxi = (sax*cay)*ky + (sax*say*saz + cax*caz)*kx + (sax*say*caz - cax*saz)*kz
            kzi = (-say)*ky + (cay*saz)*kx + (cay*caz)*kz
            kx_new += masks[i] * kxi
            ky_new += masks[i] * kyi
            kz_new += masks[i] * kzi

    # Apply translational component
    if True:
        print('Applying translational component')
        kdata = torch.reshape(kdata, grid_size)
        kdata_new = torch.zeros_like(kdata, device=device)
        for i in range(len(masks)):
            t = ts[i]
            print(i,t)
            if ndims == 2:
                kdata_i = translate_opt(kdata, torch.stack((ky_new.flatten(), kx_new.flatten())), t)
            if ndims == 3:
                kdata_i = translate_opt(kdata, torch.stack((ky_new.flatten(), kx_new.flatten(), kz_new.flatten())), t)
            kdata_new += masks[i] * kdata_i
        #kdata_new[mid-b:mid+b,:] = kdata[mid-b:mid+b,:]
        kdata = kdata_new.flatten().unsqueeze(0).unsqueeze(0)
        #kdata = kdata.flatten().unsqueeze(0).unsqueeze(0)

    # adjnufft back
    if ndims == 2:
        image_out = adjnufft_ob(kdata, torch.stack((ky_new.flatten(), kx_new.flatten())))
    if ndims == 3:
        image_out = adjnufft_ob(kdata, torch.stack((ky_new.flatten(), kx_new.flatten(), kz_new.flatten())))

    image_out = torch.abs(image_out).to(float)
    image_out = (image_out - image_out.min()) / (image_out.max() - image_out.min())
    return image_out, kdata, kx_new, ky_new, kz_new


if __name__ == '__main__':

    # Load image
    #image = shepp_logan_phantom().astype(np.complex)
    image = utils.load_png('./data/sample_2d.png').astype(np.complex)
    #image = utils.load_nii_image('./data/sample_3d.nii.gz')
    #image = zoom(image, 0.75).astype(np.complex)
    ndims = len(image.shape)

    # Visualise
    if ndims == 2:
        fig = plt.figure()
        plt.imshow(np.abs(image), cmap='gray')
    if ndims == 3:
        fig, axs = plt.subplots(1,3)
        axs[0].imshow(np.abs(image)[...,int(image.shape[2]//2)], cmap='gray')
        axs[1].imshow(np.abs(image)[:,int(image.shape[1]//2),:], cmap='gray')
        axs[2].imshow(np.abs(image)[int(image.shape[0]//2),...], cmap='gray')
    plt.suptitle('Input image')
    plt.tight_layout()

    # Create a k-space trajectory
    sampling_rate = 1.0
    if ndims == 2:
        kr = int(image.shape[0] * sampling_rate)
        kc = int(image.shape[1] * sampling_rate)
        grid_size = (kr, kc)
        kx, ky = gen_ktraj(kr, kc)
        kz = None
    if ndims == 3:
        kr = int(image.shape[0] * sampling_rate)
        kc = int(image.shape[1] * sampling_rate)
        kd = int(image.shape[2] * sampling_rate)
        grid_size = (kr, kc, kd)
        kx, ky, kz = gen_ktraj(kr, kc, kd)

    # Generate movement
    n_movements = 10
    locs = sorted(np.random.choice(kr, n_movements))
    image_out, kdata_out, kx_out, ky_out, kz_out = gen_movement(image,
                                                        kx, ky, kz,
                                                        grid_size=grid_size,
                                                        n_movements=n_movements,
                                                        locs=locs,
                                                        debug=True)

    # Show the images
    image_out_np = np.abs(np.squeeze(image_out.detach().cpu().numpy()))
    image_out_np = utils.normalise_image(image_out_np)
    diff = np.abs(image - image_out_np)
    err = diff.sum() / image_out_np.size
    print('err', err)

    if ndims == 2:
        plt.figure()
        plt.imshow(image_out_np, cmap='gray', vmin=0, vmax=1)
        plt.title('Output image')
        plt.tight_layout()

        plt.figure()
        plt.imshow(diff, cmap='jet', vmin=0, vmax=1)
        plt.title('Diff image')
        plt.tight_layout()

    if ndims == 3:
        fig, axs = plt.subplots(1,3)
        axs[0].imshow(image_out_np[...,int(image_out_np.shape[2]//2)], cmap='gray', vmin=0, vmax=1)
        axs[1].imshow(image_out_np[:,int(image_out_np.shape[1]//2),:], cmap='gray', vmin=0, vmax=1)
        axs[2].imshow(image_out_np[int(image_out_np.shape[0]//2),...], cmap='gray', vmin=0, vmax=1)
        plt.suptitle('Output image')
        plt.tight_layout()

        fig, axs = plt.subplots(1,3)
        axs[0].imshow(diff[...,int(diff.shape[2]//2)], cmap='jet', vmin=0, vmax=1)
        axs[1].imshow(diff[:,int(diff.shape[1]//2),:], cmap='jet', vmin=0, vmax=1)
        axs[2].imshow(diff[int(diff.shape[0]//2),...], cmap='jet', vmin=0, vmax=1)
        plt.suptitle('Diff image')
        plt.tight_layout()
    plt.show()


    target = image_out.clone().to(float)
    target = torch.abs(target)
    target = (target - target.min()) / (target.max() - target.min())
    target.requires_grad = False

    kdata_target = kdata_out.clone()
    kdata_target.requires_grad = False

    kx_target = kx_out.clone()
    ky_target = ky_out.clone()
    kx_target.requires_grad = False
    ky_target.requires_grad = False
    if ndims == 3:
        kz_target = kz_out.clone()
        kz_target.requires_grad = False


    image = torch.tensor(image).to(float)
    image = torch.abs(image.squeeze())
    image = (image - image.min()) / (image.max() - image.min())
    image.requires_grad = True
    print('target', target.dtype, target.shape, target.min(), target.max())
    print('input', image.dtype, image.shape, image.min(), image.max())

    image_np = image.detach().cpu().numpy()
    target_np = target.squeeze().detach().cpu().numpy()
    if ndims == 2:
        fig, axs = plt.subplots(1,2)
        axs[0].imshow(image_np, cmap='gray')
        axs[0].set_title('start')
        axs[1].imshow(target_np, cmap='gray')
        axs[1].set_title('target')
    if ndims == 3:
        fig, axs = plt.subplots(2,3)
        axs[0,0].imshow(image_np[...,int(image_np.shape[2]//2)], cmap='gray')
        axs[0,1].imshow(image_np[:,int(image_np.shape[1]//2),:], cmap='gray')
        axs[0,2].imshow(image_np[int(image_np.shape[0]//2),...], cmap='gray')
        axs[0,0].set_title('start')
        axs[1,0].imshow(target_np[...,int(target_np.shape[2]//2)], cmap='gray')
        axs[1,1].imshow(target_np[:,int(target_np.shape[1]//2),:], cmap='gray')
        axs[1,2].imshow(target_np[int(target_np.shape[0]//2),...], cmap='gray')
        axs[1,0].set_title('target')
    plt.show()

     # Convert image to tensor and unsqueeze coil and batch dimension
    im_size = image.shape
    image_tensor = image.to(dtype).unsqueeze(0).unsqueeze(0).to(device)

    # Init k-space trajectory
    if ndims == 2:
        kx_init, ky_init = gen_ktraj(kr, kc)
        kx = kx_init.clone().detach()
        ky = ky_init.clone().detach()
        kx.requires_grad = False
        ky.requires_grad = False
        korig = torch.stack((ky.flatten(), kx.flatten()))
    if ndims == 3:
        kx_init, ky_init, kz_init = gen_ktraj(kr, kc, kd)
        kx = kx_init.clone().detach()
        ky = ky_init.clone().detach()
        kz = kz_init.clone().detach()
        kx.requires_grad = False
        ky.requires_grad = False
        kz.requires_grad = False
        korig = torch.stack((ky.flatten(), kx.flatten(), kz.flatten()))

    nufft_ob = tkbn.KbNufft(
        im_size=im_size,
        grid_size=grid_size,
        ).to(dtype).to(device)
    adjnufft_ob = tkbn.KbNufftAdjoint(
        im_size=im_size,
        grid_size=grid_size,
        ).to(image_tensor).to(device)
    kdata_orig = nufft_ob(image_tensor, korig).to(device)
    kdata = kdata_orig.clone().detach()
    kdata.requires_grad = True

    masks = gen_masks(n_movements, locs, grid_size)

    # Optimise
    print('Optimising...')

    ts_init = 0.0*torch.randn(n_movements+1, ndims, dtype=torch.float32, device=device)
    ts_init[0,:] = 0.
    #ts_init[0,0] = -20.0
    ts = ts_init.clone().detach()
    ts.requires_grad = True
    print(ts)

    if ndims == 2:
        angles_init = torch.FloatTensor(n_movements+1,).uniform_(-25.0, -25.0).to(device)
    if ndims == 3:
        angles_init = torch.FloatTensor(n_movements+1,ndims).uniform_(-0.0, 0.0).to(device)
        angles_init[0,0] = 45.0
    angles = angles_init.clone().detach()
    angles.requires_grad = True
    print(angles)

    #optimizer = torch.optim.Adam([ts], lr=1.)
    #optimizer = torch.optim.Adam([angles], lr=1.)
    optimizer = torch.optim.Adam([ts, angles], lr=1.)
    #optimizer = torch.optim.Adam([ts, kx, ky], lr=1.)

    l1_loss = nn.L1Loss()
    #ssim_loss = SSIMLoss(data_range=1.)
    ms_ssim_loss = MultiScaleSSIMLoss()
    #vsi_loss = VSILoss()


    fig = plt.figure()
    plt.tight_layout()
    ims = []
    n_iter = 100
    losses = []
    for i in range(n_iter):
        optimizer.zero_grad()

        #angles = angles - 2
        image_out, kdata_out, kx_out, ky_out, kz_out = gen_movement_opt(image_tensor, ndims,
                                                                n_movements, locs, ts, angles,
                                                                kdata, korig, kx, ky, kz,
                                                                grid_size, im_size,
                                                                adjnufft_ob,
                                                                masks)
        print('output:', image_out.shape, image_out.dtype, 'target:', target.shape, target.dtype)

        # Loss functions
        loss1 = 200. * l1_loss(image_out, target)
        loss2 = l1_loss(kdata_out.real, kdata_target.real)
        loss3 = l1_loss(kdata_out.imag, kdata_target.imag)
        loss4 = 100. * l1_loss(kx_out, kx_target)
        loss5 = 100. * l1_loss(ky_out, ky_target)
        loss = loss1 + loss2 + loss3 + loss4 + loss5
        if kz_out is not None:
            loss6 = 100. * l1_loss(kz_out, kz_target)
            loss += loss6
            print('iter:', i, 'losses:', loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item(), loss6.item())
        else:
            print('iter:', i, 'losses:', loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item())

        print('ts:', ts.detach().cpu().numpy())
        print('angles:', angles.detach().cpu().numpy())
        loss.backward()
        print('grads', angles.grad, ts.grad, kx.grad, ky.grad)
        optimizer.step()

        losses.append(loss.item())

        if True:
            image_out_np = image_out.squeeze().detach().cpu().numpy()
            if ndims == 2:
                plt.subplot(1,3,1)
                plt.title('image')
                plt.axis('off')
                im1 = plt.imshow(image_out_np, cmap='gray', animated=True)
                plt.subplot(1,3,2)
                plt.title('target')
                plt.axis('off')
                im2 = plt.imshow(target_np, cmap='gray', animated=True)
                plt.subplot(1,3,3)
                plt.title('loss')
                plt.xlabel('iterations')
                im3, = plt.plot(losses, 'b-')
                plt.subplots_adjust(wspace=0.25)
                ims.append([im1,im2,im3])

            if ndims == 3:
                plt.subplot(2,4,1)
                plt.axis('off')
                im1 = plt.imshow(image_out_np[...,int(image_out_np.shape[2]//2)], cmap='gray')
                plt.subplot(2,4,2)
                plt.title('image')
                plt.axis('off')
                im2 = plt.imshow(image_out_np[:,int(image_out_np.shape[2]//2),:], cmap='gray')
                plt.subplot(2,4,3)
                plt.axis('off')
                im3 = plt.imshow(image_out_np[int(image_out_np.shape[2]//2),...], cmap='gray')

                plt.subplot(2,4,5)
                plt.axis('off')
                im4 = plt.imshow(target_np[...,int(target_np.shape[2]//2)], cmap='gray')
                plt.subplot(2,4,6)
                plt.title('target')
                plt.axis('off')
                im5 = plt.imshow(target_np[:,int(target_np.shape[2]//2),:], cmap='gray')
                plt.subplot(2,4,7)
                im6 = plt.imshow(target_np[int(target_np.shape[2]//2),...], cmap='gray')

                plt.subplot(2,4,4)
                plt.title('loss')
                plt.xlabel('iterations')
                im7, = plt.plot(losses, 'b-')
                ims.append([im1,im2,im3,im4,im5,im6,im7])

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True)
    from matplotlib import rcParams
    rcParams['animation.convert_path'] = r'/usr/bin/convert'
    ani.save('out.gif', writer='imagemagick', fps=15)
    plt.show()

'''
    axs[0].imshow(image_out.detach().cpu().numpy(), cmap='gray')
    axs[0].set_title('iter: %d' % i)
    axs[1].imshow(target.detach().cpu().numpy(), cmap='gray')
    axs[1].set_title('target')
    axs[2].plot(losses)
    axs[2].set_title('loss')
    axs[2].set_xlabel('iterations')
    plt.show()
'''
