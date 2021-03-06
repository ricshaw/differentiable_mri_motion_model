import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import cv2
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchkbnufft as tkbn
import utils
import visualisation
from skimage.data import shepp_logan_phantom
from scipy.linalg import logm, expm
from scipy.ndimage import zoom
from piq import ssim, SSIMLoss, MultiScaleSSIMLoss, VSILoss

from pytorch3d.transforms.so3 import (
    so3_exponential_map,
    so3_relative_angle,
)

# Config
debug = True
animate = False
dtype = torch.float32
complex_dtype = torch.complex64
numpoints = 6
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device)
matplotlib.use("Agg") if animate else None
torch.autograd.set_detect_anomaly(True)
mode = 1

def sample_movements(n_movements, ndims, angles_std=5.0, trans_std=10.0):
    """Sample movement affine transforms."""
    affines = []
    if ndims == 2:
        #angles = torch.FloatTensor(n_movements+1,).uniform_(-15.0, 15.0).to(device)
        #trans = torch.FloatTensor(n_movements+1,2).uniform_(-10.0, 10.0).to(device)
        angles = angles_std * torch.randn((n_movements+1,), dtype=dtype, device=device)
        trans = trans_std * torch.randn((n_movements+1,2), dtype=dtype, device=device)
    if ndims == 3:
        #angles = torch.FloatTensor(n_movements+1,ndims).uniform_(-15.0, 15.0).to(device)
        #trans = torch.FloatTensor(n_movements+1,ndims).uniform_(-10.0, 10.0).to(device)
        angles = angles_std * torch.randn((n_movements+1,ndims), dtype=dtype, device=device)
        trans = trans_std * torch.randn((n_movements+1,ndims), dtype=dtype, device=device)
    for i in range(n_movements+1):
        ang = angles[i]
        t = trans[i,:]
        A = torch.eye(ndims+1).to(device)
        if ndims == 2:
            R = utils.rotation_matrix_2d(ang, device=device)
        if ndims == 3:
            R = utils.rotation_matrix_3d(ang, device=device)
        A[:ndims,:ndims] = R.to(device)
        A[:ndims,ndims] = t.to(device)
        affines.append(A)
    return affines, angles, trans

def sample_movements_log(n_movements, ndims):
    """Sample movements in log space."""
    if ndims == 3:
        log_R = 0.05*torch.randn(n_movements+1, 3, dtype=dtype, device=device)
        trans = 10.0*torch.randn(n_movements+1, 3, dtype=dtype, device=device)
        log_R[0,:] = 0.
        trans[0,:] = 0.
        R = so3_exponential_map(log_R).to(device)
        affines = []
        for i in range(n_movements+1):
            A = torch.eye(4).to(device)
            A[:3,:3] = R[i,:]
            A[:3,3] = trans[i,:]
            affines.append(A)
        return affines, log_R, trans

def gen_masks(n_movements, locs, grid_size, use_torch=True):
    """Generate k-space masks."""
    if use_torch:
        masks = []
        if n_movements > 0:
            mask = torch.arange(0,locs[0],dtype=torch.long,device=device)
            masks.append(mask)
            for i in range(1,n_movements):
                mask = torch.arange(locs[i-1],locs[i],dtype=torch.long,device=device)
                masks.append(mask)
            mask = torch.arange(locs[-1],grid_size[0],dtype=torch.long,device=device)
            masks.append(mask)
        else:
            masks.append(torch.arange(0,grid_size[0],dtype=torch.long,device=device))
        return masks
    else:
        masks = []
        if n_movements > 0:
            mask = np.zeros(grid_size)
            mask[0:locs[0],...] = 1
            masks.append(mask)
            for i in range(1,n_movements):
                mask = np.zeros(grid_size)
                mask[locs[i-1]:locs[i],...] = 1
                masks.append(mask)
            mask = np.zeros(grid_size)
            mask[locs[-1]::,...] = 1
            masks.append(mask)
        else:
            masks.append(np.ones(grid_size))
        return masks

def gen_mask_opt(log_probs, grid_size, ndims):
    logits = log_probs * torch.ones((grid_size[0],2), dtype=dtype, device=device)
    rows = F.gumbel_softmax(logits, tau=1, hard=True, eps=1e-10, dim=-1)[...,0].unsqueeze(0)
    if ndims == 2:
        mask = torch.einsum('ij,jk->jk', [rows, torch.ones(grid_size, device=device)])
    if ndims == 3:
        mask = torch.einsum('ij,jkl->jkl', [rows, torch.ones(grid_size, device=device)])
    n_movements = torch.sum(rows).to(int).item()
    return mask, n_movements

def gen_ktraj(nlines, klen, kdepth=None, use_torch=True, device=None):
    """Generate kx, ky, kz."""
    if use_torch:
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
    else:
        kx = np.linspace(-np.pi, np.pi, klen)
        ky = np.linspace(-np.pi, np.pi, nlines)
        kx, ky = np.meshgrid(kx, ky)
        return kx, ky

def build_kspace(image_shape, sampling_rate, device=None):
    """Construct the k-space trajectory."""
    ndims = len(image_shape)
    if ndims == 2:
        kr = int(image_shape[0] * sampling_rate)
        kc = int(image_shape[1] * sampling_rate)
        grid_size = (kr, kc)
        kx, ky = gen_ktraj(kr, kc, device=device)
        kz = None
    if ndims == 3:
        kr = int(image_shape[0] * sampling_rate)
        kc = int(image_shape[1] * sampling_rate)
        kd = int(image_shape[2] * sampling_rate)
        grid_size = (kr, kc, kd)
        kx, ky, kz = gen_ktraj(kr, kc, kd, device=device)
    return kx, ky, kz, grid_size

def apply_rotation(angles, kx, ky, kz=None, ndims=None, masks=None):
    """Apply rotation to k-space trajectory."""
    if ndims == 2:
        kx_new = torch.zeros_like(kx, device=device)
        ky_new = torch.zeros_like(ky, device=device)
        kz_new = None
        for i in range(len(masks)):
            ang = torch.deg2rad(angles[i])
            kyi = torch.cos(ang)*ky - torch.sin(ang)*kx
            kxi = torch.sin(ang)*ky + torch.cos(ang)*kx
            kx_new[masks[i],...] = kxi[masks[i],...]
            ky_new[masks[i],...] = kyi[masks[i],...]
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
            kx_new[masks[i],...] = kxi[masks[i],...]
            ky_new[masks[i],...] = kyi[masks[i],...]
            kz_new[masks[i],...] = kzi[masks[i],...]
    return kx_new, ky_new, kz_new

def apply_rotation2(angles, kx, ky, kz=None, ndims=None, mask=None):
    """Apply rotation to k-space trajectory."""
    if ndims == 2:
        kx_new = torch.zeros_like(kx, device=device)
        ky_new = torch.zeros_like(ky, device=device)
        #kx_new = kx.clone().detach()
        #ky_new = ky.clone().detach()
        kz_new = None
        for i in range(kx.shape[0]):
            ang = torch.deg2rad(angles[i])
            kyi = torch.cos(ang)*ky - torch.sin(ang)*kx
            kxi = torch.sin(ang)*ky + torch.cos(ang)*kx
            kx_new[i,...] = mask[i,...] * kxi[i,...] + (1.0-mask[i,...]) * kx[i,...]
            ky_new[i,...] = mask[i,...] * kyi[i,...] + (1.0-mask[i,...]) * ky[i,...]
    if ndims == 3:
        kx_new = torch.zeros_like(kx, device=device)
        ky_new = torch.zeros_like(ky, device=device)
        kz_new = torch.zeros_like(kz, device=device)
        for i in range(kx.shape[0]):
            ang = torch.deg2rad(angles[i])
            ax, ay, az = ang[0], ang[1], ang[2]
            cax, sax = torch.cos(ax), torch.sin(ax)
            cay, say = torch.cos(ay), torch.sin(ay)
            caz, saz = torch.cos(az), torch.sin(az)

            kyi = (cax*cay)*ky + (cax*say*saz - sax*caz)*kx + (cax*say*caz + sax*saz)*kz
            kxi = (sax*cay)*ky + (sax*say*saz + cax*caz)*kx + (sax*say*caz - cax*saz)*kz
            kzi = (-say)*ky + (cay*saz)*kx + (cay*caz)*kz
            kx_new[i,...] = mask[i,...] * kxi[i,...] + (1.0-mask[i,...]) * kx[i,...]
            ky_new[i,...] = mask[i,...] * kyi[i,...] + (1.0-mask[i,...]) * ky[i,...]
            kz_new[i,...] = mask[i,...] * kzi[i,...] + (1.0-mask[i,...]) * kz[i,...]
    return kx_new, ky_new, kz_new

def apply_translation(ts, kdata, kx, ky, kz=None, grid_size=None, ndims=None, masks=None):
    """Apply translation as phase shift to k-space."""
    kdata = torch.reshape(kdata, grid_size)
    kdata_new = torch.zeros_like(kdata, device=device)
    for i in range(len(masks)):
        t = ts[i]
        if ndims == 2:
            kdata_i = utils.translate_opt(kdata, torch.stack((ky.flatten(), kx.flatten())), t, device=device)
        if ndims == 3:
            kdata_i = utils.translate_opt(kdata, torch.stack((ky.flatten(), kx.flatten(), kz.flatten())), t, device=device)
        kdata_new.real[masks[i],...] = kdata_i.real[masks[i],...]
        kdata_new.imag[masks[i],...] = kdata_i.imag[masks[i],...]
    kdata = kdata_new.flatten().unsqueeze(0).unsqueeze(0)
    return kdata

def apply_translation2(ts, kdata, kx, ky, kz=None, grid_size=None, ndims=None, mask=None):
    """Apply translation as phase shift to k-space."""
    kdata = torch.reshape(kdata, grid_size)
    kdata_new = torch.zeros_like(kdata, device=device)
    for i in range(grid_size[0]):
        t = ts[i]
        if ndims == 2:
            kdata_i = utils.translate_opt(kdata, torch.stack((ky.flatten(), kx.flatten())), t, device=device)
        if ndims == 3:
            kdata_i = utils.translate_opt(kdata, torch.stack((ky.flatten(), kx.flatten(), kz.flatten())), t, device=device)
        kdata_new.real[i,...] = mask[i,...] * kdata_i.real[i,...] + (1.0-mask[i,...]) * kdata.real[i,...]
        kdata_new.imag[i,...] = mask[i,...] * kdata_i.imag[i,...] + (1.0-mask[i,...]) * kdata.imag[i,...]
    return kdata_new.flatten().unsqueeze(0).unsqueeze(0)

def build_nufft(image, im_size, grid_size, numpoints):
    """Init NUFFT objects."""
    nufft_ob = tkbn.KbNufft(
        im_size=im_size,
        grid_size=grid_size,
        numpoints=numpoints,
        ).to(complex_dtype).to(device)
    adjnufft_ob = tkbn.KbNufftAdjoint(
        im_size=im_size,
        grid_size=grid_size,
        numpoints=numpoints,
        ).to(image).to(device)
    return nufft_ob, adjnufft_ob

def gen_movement(image, ndims, im_size,
                 kx, ky, kz=None, grid_size=None,
                 n_movements=None, locs=None, angles_std=None, trans_std=None,
                 debug=False):

    # Sample affines
    affines, angles, ts = sample_movements(n_movements, ndims, angles_std, trans_std)

    # Generate k-space masks
    masks = gen_masks(n_movements, locs, grid_size)
    if False:
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
    kx_new, ky_new, kz_new = apply_rotation(angles, kx, ky, kz, ndims, masks)

    # Fix k-space centre
    if False:
        mid = kx_new.shape[0]//2
        b = int(kx_new.shape[0] * 3/100.0)
        kx_new[mid-b:mid+b,:] = kx[mid-b:mid+b,:]
        ky_new[mid-b:mid+b,:] = ky[mid-b:mid+b,:]

    # Plot ktraj
    if debug:
        visualisation.plot_ktraj(kx_new, ky_new, kz_new)
        visualisation.plot_ktraj_image(kx_new, ky_new, kz_new)

    # create NUFFT objects, use 'ortho' for orthogonal FFTs
    nufft_ob, adjnufft_ob = build_nufft(image, im_size, grid_size, numpoints)

    # Calculate k-space data
    if ndims == 2:
        kdata = nufft_ob(image, torch.stack((ky.flatten(), kx.flatten()))).to(device)
    if ndims == 3:
        kdata = nufft_ob(image, torch.stack((ky.flatten(), kx.flatten(), kz.flatten()))).to(device)

    # Apply translational component
    kdata = apply_translation(ts, kdata, kx_new, ky_new, kz_new, grid_size, ndims, masks)

    # Plot the k-space data on log-scale
    if debug:
        kdata_numpy = np.reshape(kdata.detach().cpu().numpy(), grid_size)
        visualisation.plot_kdata(kdata_numpy, ndims)

    # Adjnufft back
    if ndims == 2:
        image_out = adjnufft_ob(kdata, torch.stack((ky_new.flatten(), kx_new.flatten())))
    if ndims == 3:
        image_out = adjnufft_ob(kdata, torch.stack((ky_new.flatten(), kx_new.flatten(), kz_new.flatten())))

    # Output image
    image_out = utils.normalise_image(image_out).to(dtype)
    return image_out, kdata, kx_new, ky_new, kz_new

def gen_movement2(image, ndims, im_size,
                 kx, ky, kz=None, grid_size=None,
                 log_probs=None, angles_std=None, trans_std=None,
                 debug=False):

    # create NUFFT objects, use 'ortho' for orthogonal FFTs
    nufft_ob, adjnufft_ob = build_nufft(image, im_size, grid_size, numpoints)

    # Calculate k-space data
    if ndims == 2:
        kdata = nufft_ob(image, torch.stack((ky.flatten(), kx.flatten()))).to(device)
    if ndims == 3:
        kdata = nufft_ob(image, torch.stack((ky.flatten(), kx.flatten(), kz.flatten()))).to(device)

    # Sample mask
    mask, n = gen_mask_opt(log_probs, grid_size, ndims)

    if debug:
        mask_np = mask.detach().cpu().numpy()
        if ndims == 2:
            fig = plt.figure()
            plt.imshow(mask_np, vmin=0, vmax=1)
        if ndims == 3:
            fig, axs = plt.subplots(1,3)
            visualisation.show_3d(mask_np, axs, vmin=0, vmax=1)

    # Sample rotations
    if ndims == 2:
        angles = angles_std**2 * torch.randn((grid_size[0],), dtype=dtype, device=device)
    if ndims == 3:
        angles = angles_std**2 * torch.randn((grid_size[0],ndims), dtype=dtype, device=device)

    # Sample translations
    ts = trans_std**2 * torch.randn((grid_size[0],ndims), dtype=dtype, device=device)

    # Apply rotation component
    kx_new, ky_new, kz_new = apply_rotation2(angles, kx, ky, kz, ndims, mask)

    # Plot ktraj
    if debug:
        visualisation.plot_ktraj(kx_new, ky_new, kz_new)
        visualisation.plot_ktraj_image(kx_new, ky_new, kz_new)

    # Apply translational component
    kdata = apply_translation2(ts, kdata, kx_new, ky_new, kz_new, grid_size, ndims, mask)

    # Plot the k-space data on log-scale
    if debug:
        kdata_numpy = np.reshape(kdata.detach().cpu().numpy(), grid_size)
        visualisation.plot_kdata(kdata_numpy, ndims)

    # Adjnufft back
    if ndims == 2:
        image_out = adjnufft_ob(kdata, torch.stack((ky_new.flatten(), kx_new.flatten())))
    if ndims == 3:
        image_out = adjnufft_ob(kdata, torch.stack((ky_new.flatten(), kx_new.flatten(), kz_new.flatten())))

    # Output image
    image_out = utils.normalise_image(image_out).to(dtype)
    return image_out, kdata, kx_new, ky_new, kz_new, n

def gen_movement_opt(image, ndims,
                     ts, angles,
                     kdata, kx, ky, kz=None,
                     grid_size=None, adjnufft_ob=None,
                     masks=None):

    # Apply rotation component
    kx_new, ky_new, kz_new = apply_rotation(angles, kx, ky, kz, ndims, masks)

    # Apply translational component
    kdata = apply_translation(ts, kdata, kx_new, ky_new, kz_new, grid_size, ndims, masks)

    # Adjnufft back
    if ndims == 2:
        image_out = adjnufft_ob(kdata, torch.stack((ky_new.flatten(), kx_new.flatten())))
    if ndims == 3:
        image_out = adjnufft_ob(kdata, torch.stack((ky_new.flatten(), kx_new.flatten(), kz_new.flatten())))

    # Output image
    image_out = utils.normalise_image(image_out).to(dtype)
    return image_out, kdata, kx_new, ky_new, kz_new

def gen_movement_opt2(image, ndims,
                     log_probs, trans_std, angles_std,
                     kdata, kx, ky, kz=None,
                     grid_size=None, adjnufft_ob=None):

    # Sample mask
    mask_opt, n_opt = gen_mask_opt(log_probs, grid_size, ndims)

    # Sample rotations
    if ndims == 2:
        angles = angles_std**2 * torch.randn((grid_size[0],), dtype=dtype, device=device)
    if ndims == 3:
        angles = angles_std**2 * torch.randn((grid_size[0],ndims), dtype=dtype, device=device)

    # Sample translations
    ts = trans_std**2 * torch.randn((grid_size[0],ndims), dtype=dtype, device=device)

    # Apply rotation component
    kx_new, ky_new, kz_new = apply_rotation2(angles, kx, ky, kz, ndims, mask_opt)

    # Apply translational component
    kdata = apply_translation2(ts, kdata, kx_new, ky_new, kz_new, grid_size, ndims, mask_opt)

    # Adjnufft back
    if ndims == 2:
        image_out = adjnufft_ob(kdata, torch.stack((ky_new.flatten(), kx_new.flatten())))
    if ndims == 3:
        image_out = adjnufft_ob(kdata, torch.stack((ky_new.flatten(), kx_new.flatten(), kz_new.flatten())))

    # Output image
    image_out = utils.normalise_image(image_out).to(dtype)
    return image_out, kdata, kx_new, ky_new, kz_new, mask_opt, n_opt


if __name__ == '__main__':

    # Load image
    #image = shepp_logan_phantom().astype(np.complex)
    image = utils.load_png('./data/sample_2d.png').astype(np.complex)
    #image = utils.load_nii_image('./data/sample_3d.nii.gz')
    #image = zoom(image, 0.5).astype(np.complex)
    ndims = len(image.shape)
    im_size = image.shape

    # Visualise
    if debug:
        if ndims == 2:
            fig = plt.figure()
            plt.imshow(np.abs(image), cmap='gray')
        if ndims == 3:
            fig, axs = plt.subplots(1,3)
            visualisation.show_3d(np.abs(image), axs, vmin=0, vmax=1)
        plt.suptitle('Input image')
        plt.tight_layout()

    # Convert image to tensor and unsqueeze coil and batch dimension
    image = torch.tensor(image).to(complex_dtype).unsqueeze(0).unsqueeze(0).to(device)
    image.requires_grad = True

    # Create a k-space trajectory
    sampling_rate = 1.0
    kx_init, ky_init, kz_init, grid_size = build_kspace(im_size, sampling_rate, device=device)

    # Movements
    n_movements = np.minimum(50, grid_size[0])
    prob = n_movements/grid_size[0]
    probs = torch.tensor([prob, 1.0-prob], dtype=dtype, device=device)
    log_probs = torch.log(probs + 1e-12)
    angles_std_val = 3.0
    trans_std_val = 10.0
    print('n_movements:', n_movements)

    # Generate movement
    if mode == 1:
        locs, _ = torch.sort(torch.randperm(grid_size[0])[:n_movements])
        image_out, kdata_out, kx_out, ky_out, kz_out = gen_movement(image, ndims, im_size,
                                                                kx_init, ky_init, kz_init, grid_size=grid_size,
                                                                n_movements=n_movements, locs=locs,
                                                                angles_std=angles_std_val, trans_std=trans_std_val,
                                                                debug=True)
    if mode == 2:
        image_out, kdata_out, kx_out, ky_out, kz_out, n_out = gen_movement2(image, ndims, im_size,
                                                                kx_init, ky_init, kz_init, grid_size=grid_size,
                                                                log_probs=log_probs, angles_std=angles_std_val, trans_std=trans_std_val,
                                                                debug=True)

    # Show the images
    image_np = np.abs(np.squeeze(image.detach().cpu().numpy()))
    image_out_np = np.abs(np.squeeze(image_out.detach().cpu().numpy()))
    image_out_np = utils.normalise_image(image_out_np, use_torch=False)
    diff = np.abs(image_np - image_out_np)
    err = diff.sum() / diff.size

    if debug:
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
            visualisation.show_3d(image_out_np, axs)
            plt.suptitle('Output image')
            plt.tight_layout()

            fig, axs = plt.subplots(1,3)
            visualisation.show_3d(diff, axs, cmap='jet')
            plt.suptitle('Diff image')
            plt.tight_layout()
        plt.show()

    # Target images
    image_target = image_out.clone().detach().to(dtype)
    image_target = utils.normalise_image(image_target, use_torch=True)
    image_target.requires_grad = False

    kdata_target = kdata_out.detach().clone()
    kdata_target.requires_grad = False

    kx_target = kx_out.clone().detach()
    ky_target = ky_out.clone().detach()
    kx_target.requires_grad = False
    ky_target.requires_grad = False
    if ndims == 3:
        kz_target = kz_out.clone().detach()
        kz_target.requires_grad = False

    # Starting image
    image.requires_grad = True

    # Visualise
    image_target_np = image_target.squeeze().detach().cpu().numpy()
    if debug:
        if ndims == 2:
            fig = plt.figure()
            plt.imshow(image_target_np, cmap='gray')
        if ndims == 3:
            fig, axs = plt.subplots(1,3)
            visualisation.show_3d(image_target_np, axs)
        plt.suptitle('target')
        plt.show()

    # Init k-space trajectory
    if ndims == 2:
        kx = kx_init.clone().detach()
        ky = ky_init.clone().detach()
        kz = None
        kx.requires_grad = True
        ky.requires_grad = True
        korig = torch.stack((ky.flatten(), kx.flatten()))
    if ndims == 3:
        kx = kx_init.clone().detach()
        ky = ky_init.clone().detach()
        kz = kz_init.clone().detach()
        kx.requires_grad = True
        ky.requires_grad = True
        kz.requires_grad = True
        korig = torch.stack((ky.flatten(), kx.flatten(), kz.flatten()))

    # Init NUFFT objects
    nufft_ob, adjnufft_ob = build_nufft(image, im_size, grid_size, numpoints)
    kdata_orig = nufft_ob(image, korig).to(device)
    kdata = kdata_orig.clone().detach()
    kdata.requires_grad = True

    # Init masks
    if mode == 1:
        masks = gen_masks(n_movements, locs, grid_size)

    # Init optimization
    print('Optimizing...')

    # Movement prob
    prob = 0.5
    probs = torch.tensor([prob, 1.0-prob], dtype=dtype, device=device)
    log_probs_init = torch.log(probs + 1e-12)
    log_probs = log_probs_init.clone().detach()
    log_probs.requires_grad = True

    target_prob = n_movements/grid_size[0]
    target_probs = torch.tensor([target_prob, 1.0-target_prob], dtype=dtype, device=device)
    log_probs_target = torch.log(probs + 1e-12)
    log_probs_target.requires_grad = False

    # Init translation
    if mode == 1:
        trans_init = 0.0*torch.randn(n_movements+1, ndims, dtype=dtype, device=device)
        trans = trans_init.clone().detach()
        trans.requires_grad = True

    if mode == 2:
        trans_std_init = torch.tensor([1.0], dtype=dtype, device=device)
        trans_std = trans_std_init.clone().detach()
        trans_std.requires_grad = True
        trans_std_target = torch.tensor([trans_std_val], dtype=dtype, device=device)
        trans_std_target.requires_grad = False

    # Init rotation
    if mode == 1:
        if ndims == 2:
            angles_init = torch.FloatTensor(n_movements+1,).uniform_(25.0, 25.0).to(device)
            angles_init[0] = 45.0
        if ndims == 3:
            angles_init = torch.FloatTensor(n_movements+1,ndims).uniform_(0.0, 0.0).to(device)
            angles_init[0,0] = 45.0
        angles = angles_init.clone().detach()
        angles.requires_grad = True

    if mode == 2:
        angles_std_init = torch.tensor([1.0], dtype=dtype, device=device)
        angles_std = angles_std_init.clone().detach()
        angles_std.requires_grad = True
        angles_std_target = torch.tensor([angles_std_val], dtype=dtype, device=device)
        angles_std_target.requires_grad = False

    # Init optimizer
    if mode == 1:
        optimizer = torch.optim.Adam([trans, angles], lr=1.)
    if mode == 2:
        optimizer = torch.optim.Adam([log_probs, angles_std, trans_std], lr=0.1)

    # Init loss functions
    l1_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()

    # Init animation
    if animate:
        fig = plt.figure()
        plt.tight_layout()
        ims = []
        if ndims == 2:
            visualisation.animate_2d(ims, image_np, image_target_np, losses=None)
        if ndims == 3:
            visualisation.animate_3d(ims, image_np, image_target_np, losses=None)

    # Optimize...
    n_iter = 100
    losses = []
    for i in range(n_iter):
        if mode == 2:
            print('log_probs:', log_probs, 'log_probs_target:', log_probs_target)
            print('angles_std:', angles_std.item(), 'angles_std_target:', angles_std_target.item(), 'grad:', angles_std.grad)
            print('trans_std:', trans_std.item(), 'trans_std_target', trans_std_target.item(), 'grad:', trans_std.grad)

        optimizer.zero_grad()
        if mode == 1:
            image_out, kdata_out, kx_out, ky_out, kz_out = gen_movement_opt(image, ndims,
                                                                            trans, angles,
                                                                            kdata, kx, ky, kz,
                                                                            grid_size, adjnufft_ob,
                                                                            masks)
        if mode == 2:
            image_out, kdata_out, kx_out, ky_out, kz_out, mask_out, n_out = gen_movement_opt2(image, ndims,
                                                                                              log_probs, trans_std, angles_std,
                                                                                              kdata, kx, ky, kz,
                                                                                              grid_size, adjnufft_ob)
            if debug:
                fig, axs = plt.subplots(1,3)
                image_out_np = image_out.squeeze().detach().cpu().numpy()
                mask_np = mask_out.detach().cpu().numpy()
                axs[0].imshow(mask_np, vmin=0, vmax=1)
                axs[0].set_title('iter: %d, num lines: %d, proportion: %.2f' % (i, n_out, n_out/grid_size[0]))
                axs[1].imshow(image_out_np, cmap='gray')
                axs[1].set_title('image')
                axs[2].imshow(image_target_np, cmap='gray')
                axs[2].set_title('target')
                plt.show()

        # Loss functions
        loss1 = 200. * l1_loss(image_out, image_target)
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
        if mode == 2:
            loss_prob = mse_loss(log_probs, log_probs_target)
            loss_angles = mse_loss(angles_std, angles_std_target)
            loss_trans = mse_loss(trans_std, trans_std_target)
            loss += loss_prob + loss_angles + loss_trans

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if animate:
            image_out_np = image_out.squeeze().detach().cpu().numpy()
            if ndims == 2:
                visualisation.animate_2d(ims, image_out_np, None, losses)
            if ndims == 3:
                visualisation.animate_3d(ims, image_out_np, None, losses)

    if animate:
        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True)
        from matplotlib import rcParams
        rcParams['animation.convert_path'] = r'/usr/bin/convert'
        ani.save('out.gif', writer='imagemagick', fps=15)
        print('Saved out.gif')
    else:
        image_out_np = image_out.squeeze().detach().cpu().numpy()
        if ndims == 2:
            fig, axs = plt.subplots(1,3)
            axs[0].imshow(image_out_np, cmap='gray')
            axs[0].set_title('output')
            axs[1].imshow(image_target_np, cmap='gray')
            axs[1].set_title('target')
            axs[2].plot(losses)
            axs[2].set_title('loss')
            axs[2].set_xlabel('iterations')
        if ndims == 3:
            fig, axs = plt.subplots(1,3)
            visualisation.show_3d(image_out_np, axs, vmin=0, vmax=1)
            plt.suptitle('output')
        plt.show()
