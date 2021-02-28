import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import cv2
import nibabel as nib
import torch
import torch.nn as nn
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
animate = True
dtype = torch.float32
complex_dtype = torch.complex64
numpoints = 6
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device)
matplotlib.use("Agg") if animate else None


def sample_movements(n_movements, ndims):
    """Sample movement affine transforms."""
    affines = []
    if ndims == 2:
        angles = torch.FloatTensor(n_movements+1,).uniform_(-15.0, 15.0).to(device)
        trans = torch.FloatTensor(n_movements+1,2).uniform_(-10.0, 10.0).to(device)
    if ndims == 3:
        angles = torch.FloatTensor(n_movements+1,ndims).uniform_(-15.0, 15.0).to(device)
        trans = torch.FloatTensor(n_movements+1,ndims).uniform_(-10.0, 10.0).to(device)
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

def gen_ktraj(nlines, klen, kdepth=None, use_torch=True):
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

def build_kspace(image_shape, sampling_rate):
    """Construct the k-space trajectory."""
    ndims = len(image_shape)
    if ndims == 2:
        kr = int(image_shape[0] * sampling_rate)
        kc = int(image_shape[1] * sampling_rate)
        grid_size = (kr, kc)
        kx, ky = gen_ktraj(kr, kc)
        kz = None
    if ndims == 3:
        kr = int(image_shape[0] * sampling_rate)
        kc = int(image_shape[1] * sampling_rate)
        kd = int(image_shape[2] * sampling_rate)
        grid_size = (kr, kc, kd)
        kx, ky, kz = gen_ktraj(kr, kc, kd)
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

def gen_movement(image, kx, ky, kz=None, grid_size=None, n_movements=None, locs=None, debug=False):

    # Convert image to tensor and unsqueeze coil and batch dimension
    ndims = len(image.shape)
    im_size = image.shape
    image = torch.tensor(image).to(complex_dtype).unsqueeze(0).unsqueeze(0).to(device)
    print('image shape: {}'.format(image.shape))

    # Sample affines
    affines, angles, ts = sample_movements(n_movements, ndims)

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


if __name__ == '__main__':

    # Load image
    #image = shepp_logan_phantom().astype(np.complex)
    image = utils.load_png('./data/sample_2d.png').astype(np.complex)
    #image = utils.load_nii_image('./data/sample_3d.nii.gz')
    #image = zoom(image, 0.5).astype(np.complex)
    ndims = len(image.shape)

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

    # Create a k-space trajectory
    sampling_rate = 1.0
    kx_init, ky_init, kz_init, grid_size = build_kspace(image.shape, sampling_rate)

    # Generate movement
    n_movements = 10
    locs = sorted(np.random.choice(kx_init.shape[0], n_movements))
    image_out, kdata_out, kx_out, ky_out, kz_out = gen_movement(image,
                                                                kx_init, ky_init, kz_init,
                                                                grid_size=grid_size,
                                                                n_movements=n_movements,
                                                                locs=locs,
                                                                debug=True)

    # Show the images
    image_out_np = np.abs(np.squeeze(image_out.detach().cpu().numpy()))
    image_out_np = utils.normalise_image(image_out_np, use_torch=False)
    diff = np.abs(image - image_out_np)
    err = diff.sum() / diff.size

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
    target = image_out.clone().to(dtype)
    target = utils.normalise_image(target, use_torch=True)
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

    # Starting image
    image = torch.tensor(image).to(dtype)
    image = utils.normalise_image(image.squeeze(), use_torch=True)
    image.requires_grad = True
    print('target', target.dtype, target.shape, target.min(), target.max())
    print('input', image.dtype, image.shape, image.min(), image.max())
    im_size = image.shape
    image_tensor = image.to(complex_dtype).unsqueeze(0).unsqueeze(0).to(device)

    # Visualise
    image_np = image.detach().cpu().numpy()
    target_np = target.squeeze().detach().cpu().numpy()
    if ndims == 2:
        fig = plt.figure()
        plt.imshow(target_np, cmap='gray')
    if ndims == 3:
        fig, axs = plt.subplots(1,3)
        visualisation.show_3d(target_np, axs)
    plt.suptitle('target')
    plt.show()

    # Init k-space trajectory
    if ndims == 2:
        kx = kx_init.clone().detach()
        ky = ky_init.clone().detach()
        kz = None
        kx.requires_grad = False
        ky.requires_grad = False
        korig = torch.stack((ky.flatten(), kx.flatten()))
    if ndims == 3:
        kx = kx_init.clone().detach()
        ky = ky_init.clone().detach()
        kz = kz_init.clone().detach()
        kx.requires_grad = False
        ky.requires_grad = False
        kz.requires_grad = False
        korig = torch.stack((ky.flatten(), kx.flatten(), kz.flatten()))

    # Init NUFFT objects
    nufft_ob, adjnufft_ob = build_nufft(image_tensor, im_size, grid_size, numpoints)
    kdata_orig = nufft_ob(image_tensor, korig).to(device)
    kdata = kdata_orig.clone().detach()
    kdata.requires_grad = True

    # Init masks
    masks = gen_masks(n_movements, locs, grid_size)

    # Init optimization
    print('Optimizing...')

    # Init translation
    ts_init = 0.0*torch.randn(n_movements+1, ndims, dtype=dtype, device=device)
    ts_init[0,:] = 0.
    ts = ts_init.clone().detach()
    ts.requires_grad = True
    print(ts)

    # Init rotation
    if ndims == 2:
        angles_init = torch.FloatTensor(n_movements+1,).uniform_(-25.0, -25.0).to(device)
        angles_init[0] = 45.0
    if ndims == 3:
        angles_init = torch.FloatTensor(n_movements+1,ndims).uniform_(-0.0, 0.0).to(device)
        angles_init[0,0] = 45.0
    angles = angles_init.clone().detach()
    angles.requires_grad = True
    print(angles)

    # Init optimizer
    optimizer = torch.optim.Adam([ts, angles], lr=1.)

    # Init loss functions
    l1_loss = nn.L1Loss()

    # Init animation
    if animate:
        fig = plt.figure()
        plt.tight_layout()
        ims = []
        if ndims == 2:
            visualisation.animate_2d(ims, image_np, target_np, losses=None)
        if ndims == 3:
            visualisation.animate_3d(ims, image_np, target_np, losses=None)

    # Optimize...
    n_iter = 100
    losses = []
    for i in range(n_iter):
        optimizer.zero_grad()
        image_out, kdata_out, kx_out, ky_out, kz_out = gen_movement_opt(image_tensor, ndims,
                                                                        ts, angles,
                                                                        kdata, kx, ky, kz,
                                                                        grid_size, adjnufft_ob,
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
            axs[1].imshow(target_np, cmap='gray')
            axs[1].set_title('target')
            axs[2].plot(losses)
            axs[2].set_title('loss')
            axs[2].set_xlabel('iterations')
        if ndims == 3:
            fig, axs = plt.subplots(1,3)
            visualisation.show_3d(image_out_np, axs, vmin=0, vmax=1)
            plt.suptitle('output')
        plt.show()
