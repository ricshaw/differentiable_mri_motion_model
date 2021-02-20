import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchkbnufft as tkbn
from skimage.data import shepp_logan_phantom
from scipy.linalg import logm, expm
from piq import ssim, SSIMLoss, MultiScaleSSIMLoss, VSILoss

from pytorch3d.transforms.so3 import (
    so3_exponential_map,
    so3_relative_angle,
)

dtype = torch.complex64
#device = 'cpu'
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

def plot_kdata(kdata):
    plt.figure()
    plt.imshow(np.log10(np.abs(kdata)), cmap='gray')
    plt.tight_layout()
    plt.title('k-space data, log10 scale')

def plot_ktraj(kx, ky):
    kx_np = kx.detach().cpu().numpy()
    ky_np = ky.detach().cpu().numpy()
    plt.figure()
    plt.plot(kx_np[:,:].T, -ky_np[:,:].T)
    plt.axis('equal')
    plt.title('k-space trajectory')
    plt.tight_layout()

def plot_ktraj_image(kx, ky):
    kx_np = kx.detach().cpu().numpy()
    ky_np = ky.detach().cpu().numpy()
    fig, axs = plt.subplots(1,2)
    axs[0].imshow(kx_np)
    axs[0].set_title('kx')
    axs[1].imshow(ky_np)
    axs[1].set_title('ky')
    plt.suptitle('k-space trajectory')
    plt.tight_layout()
    plt.show()

def rotation_matrix(ang):
    #ang = torch.deg2rad(ang)
    ang = ang * 3.14159 / 180.0
    return torch.tensor([[torch.cos(ang), -torch.sin(ang)],
                         [torch.sin(ang), torch.cos(ang)]], device=device)

def rotation_matrix_np(ang):
    ang = np.deg2rad(ang)
    return np.array([[np.cos(ang), -np.sin(ang)],
                     [np.sin(ang), np.cos(ang)]])

def rotation_matrix_3D(angles):
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
    #phase = t[0,0] * ktraj[0,...] + t[0,1] * ktraj[1,...]

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

def sample_movements(n_movements):
    affines = []
    angles = torch.FloatTensor(n_movements+1,).uniform_(-15.0, 15.0).to(device)
    trans = torch.FloatTensor(n_movements+1,2).uniform_(-10.0, 10.0).to(device)
    #affines.append(torch.eye(3))
    for i in range(n_movements+1):
        ang = angles[i]
        t = trans[i,:]
        A = torch.eye(3).to(device)
        R = rotation_matrix(ang).to(device)
        A[:2,:2] = R
        A[:2,2] = t.to(device)
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

def gen_masks(n_movements, locs, nlines, klen):
    masks = []
    if n_movements > 0:
        #lines = torch.randperm(nlines)[:n_movements]
        #lines, _ = torch.sort(lines)
        mask = torch.zeros((nlines, klen), device=device)
        mask[0:locs[0],:] = 1
        masks.append(mask)
        for i in range(1,n_movements):
            mask = torch.zeros((nlines, klen), device=device)
            mask[locs[i-1]:locs[i],:] = 1
            masks.append(mask)
        mask = torch.zeros((nlines, klen), device=device)
        mask[locs[-1]::,:] = 1
        masks.append(mask)
    else:
        masks.append(torch.ones((nlines, klen), device=device))
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

def gen_ktraj(nlines, klen):
    kx = torch.linspace(-np.pi, np.pi, klen)
    ky = torch.linspace(-np.pi, np.pi, nlines)
    kx, ky = torch.meshgrid(kx, ky)
    kx = kx.T
    ky = ky.T
    return kx.to(device), ky.to(device)

def gen_ktraj_np(nlines, klen):
    kx = np.linspace(-np.pi, np.pi, klen)
    ky = np.linspace(-np.pi, np.pi, nlines)
    kx, ky = np.meshgrid(kx, ky)
    return kx, ky

def to_1d(kx, ky):
    return torch.stack((ky.flatten(), kx.flatten()))

def to_1d_np(kx, ky):
    return np.stack((ky.flatten(), kx.flatten()))

def to_2d(ktraj, nlines, klen):
    kx = torch.reshape(ktraj[1,...], (nlines, klen))
    ky = torch.reshape(ktraj[0,...], (nlines, klen))
    return kx, ky

def to_2d_np(ktraj, nlines, klen):
    kx = ktraj[1,...]
    ky = ktraj[0,...]
    kx = np.reshape(kx, (nlines, klen))
    ky = np.reshape(ky, (nlines, klen))
    return kx, ky

def gen_movement(image, n_movements, locs, debug=False):

    # Convert image to tensor and unsqueeze coil and batch dimension
    im_size = image.shape
    image = torch.tensor(image).to(dtype).unsqueeze(0).unsqueeze(0).to(device)
    print('image shape: {}'.format(image.shape))

    # Create a k-space trajectory
    sampling_rate = 1.5
    klen = int(image.shape[-1] * sampling_rate)
    nlines = int(image.shape[-2] * sampling_rate)
    grid_size = (nlines, klen)
    #nlines = nlines // 4

    kx, ky = gen_ktraj(nlines, klen)
    kx = kx.to(device)
    ky = ky.to(device)
    #kx_np, ky_np = gen_ktraj_np(nlines, klen)

    ktraj = to_1d(kx, ky).to(device)
    #ktraj = to_1d_np(kx, ky)
    korig = ktraj.clone()
    #korig = ktraj.copy()
    #korig = torch.tensor(korig).to(torch.float)

    affines = sample_movements(n_movements)
    #affines = sample_movements_log(n_movements)
    #affines = sample_movements_np(n_movements)

    # Combine affines
    #affines = combine_affines(affines)

    # Generate k-space masks
    masks = gen_masks(n_movements, locs, nlines, klen)
    #masks = gen_masks_np(n_movements, nlines, klen)
    if debug:
        fig = plt.figure()
        n_plots = np.minimum(10,len(masks))
        for i in range(n_plots):
            ax = fig.add_subplot(1,n_plots,i+1)
            plt.imshow(masks[i].detach().cpu().numpy())

    # Apply rotation component
    ktrajs = []
    kx_new = torch.zeros_like(kx, device=device)
    ky_new = torch.zeros_like(ky, device=device)
    #kx_new = np.zeros_like(kx)
    #ky_new = np.zeros_like(ky)
    for i in range(len(affines)):
        R = affines[i][:2,:2].to(device)
        ktraji = rotate(ktraj, R)
        #ktraji = rotate_np(ktraj, R)
        ktrajs.append(ktraji)
        kxi, kyi = to_2d(ktraji, nlines, klen)
        #kxi, kyi = to_2d_np(ktraji, nlines, klen)
        kx_new += masks[i] * kxi
        ky_new += masks[i] * kyi

    mid = kx_new.shape[0]//2
    b = int(kx_new.shape[0] * 3/100.0)
    b = 0
    print('k-space centre:', b)
    kx_new[mid-b:mid+b,:] = kx[mid-b:mid+b,:]
    ky_new[mid-b:mid+b,:] = ky[mid-b:mid+b,:]

    ktraj = to_1d(kx_new, ky_new)
    #ktraj = to_1d_np(kx_new, ky_new)
    #kx, ky = to_2d(ktraj, nlines, klen)
    #kx, ky = to_2d_np(ktraj, nlines, klen)

    # Plot ktraj
    if debug:
        plot_ktraj(kx_new, ky_new)
        plot_ktraj_image(kx_new, ky_new)

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

    #print(nufft_ob)
    #print(adjnufft_ob)

    # Calculate k-space data
    kdata = nufft_ob(image, korig).to(device)
    print('kdata', kdata.shape)

    # Apply translational component
    print('Applying translational component')
    kdata = torch.reshape(kdata, (nlines, klen))
    #kdata = np.reshape(kdata, (nlines, klen))
    kdata_new = torch.zeros_like(kdata)
    #kdata_new = np.zeros_like(kdata)
    for i in range(len(masks)):
        t = affines[i][:2,2]
        print(i,t)
        #kdata_i = translate(kdata, ktraj, t)
        kdata_i = translate_opt(kdata, ktraj, t)
        #kdata_i = translate_np(kdata, ktraj, t)
        kdata_new += masks[i] * kdata_i
    kdata_new[mid-b:mid+b,:] = kdata[mid-b:mid+b,:]
    kdata = kdata_new.flatten().unsqueeze(0).unsqueeze(0)

    # Plot the k-space data on log-scale
    if debug:
        kdata_numpy = np.reshape(kdata.detach().cpu().numpy(), (nlines, klen))
        plot_kdata(kdata_numpy)

    # adjnufft back
    image_out = adjnufft_ob(kdata, ktraj)

    return image_out, kdata, kx_new, ky_new


def gen_movement_opt(image, n_movements, locs, ts, angles, kdata, korig, kx, ky, klen, nlines, im_size, adjnufft_ob, masks):

    # Convert image to tensor and unsqueeze coil and batch dimension
    #im_size = image.shape
    #image = image.to(dtype).unsqueeze(0).unsqueeze(0).to(device)
    #print('image shape: {}'.format(image.shape), image.dtype)

    # Create a k-space trajectory
    #sampling_rate = 1.5
    #klen = int(image.shape[-1] * sampling_rate)
    #nlines = int(image.shape[-2] * sampling_rate)
    #grid_size = (nlines, klen)

    #kx, ky = gen_ktraj(nlines, klen)
    #ktraj = to_1d(kx, ky).to(device)
    #korig = ktraj.clone()

    #mid = kx.shape[0]//2
    #b = int(kx.shape[0] * 3/100.0)
    #b = 0

    # Generate k-space masks
    #masks = gen_masks(n_movements, locs, nlines, klen)

    # Apply rotation component
    kx_new = torch.zeros_like(kx, device=device)
    ky_new = torch.zeros_like(ky, device=device)
    for i in range(len(angles)):
        kyi = torch.cos(angles[i] * 3.14159/180.0)*ky - torch.sin(angles[i] * 3.14159/180.0)*kx
        kxi = torch.sin(angles[i] * 3.14159/180.0)*ky + torch.cos(angles[i] * 3.14159/180.0)*kx
        #R = rotation_matrix(angles[i])
        #ktraji = rotate(ktraj, R)
        #ktraji = rotate(torch.stack((ky.flatten(), kx.flatten())), R)
        #kxi, kyi = to_2d(ktraji, nlines, klen)
        kx_new += masks[i] * kxi
        ky_new += masks[i] * kyi

    #ktraj = torch.stack((ky_new.flatten(), kx_new.flatten()))
    #ktraj = to_1d(kx_new, ky_new)
    #ktraj = torch.tensor(ktraj).to(torch.float32).to(device)
    #print('ktraj shape: {}'.format(ktraj.shape), ktraj.dtype)

    # create NUFFT objects, use 'ortho' for orthogonal FFTs
    #nufft_ob = tkbn.KbNufft(
    #    im_size=im_size,
    #    grid_size=grid_size,
    #    ).to(dtype).to(device)
    #adjnufft_ob = tkbn.KbNufftAdjoint(
    #    im_size=im_size,
    #    grid_size=grid_size,
    #    ).to(image).to(device)

    #print(nufft_ob)
    #print(adjnufft_ob)

    # Calculate k-space data
    #kdata = nufft_ob(image, korig).to(device)
    #print('kdata', kdata.shape, kdata.dtype)

    # Apply translational component
    if True:
        print('Applying translational component')
        kdata = torch.reshape(kdata, (nlines, klen))
        kdata_new = torch.zeros_like(kdata, device=device)
        for i in range(len(masks)):
            t = ts[i]
            print(i,t)
            #kdata_i = translate_opt(kdata, ktraj, t)
            kdata_i = translate_opt(kdata, torch.stack((ky_new.flatten(), kx_new.flatten())), t)
            #kdata_i = translate_opt(kdata, torch.stack((ky.flatten(), kx.flatten())), t)
            kdata_new += masks[i] * kdata_i
        #kdata_new[mid-b:mid+b,:] = kdata[mid-b:mid+b,:]
        kdata = kdata_new.flatten().unsqueeze(0).unsqueeze(0)

    # adjnufft back
    #image_out = adjnufft_ob(kdata, ktraj)
    image_out = adjnufft_ob(kdata, torch.stack((ky_new.flatten(), kx_new.flatten())))
    #image_out = adjnufft_ob(kdata, torch.stack((ky.flatten(), kx.flatten())))
    image_out = torch.abs(image_out.squeeze())
    image_out = (image_out - image_out.min()) / (image_out.max() - image_out.min())
    return image_out, kdata, kx_new, ky_new


if __name__ == '__main__':

    # Create a simple shepp logan phantom and plot it
    image = shepp_logan_phantom().astype(np.complex)
    plt.imshow(np.abs(image), cmap='gray', vmin=0, vmax=1)
    plt.title('Input image')
    plt.tight_layout()


    # Generate movement
    sampling_rate = 1.5
    nlines = int(image.shape[0] * sampling_rate)
    n_movements = 4
    locs = sorted(np.random.choice(nlines, n_movements))
    image_out, kdata_out, kx_out, ky_out = gen_movement(image, n_movements, locs, debug=True)


    # Show the images
    image_out_np = np.abs(np.squeeze(image_out.detach().cpu().numpy()))
    image_out_np -= image_out_np.min()
    image_out_np /= image_out_np.max()
    plt.figure()
    plt.imshow(image_out_np, cmap='gray', vmin=0, vmax=1)
    plt.title('Output image')
    plt.tight_layout()

    diff = np.abs(image - image_out_np)
    err = diff.sum() / image_out_np.size
    print('err', err)
    plt.figure()
    plt.imshow(diff, cmap='jet', vmin=0, vmax=1)
    plt.title('Diff image')
    plt.tight_layout()
    plt.show()



    target = image_out.clone().to(float)
    target = torch.abs(target.squeeze())
    target = (target - target.min()) / (target.max() - target.min())
    target.requires_grad = False

    kdata_target = kdata_out.clone()
    kdata_target.requires_grad = False

    kx_target = kx_out.clone()
    ky_target = ky_out.clone()
    kx_target.requires_grad = False
    ky_target.requires_grad = False

    image = torch.tensor(image).to(float)
    image = torch.abs(image.squeeze())
    image = (image - image.min()) / (image.max() - image.min())
    image.requires_grad = True
    print('target', target.dtype, target.shape, target.min(), target.max())
    print('input', image.dtype, image.shape, image.min(), image.max())

    fig, axs = plt.subplots(1,2)
    axs[0].imshow(image.detach().cpu().numpy(), cmap='gray')
    axs[0].set_title('start')
    axs[1].imshow(target.detach().cpu().numpy(), cmap='gray')
    axs[1].set_title('target')
    plt.show()

     # Convert image to tensor and unsqueeze coil and batch dimension
    im_size = image.shape
    image_tensor = image.to(dtype).unsqueeze(0).unsqueeze(0).to(device)

    # Create a k-space trajectory
    sampling_rate = 1.5
    klen = int(image.shape[1] * sampling_rate)
    nlines = int(image.shape[0] * sampling_rate)
    grid_size = (nlines, klen)

    kx_init, ky_init = gen_ktraj(nlines, klen)
    kx = kx_init.clone().detach()
    ky = ky_init.clone().detach()
    kx.requires_grad = False
    ky.requires_grad = False
    #ktraj = to_1d(kx, ky).to(device)
    #ktraj.requires_grad = True
    #korig = ktraj.clone().detach()
    korig = torch.stack((ky_init.flatten(), kx_init.flatten()))

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

    masks = gen_masks(n_movements, locs, nlines, klen)

    # Optimise
    print('Optimising...')

    ts_init = 0.0*torch.randn(n_movements+1, 2, dtype=torch.float32, device=device)
    ts_init[0,:] = 0.
    ts = ts_init.clone().detach()
    ts.requires_grad = True
    print(ts)

    #angles_init = 0.0*torch.randn(n_movements+1, 1, dtype=torch.float32, device=device)
    angles_init = torch.FloatTensor(n_movements+1,).uniform_(-25.0, -25.0).to(device)
    #angles_init[0,:] = 0.
    angles = angles_init.clone().detach()
    angles.requires_grad = True
    print(angles)

    #optimizer = torch.optim.Adam([ts], lr=1.)
    #optimizer = torch.optim.Adam([angles], lr=1.)
    optimizer = torch.optim.Adam([ts, angles], lr=1.)
    #optimizer = torch.optim.Adam([ts, kx, ky], lr=1.)

    #ssim_loss = SSIMLoss(data_range=1.)
    ms_ssim_loss = MultiScaleSSIMLoss()
    vsi_loss = VSILoss()
    l1_loss = nn.L1Loss()

    fig, axs = plt.subplots(1,3)
    n_iter = 1000
    losses = []
    for i in range(n_iter):
        optimizer.zero_grad()

        #angles = angles - 2
        image_out, kdata_out, kx_new, ky_new = gen_movement_opt(image_tensor, n_movements, locs, ts, angles, kdata, korig, kx, ky, klen, nlines, im_size, adjnufft_ob, masks)

        loss1 = 200. * l1_loss(image_out, target)
        loss2 = 50. * ms_ssim_loss(image_out, target)
        loss3 = 200. * vsi_loss(image_out, target)
        loss4 = l1_loss(kdata_out.real, kdata_target.real)
        loss5 = l1_loss(kdata_out.imag, kdata_target.imag)
        loss6 = 100. * l1_loss(kx_new, kx_target)
        loss7 = 100. * l1_loss(ky_new, ky_target)
        loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7
        print('iter:', i, 'losses:', loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item(), loss6.item(), loss7.item())
        print('ts:', ts.detach().cpu().numpy())
        print('angles:', angles.detach().cpu().numpy())
        loss.backward()
        print('grads', angles.grad, ts.grad, kx.grad, ky.grad)
        optimizer.step()

        losses.append(loss.item())

        if True:
            axs[0].clear()
            axs[0].imshow(image_out.detach().cpu().numpy(), cmap='gray')
            axs[0].set_title('iter: %d' % i)
            axs[1].clear()
            axs[1].imshow(target.detach().cpu().numpy(), cmap='gray')
            axs[1].set_title('target')
            axs[2].clear()
            axs[2].plot(losses)
            axs[2].set_title('loss')
            axs[2].set_xlabel('iterations')
            plt.pause(0.0001)

    axs[0].imshow(image_out.detach().cpu().numpy(), cmap='gray')
    axs[0].set_title('iter: %d' % i)
    axs[1].imshow(target.detach().cpu().numpy(), cmap='gray')
    axs[1].set_title('target')
    axs[2].plot(losses)
    axs[2].set_title('loss')
    axs[2].set_xlabel('iterations')
    plt.show()
