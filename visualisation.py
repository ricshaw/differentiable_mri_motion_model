import numpy as np
import matplotlib.pyplot as plt

def show_3d(image, axs, cmap='gray', vmin=None, vmax=None):
    axs[0].imshow(image[...,int(image.shape[2]//2)], cmap=cmap, vmin=vmin, vmax=vmax)
    axs[1].imshow(image[:,int(image.shape[1]//2),:], cmap=cmap, vmin=vmin, vmax=vmax)
    axs[2].imshow(image[int(image.shape[0]//2),...], cmap=cmap, vmin=vmin, vmax=vmax)

def animate_2d(ims, image1=None, image2=None, losses=None):
    h = []
    if image1 is not None:
        ax1 = plt.subplot(1,3,1)
        plt.title('image')
        plt.axis('off')
        im1 = plt.imshow(image1, cmap='gray', animated=True)
        h += [im1]
    if image2 is not None:
        ax2 = plt.subplot(1,3,2)
        plt.title('target')
        plt.axis('off')
        im2 = plt.imshow(image2, cmap='gray', animated=True)
        h += [im2]
    else:
        h += [ims[0][1]]
    if losses is not None:
        ax3 = plt.subplot(1,3,3)
        plt.title('loss')
        plt.xlabel('iterations')
        im3, = plt.plot(losses, 'b-')
        plt.subplots_adjust(wspace=0.25)
        h += [im3]
    ims.append(h)

def animate_3d(ims, image1=None, image2=None, losses=None):
    h = []
    if image1 is not None:
        ax1 = plt.subplot(2,4,1)
        plt.axis('off')
        im1 = plt.imshow(image1[...,int(image1.shape[2]//2)], cmap='gray')
        ax2 = plt.subplot(2,4,2)
        plt.title('image')
        plt.axis('off')
        im2 = plt.imshow(image1[:,int(image1.shape[2]//2),:], cmap='gray')
        ax3 = plt.subplot(2,4,3)
        plt.axis('off')
        im3 = plt.imshow(image1[int(image1.shape[2]//2),...], cmap='gray')
        h += [im1,im2,im3]
    if image2 is not None:
        ax4 = plt.subplot(2,4,5)
        plt.axis('off')
        im4 = plt.imshow(image2[...,int(image2.shape[2]//2)], cmap='gray')
        ax5 = plt.subplot(2,4,6)
        plt.title('target')
        plt.axis('off')
        im5 = plt.imshow(image2[:,int(image2.shape[2]//2),:], cmap='gray')
        ax6 = plt.subplot(2,4,7)
        plt.axis('off')
        im6 = plt.imshow(image2[int(image2.shape[2]//2),...], cmap='gray')
        h += [im4,im5,im6]
    else:
        h += [ims[0][3],ims[0][4],ims[0][5]]
    if losses is not None:
        ax7 = plt.subplot(2,4,4)
        plt.title('loss')
        plt.xlabel('iterations')
        im7, = plt.plot(losses, 'b-')
        plt.subplots_adjust(wspace=0.4)
        h += [im7]
    ims.append(h)

def plot_kdata(kdata, ndims=2):
    if ndims == 2:
        plt.figure()
        plt.imshow(np.log10(np.abs(kdata)), cmap='gray')
        plt.tight_layout()
        plt.title('k-space data, log10 scale')
    if ndims == 3:
        fig, axs = plt.subplots(1,3)
        show_3d(np.log10(np.abs(kdata)), axs)
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





