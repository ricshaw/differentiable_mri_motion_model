import numpy as np
import matplotlib.pyplot as plt

def show_3d(image, axs, cmap='gray', vmin=None, vmax=None):
    axs[0].imshow(image[...,int(image.shape[2]//2)], cmap=cmap, vmin=vmin, vmax=vmax)
    axs[1].imshow(image[:,int(image.shape[1]//2),:], cmap=cmap, vmin=vmin, vmax=vmax)
    axs[2].imshow(image[int(image.shape[0]//2),...], cmap=cmap, vmin=vmin, vmax=vmax)

def animate_2d(ims, image1=None, image2=None, losses=None):
    h = []
    if image1 is not None:
        plt.subplot(1,3,1)
        plt.title('image')
        plt.axis('off')
        im1 = plt.imshow(image1, cmap='gray', animated=True)
        h += [im1]
    if image2 is not None:
        plt.subplot(1,3,2)
        plt.title('target')
        plt.axis('off')
        im2 = plt.imshow(image2, cmap='gray', animated=True)
        h += [im2]
    else:
        h += [ims[0][1]]
    if losses is not None:
        plt.subplot(1,3,3)
        plt.title('loss')
        plt.xlabel('iterations')
        im3, = plt.plot(losses, 'b-')
        plt.subplots_adjust(wspace=0.25)
        h += [im3]
    ims.append(h)

def animate_3d(ims, image1=None, image2=None, losses=None):
    h = []
    if image1 is not None:
        plt.subplot(2,4,1)
        plt.axis('off')
        im1 = plt.imshow(image1[...,int(image1.shape[2]//2)], cmap='gray')
        plt.subplot(2,4,2)
        plt.title('image')
        plt.axis('off')
        im2 = plt.imshow(image1[:,int(image1.shape[2]//2),:], cmap='gray')
        plt.subplot(2,4,3)
        plt.axis('off')
        im3 = plt.imshow(image1[int(image1.shape[2]//2),...], cmap='gray')
        h += [im1,im2,im3]
    if image2 is not None:
        plt.subplot(2,4,5)
        plt.axis('off')
        im4 = plt.imshow(image2[...,int(image2.shape[2]//2)], cmap='gray')
        plt.subplot(2,4,6)
        plt.title('target')
        plt.axis('off')
        im5 = plt.imshow(image2[:,int(image2.shape[2]//2),:], cmap='gray')
        plt.subplot(2,4,7)
        plt.axis('off')
        im6 = plt.imshow(image2[int(image2.shape[2]//2),...], cmap='gray')
        h += [im4,im5,im6]
    else:
        h += [ims[0][3],ims[0][4],ims[0][5]]
    if losses is not None:
        plt.subplot(2,4,4)
        plt.title('loss')
        plt.xlabel('iterations')
        im7, = plt.plot(losses, 'b-')
        plt.subplots_adjust(wspace=0.4)
        h += [im7]
    ims.append(h)


