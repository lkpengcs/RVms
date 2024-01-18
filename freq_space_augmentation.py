import torch
import numpy as np
import random
import cv2


def extract_amp_spectrum(trg_img):

    fft_trg_np = torch.fft.fft2( trg_img, dim=(-2, -1) )
    amp_target, pha_trg = torch.abs(fft_trg_np), torch.angle(fft_trg_np)

    return amp_target

def amp_spectrum_swap( amp_local, amp_target, L=0.1 , ratio=0):
    
    a_local = torch.fft.fftshift( amp_local, dim=(-2, -1) )
    a_trg = torch.fft.fftshift( amp_target, dim=(-2, -1) )

    _, h, w = a_local.shape
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)

    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1

    a_local[:,h1:h2,w1:w2] = a_local[:,h1:h2,w1:w2] * ratio + a_trg[:,h1:h2,w1:w2] * (1- ratio)
    a_local = torch.fft.ifftshift( a_local, dim=(-2, -1) )
    return a_local

def freq_space_interpolation( local_img, amp_target, L=0 , ratio=0):
    
    local_img_np = local_img 

    # get fft of local sample
    fft_local_np = torch.fft.fft2( local_img_np, dim=(-2, -1) )

    # extract amplitude and phase of local sample
    amp_local, pha_local = torch.abs(fft_local_np), torch.angle(fft_local_np)

    # swap the amplitude part of local image with target amplitude spectrum
    amp_local_ = amp_spectrum_swap( amp_local, amp_target, L=L , ratio=ratio)

    # get transformed image via inverse fft
    fft_local_ = amp_local_ * torch.exp( 1j * pha_local )
    local_in_trg = torch.fft.ifft2( fft_local_, dim=(-2, -1) )
    local_in_trg = torch.real(local_in_trg)

    return local_in_trg


def freq_aug(im_local, im_trg):
    batch_size = im_local.size(0)
    L = 0.2
    transferred_imgs = []
    for i in range(batch_size):
        ratio = random.choice([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        amp_target = extract_amp_spectrum(im_trg[i,...]*255.)
        local_in_trg = freq_space_interpolation(im_local[i,...], amp_target, L=0.2, ratio=ratio)
        transferred_imgs.append(local_in_trg/255.)
    transferred_img = transferred_imgs[0].unsqueeze(0)
    for i in range(1, batch_size):
        transferred_img = torch.cat((transferred_img, transferred_imgs[i].unsqueeze(0)), 0)
    return transferred_img
