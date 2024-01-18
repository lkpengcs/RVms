import argparse
import numpy as np
import torch
import random
from skimage.feature import local_binary_pattern
from skimage.filters import hessian
from scipy.special import comb
import torch
import pandas as pd
from PIL import Image, ImageOps
import torch.nn as nn
import cv2
import math
import os
import joblib

def LIOT(img):
    '''
    This function is faster than LIOT_example.py;
    More efficient LIOT will be continuously updated;
    '''
    # img = np.asarray(img)#input image H*W*C
    gray_img= np.asarray(img)#convert to gray; if not retinal dataset, you can use standard grayscale api
    pad_img = np.pad(gray_img, ((8,8)), 'constant')
    Weight = pad_img.shape[0]
    Height = pad_img.shape[1]
    sum_map = np.zeros((gray_img.shape[0], gray_img.shape[1], 4)).astype(np.uint8)
    directon_map = np.zeros((gray_img.shape[0], gray_img.shape[1], 8)).astype(np.uint8)

    for direction in range(0,4):
        for postion in range(0,8):
            if direction == 0:#Right
                new_pad = pad_img[postion + 9: Weight - 7 + postion, 8:-8]  #   from low to high
                #new_pad = pad_img[16-postion: Weight - postion, 8:-8]  	#   from high to low
            elif direction==1:#Left
                #new_pad = pad_img[7 - postion:-1 * (9 + postion), 8:-8]  	#from low to high
                new_pad = pad_img[postion:-1 * (16 - postion), 8:-8]  	  	#from high to low
            elif direction==2:#Up
                new_pad = pad_img[8:-8, postion + 9:Height - 7 + postion]  	# from low to high
                #new_pad = pad_img[8:-8, 16 - postion: Height - postion]   	#from high to low
            elif direction==3:#Down
                #new_pad = pad_img[8:-8, 7 - postion:-1 * (9 + postion)]  	# from low to high
                new_pad = pad_img[8:-8, postion:-1 * (16 - postion)]  		#from high to low

            tmp_map = gray_img.astype(np.int64) - new_pad.astype(np.int64)
            tmp_map[tmp_map > 0] = 1
            tmp_map[tmp_map <= 0] = 0
            directon_map[:,:,postion] = tmp_map * math.pow( 2, postion)
        sum_direction = np.sum(directon_map,2)
        sum_map[:,:,direction] = sum_direction

    per = np.random.permutation(sum_map.shape[-1])
    sum_map = sum_map[:,:,per]
    return sum_map


def create_hessian(img):
    hessian_img = hessian(img, sigmas=(0.1, 0.2, 0.05), black_ridges=False)
    return hessian_img * 255.


def lbp(img):
    lbp_img = local_binary_pattern(img, 8, 3.0, 'uniform')
    lbp_img = lbp_img * 255 / 9
    return lbp_img


def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.
       Control points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000
        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])
    
    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals


def bezier_curve_aug(x, flip=False, prob=0.5):
    if random.random() >= prob:
        if flip == False:
            return x
        else:
            x = x * 255.
            x = x.astype(np.uint8)
            x = Image.fromarray(x)
            x = ImageOps.invert(x)
            x = np.asarray(x)
            x = x / 255.
    points = [[0, 0], [random.random(), random.random()], [random.random(), random.random()], [1, 1]]
    xpoints = [p[0] for p in points]
    ypoints = [p[1] for p in points]
    xvals, yvals = bezier_curve(points, nTimes=100000)
    if flip == True:
        xvals = np.sort(xvals)
    else:
        xvals, yvals = np.sort(xvals), np.sort(yvals)
    nonlinear_x = np.interp(x, xvals, yvals)
    return nonlinear_x


def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def create_train_arg_parser():
    
    parser = argparse.ArgumentParser(description="train setup")
    parser.add_argument("--root", type=str, help="path to root directory")
    parser.add_argument("--domains", type=str, default='OCTA,OCT,DRIVE', help="name of all domains, separate by comma")
    parser.add_argument("--phase", type=str, default='train', help="train or test")
    parser.add_argument("--num_epochs", type=int, default=500, help="number of epochs")
    parser.add_argument("--image_size", type=int, default=512, help="the size of image")
    parser.add_argument("--student_epochs", type=int, default=500, help="# of epochs to train student")
    parser.add_argument("--expert_epochs", type=int, default=500, help="# of epochs to train expert")
    parser.add_argument("--cuda_no", type=str, default='0', help="cuda number")
    parser.add_argument("--use_pretrained", type=bool, default=False, help="load pretrained checkpoint")
    parser.add_argument("--pretrained_model_path", type=str, default=None, help="if use_pretrained is true, provide checkpoint")
    parser.add_argument("--LR_seg", default=1e-4, type=float, help='learning rate')
    parser.add_argument("--save_path", type=str, help="Model save path")
    parser.add_argument("--classnum", type=int, default=1, help="clf class number")
    parser.add_argument("--input_nc", type=int, default=1, help="input channels")
    parser.add_argument("--batch_size", type=int, default=64, help="train batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="# of threads for data loading")
    parser.add_argument("--transfer_type", type=str, default='wct2', help="wct2|freq")
    parser.add_argument("--net_D", type=str, default='n_layers', help="n_layers|basic|pixel")  
    parser.add_argument("--aug_types", type=str, default=None, help="special rules for data augmentation, e.g. A/B/C,D/E for two domains")
    parser.add_argument("--resume_path", type=str, default=None, help="resume_training")
    return parser


def create_test_arg_parser():

    parser = argparse.ArgumentParser(description="train setup for segmentation")
    parser.add_argument("--root", type=str, help="path to root directory")
    parser.add_argument("--domains", type=str, help="name of all domains, separate by comma")
    parser.add_argument("--image_size", type=int, default=512, help="the size of image")
    parser.add_argument("--model_file", type=str, help="model_file")
    parser.add_argument("--save_path", type=str, help="results save path")
    parser.add_argument("--cuda_no", type=str, default='0', help="cuda number")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--classnum", type=int, default=1, help="clf class number")
    parser.add_argument("--input_nc", type=int, default=1, help="input channels")
    parser.add_argument("--num_workers", type=int, default=8, help="# of threads for data loading")
    return parser


def save_args(args,save_path):
    if not os.path.exists(save_path):
        os.makedirs('%s' % save_path)

    print('Config info -----')
    for arg in vars(args):
        print('%s: %s' % (arg, getattr(args, arg)))
    with open('%s/args.txt' % save_path, 'w') as f:
        for arg in vars(args):
            print('%s: %s' % (arg, getattr(args, arg)), file=f)
    joblib.dump(args, '%s/args.pkl' % save_path)
    print('\033[0;33m================config infomation has been saved=================\033[0m')


def process_train_args(args):
    str_cudas = args.cuda_no.split(',')
    args.cuda_no = []
    for id in range(len(str_cudas)):
        args.cuda_no.append(id)

    str_domains = args.domains.split(',')
    args.domains = []
    for domain in str_domains:
        args.domains.append(domain)

    save_args(args,args.save_path)


def process_test_args(args):
    str_cudas = args.cuda_no.split(',')
    args.cuda_no = []
    for id in range(len(str_cudas)):
        args.cuda_no.append(id)

    str_domains = args.domains.split(',')
    args.domains = []
    for domain in str_domains:
        args.domains.append(domain)
