import cv2
import numpy as np
import random
import os
import itertools
from PIL import Image, ImageOps
import time


def LIOT(img, dirs, mask):
    o_h, o_w = img.shape
    img = np.pad(img, 8, 'edge')
    cnt = 0
    for dir in dirs:
        if cnt == 0:
            new_img = np.expand_dims(create_one_dimension(img, o_h, o_w, dir, mask), axis=-1)
        else:
            new_img = np.concatenate((new_img, np.expand_dims(create_one_dimension(img, o_h, o_w, dir, mask), axis=-1)), axis=-1)
        cnt += 1
    return new_img


def create_one_dimension(img, o_h, o_w, dir, mask):
    new_img = np.empty((o_h, o_w), dtype=int)
    for i in range(o_h):
        for j in range(o_w):
            pix_val = 0
            multiple = 1
            try:
                for k in range(1, 9):
                    if img[i+8,j+8] > img[i+8+k*dir[0],j+8+k*dir[1]]:
                        pix_val += multiple
                    multiple *= 2
            except:
                pass
            new_img[i,j] = pix_val
    return new_img

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP','.tif'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


def trans():
    o_img_dir = '/data/fundus_vessels/DOMAIN_NAME/test/img/'
    o_gt_dir = '/data/fundus_vessels/DOMAIN_NAME/test/gt/'
    t_img_dir = '/data/fundus_vessels/NEW_DOMAIN_NAME/test/img/'
    t_gt_dir = '/data/fundus_vessels/NEW_DOMAIN_NAME/test/gt/'

    os.makedirs(t_img_dir, exist_ok=True)
    os.makedirs(t_gt_dir, exist_ok=True)
    imgs = os.listdir(o_img_dir)
    gts = os.listdir(o_gt_dir)

    for i in range(len(imgs)):
        img = cv2.imread(os.path.join(o_img_dir, imgs[i]))
        # img = Image.fromarray(img)
        # img = ImageOps.invert(img)
        # img = np.asarray(img)
        img = img[:,:,1]
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        cv2.imwrite(os.path.join(t_img_dir, imgs[i]), np.expand_dims(img, -1))
        gt = cv2.imread(os.path.join(o_gt_dir, gts[i]), 0)
        cv2.imwrite(os.path.join(t_gt_dir, gts[i]), np.expand_dims(gt, -1))      
        

# trans()
image_paths = ['/data/fundus_vessels/DOMAIN_NAME/test/img/']
label_paths = ['/data/fundus_vessels/DOMAIN_NAME/test/gt/']
save_image_paths = ['/data/fundus_vessels/NEW_DOMAIN_NAME/test/img/']
save_label_paths = ['/data/fundus_vessels/NEW_DOMAIN_NAME/test/gt/']

for i in range(1):
    os.makedirs(save_image_paths[i], exist_ok=True)
    os.makedirs(save_label_paths[i], exist_ok=True)

dirs = [(0,1), (1,0), (-1,0), (0,-1)]
[(0,-1),(-1,0),(1,0),(0,1)]
all_dirs = list(itertools.permutations([(0,1), (1,0), (-1,0), (0,-1)]))

for i in range(1):
    images = make_dataset(image_paths[i])
    labels = make_dataset(label_paths[i])
    images = sorted(images)
    labels = sorted(labels)
    for idx in range(len(images)):
        print(images[idx])
        image = cv2.imread(images[idx], 0)
        label = cv2.imread(labels[idx], 0)
        cnt = 0
        for dir in all_dirs:
            save_name = images[idx][-11:-4] + '_' + str(cnt)
            cnt += 1
            new_image = LIOT(image, dir, None)
            np.save(save_image_paths[i] + save_name, new_image)
            np.save(save_label_paths[i] + save_name, label)

