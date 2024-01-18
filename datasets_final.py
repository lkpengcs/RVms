from torch.utils.data import Dataset, ConcatDataset, DataLoader
import cv2
cv2.setNumThreads(1)
import numpy as np
import os
from PIL import Image, ImageOps
import random
from utils import LIOT, lbp, create_hessian, bezier_curve_aug


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.npy', '.tif']
BLACK_VESSELS = ['OCT', 'OCTAinvert', 'DRIVE_CLAHE', 'HRF_CLAHE', 'PRIME-FP_CLAHE', 'ROSEinvert', 'PRIME-FP_CLAHE', 'DRIVE_CLAHE']
WHITE_VESSELS = ['OCTA', 'OCTinvert', 'DRIVE_CLAHEinvert', 'HRF_CLAHEinvert', 'PRIME-FP_CLAHEinvert', 'ROSE', 'PRIME-FP_CLAHEinvert', 'DRIVE_CLAHEinvert']
AUG_TYPES = ['bezier','bezier','bezier','bezier','bezier','bezier','bezier','bezier','bezier','bezier','bezier','bezier']

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


class CustomDataSet(Dataset):
    def __init__(self, args, image_path=None, label_path=None, liot_image_path=None, transform=None, aug_type=None, invert=False, domain=None):
        self.args = args
        self.transform = transform
        self.image_path = image_path
        self.label_path = label_path
        self.images = make_dataset(self.image_path)
        self.labels = make_dataset(self.label_path)
        self.images = sorted(self.images)
        self.labels = sorted(self.labels)
        self.aug_type = aug_type
        self.invert = False
        self.liot_image_path = liot_image_path
        self.domain = domain

    def __getitem__(self, idx):
        if self.args.input_nc == 3:
            image = cv2.imread(self.images[idx])
            o_image = cv2.imread(self.images[idx])
        elif self.args.input_nc == 1:
            image = cv2.imread(self.images[idx], 0)
            o_image = cv2.imread(self.images[idx], 0)
        else:
            image = cv2.imread(self.images[idx], 0)
            o_image = cv2.imread(self.images[idx], 0)
        
        label = cv2.imread(self.labels[idx], 0)
        label = np.where(label == 255, 1, label)

        if self.invert == True:
            o_image = Image.fromarray(o_image)
            o_image = ImageOps.invert(o_image)
            o_image = np.asarray(o_image)
        o_image = np.expand_dims(o_image, -1)
        
        random_num = random.randint(0, 23)

        liot_image = np.load(os.path.join(self.liot_image_path, self.images[idx][-11:-4]+'_'+str(random_num)+'.npy')).astype(np.uint8)

        if self.aug_type is not None:
            if 'lbp' in self.aug_type:
                image = lbp(image).astype(np.uint8)
            elif 'hessian' in self.aug_type:
                image = create_hessian(image).astype(np.uint8)
            elif 'bezier' in self.aug_type:
                image = bezier_curve_aug(image/255., self.invert) * 255.
                image = image.astype(np.uint8)
        image = np.expand_dims(image, -1)
        image = np.concatenate((o_image, image, liot_image),-1)
        transformed = self.transform(image=image, mask=label)
        images = transformed["image"]
        masks = transformed["mask"]

        return images, masks

    def __len__(self):
        return len(self.images)


def create_multi_train_data_loader(args, transforms, phase, similar):
    root_path = args.root
    img_paths = []
    gt_paths = []
    liot_img_paths = []
    aug_types = []
    all_domains = []
    if similar == True:
        source_domain = args.domains[0]
    else:
        source_domain = args.domains[1]
    
    for i in range(2, len(args.domains)):
        if (args.domains[i] in BLACK_VESSELS and source_domain in BLACK_VESSELS) or (args.domains[i] in WHITE_VESSELS and source_domain in WHITE_VESSELS):
            all_domains.append(args.domains[i])
    print(all_domains)
    for i in range(2, len(args.domains)):
        if args.domains[i] not in all_domains:
            continue
        img_paths.append(os.path.join(root_path, args.domains[i], phase, 'img'))
        gt_paths.append(os.path.join(root_path, args.domains[i], phase, 'gt'))
        liot_img_paths.append(os.path.join(root_path, args.domains[i]+'_LIOT', phase, 'img'))
        aug_types.append(AUG_TYPES[i])

    datasets = []
    for i in range(len(all_domains)):
        datasets.append(CustomDataSet(args, image_path=img_paths[i], label_path=gt_paths[i], liot_image_path=liot_img_paths[i], transform=transforms, aug_type=aug_types[i], invert=False, domain=all_domains[i]))

    dataloaders = {}
    for i in range(len(all_domains)):
        dataloader = DataLoader(dataset=datasets[i], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True) 
        dataloaders.update({all_domains[i]:dataloader})

    return dataloaders


def create_target_dataloader(args, transforms, phase):
    root_path = args.root
    img_paths = []
    gt_paths = []
    liot_img_paths = []
    for i in range(len(args.domains)):
        img_paths.append(os.path.join(root_path, args.domains[i], phase, 'img'))
        gt_paths.append(os.path.join(root_path, args.domains[i], phase, 'gt'))
        liot_img_paths.append(os.path.join(root_path, args.domains[i]+'_LIOT', phase, 'img'))
    datasets = []
    for i in range(len(img_paths)):
        datasets.append(CustomDataSet(args, image_path=img_paths[i], label_path=gt_paths[i], liot_image_path=liot_img_paths[i], transform=transforms, aug_type=AUG_TYPES[i], invert=False, domain=args.domains[i]))
    dataloaders = {}
    for i in range(len(datasets)):
        dataloader = DataLoader(dataset=datasets[i], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=True) 
        dataloaders.update({args.domains[i]:dataloader})
    return dataloaders

def create_test_dataloader(args, transforms, phase):
    root_path = args.root
    img_paths = []
    gt_paths = []
    liot_img_paths = []
    for i in range(len(args.domains)):
        img_paths.append(os.path.join(root_path, args.domains[i], phase, 'img'))
        gt_paths.append(os.path.join(root_path, args.domains[i], phase, 'gt'))
        liot_img_paths.append(os.path.join(root_path, args.domains[i]+'_LIOT', phase, 'img'))
    datasets = []
    for i in range(len(img_paths)):
        datasets.append(CustomDataSet(args, image_path=img_paths[i], label_path=gt_paths[i], liot_image_path=liot_img_paths[i], transform=transforms, aug_type=None, invert=False, domain=args.domains[i]))
    dataloaders = {}
    for i in range(len(datasets)):
        dataloader = DataLoader(dataset=datasets[i], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=True) 
        dataloaders.update({args.domains[i]:dataloader})
    return dataloaders

def create_single_data_loader(args, transform, phase, invert=False):
    if invert == False:
        domain_idx = 0
    else:
        domain_idx = 1
    img_path = os.path.join(args.root, args.domains[domain_idx], phase, 'img')
    gt_path = os.path.join(args.root, args.domains[domain_idx], phase, 'gt')
    liot_img_path = os.path.join(args.root, args.domains[domain_idx]+'_LIOT', phase, 'img')
    dataset = CustomDataSet(args, image_path=img_path, label_path=gt_path, liot_image_path=liot_img_path, transform=transform, aug_type=AUG_TYPES[domain_idx], invert=False, domain=args.domains[domain_idx])
    if phase == 'train':
        return DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True) 
    else:
        return DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True) 
