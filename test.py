from matplotlib.pyplot import axis
import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import segmentation_models_pytorch as smp
from utils import create_test_arg_parser, kill_border, process_test_args, AverageMeter, tensor2im, write_csv, recompone_overlap
from datasets import CustomDataSet, CustomTestDatasetWithPatch
from augmentation import val_aug
from models.student import Student
from models.expert import Expert
import scipy.spatial
import surface_distance
from constants import *
from torch.utils.data import DataLoader

def getDSC(testImage, resultImage):
    """Compute the Dice Similarity Coefficient."""
    testArray = testImage.flatten()
    resultArray = resultImage.flatten()

    return 1.0 - scipy.spatial.distance.dice(testArray, resultArray)

def getJaccard(testImage, resultImage):
    """Compute the Dice Similarity Coefficient."""
    testArray = testImage.flatten()
    resultArray = resultImage.flatten()

    return 1.0 - scipy.spatial.distance.jaccard(testArray, resultArray)

def getHD_ASSD(seg_preds, seg_labels):
    label_seg = np.array(seg_labels, dtype=bool)
    predict = np.array(seg_preds, dtype=bool)
    # predict = morphology.skeletonize(predict)
    # label_seg = morphology.skeletonize(label_seg)
    surface_distances = surface_distance.compute_surface_distances(
        label_seg, predict, spacing_mm=(1, 1))

    HD = surface_distance.compute_robust_hausdorff(surface_distances, 95)

    distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
    distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
    surfel_areas_gt = surface_distances["surfel_areas_gt"]
    surfel_areas_pred = surface_distances["surfel_areas_pred"]

    ASSD = (np.sum(distances_pred_to_gt * surfel_areas_pred) + np.sum(distances_gt_to_pred * surfel_areas_gt))/(np.sum(surfel_areas_gt)+np.sum(surfel_areas_pred))

    return HD, ASSD

def eval(model, device, test_loader, image_save_path):
    model.eval()
    dice_loss = smp.losses.DiceLoss(mode='binary', from_logits=False)
    # dices = AverageMeter("Dice", ".4f")
    names = []
    dices = []
    jacs = []
    HDs = []
    ASSDs = []
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            liot_data = data[:,2:,:,:]
            o_data = data[:,0,:,:].unsqueeze(1)
            data = torch.cat((o_data, liot_data), 1)
            # data = o_data
            data, target = data.to(device), target.to(device)
            output,_ = model(data)
            pred = torch.round(output).squeeze().detach().cpu().numpy()
            target = target.squeeze().detach().cpu().numpy()
            cv2.imwrite(os.path.join(image_save_path, 'pred_' + str(i) + '.png'), np.expand_dims(pred, -1)*255.)
            cv2.imwrite(os.path.join(image_save_path, 'gt_' + str(i) + '.png'), np.expand_dims(target, -1)*255.)
            dice = getDSC(target, pred)
            jac = getJaccard(target, pred)
            HD, ASSD = getHD_ASSD(pred, target)
            dices.append(dice)
            jacs.append(jac)
            HDs.append(HD)
            ASSDs.append(ASSD)
            names.append(str(i))

    return names, dices, jacs, HDs, ASSDs


class CustomTestWithPatch():
    def __init__(self, args, path=None, transform=None, domain=None):
        self.dataset = CustomTestDatasetWithPatch(args, path, transform, domain)
        self.test_loader = DataLoader(self.dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)

    def inference(self, model, device):
        model.eval()
        preds = []
        with torch.no_grad():
            for i, (data, target) in enumerate(self.test_loader):
                liot_data = data[:,2:,:,:]
                o_data = data[:,0,:,:].unsqueeze(1)
                data = torch.cat((o_data, liot_data), 1)
                # data = o_data
                data, target = data.to(device), target.to(device)
                output,_ = model(data)
                pred = torch.round(output).squeeze().detach().cpu().numpy()
                preds.append(np.expand_dims(pred, 0))
        predictions = np.concatenate(preds, axis=0)
        self.pred_patches = np.expand_dims(predictions,axis=1)

    def eval(self, image_save_path):
        self.pred_imgs = recompone_overlap(
            self.pred_patches, self.dataset.new_height, self.dataset.new_width, 64, 64)
        ## restore to original dimensions
        self.pred_imgs = self.pred_imgs[:, :, 0:self.dataset.img_height, 0:self.dataset.img_width]
        kill_border(self.pred_imgs, self.dataset.test_FOVs)
        names = []
        dices = []
        jacs = []
        HDs = []
        ASSDs = []
        for i in range(self.pred_imgs.shape[0]):
            pred = self.pred_imgs[i][0]
            gt = self.dataset.test_masks[i][0]
            fov = self.dataset.test_FOVs[i][0]
            dice = getDSC(gt, pred)
            jac = getJaccard(gt, pred)
            HD, ASSD = getHD_ASSD(pred, gt)
            dices.append(dice)
            jacs.append(jac)
            HDs.append(HD)
            ASSDs.append(ASSD)
            names.append(str(i))
            cv2.imwrite(os.path.join(image_save_path, 'pred_' + str(i) + '.png'), np.expand_dims(pred, -1)*255.)
            cv2.imwrite(os.path.join(image_save_path, 'gt_' + str(i) + '.png'), np.expand_dims(gt, -1)*255.)
            cv2.imwrite(os.path.join(image_save_path, 'fov_' + str(i) + '.png'), np.expand_dims(fov, -1)*255.)
        #predictions only inside the FOV
        # y_scores, y_true = pred_only_in_FOV(self.pred_imgs, self.dataset.test_masks, self.dataset.test_FOVs)
        return names, dices, jacs, HDs, ASSDs    

    
class CustomTestWithoutPatch():
    def __init__(self, args, path=None, transform=None, domain=None):
        self.dataset = CustomDataSet(args, path, transform, domain)
        self.test_loader = DataLoader(self.dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)

    def eval(self, model, device, image_save_path):
        names = []
        dices = []
        jacs = []
        HDs = []
        ASSDs = []
        with torch.no_grad():
            for i, (data, target) in enumerate(self.test_loader):
                liot_data = data[:,2:,:,:]
                o_data = data[:,0,:,:].unsqueeze(1)
                data = torch.cat((o_data, liot_data), 1)
                # data = o_data
                data, target = data.to(device), target.to(device)
                output,_ = model(data)
                pred = torch.round(output).squeeze().detach().cpu().numpy()
                target = target.squeeze().detach().cpu().numpy()
                cv2.imwrite(os.path.join(image_save_path, 'pred_' + str(i) + '.png'), np.expand_dims(pred, -1)*255.)
                cv2.imwrite(os.path.join(image_save_path, 'gt_' + str(i) + '.png'), np.expand_dims(target, -1)*255.)
                dice = getDSC(target, pred)
                jac = getJaccard(target, pred)
                HD, ASSD = getHD_ASSD(pred, target)
                dices.append(dice)
                jacs.append(jac)
                HDs.append(HD)
                ASSDs.append(ASSD)
                names.append(str(i))
                cv2.imwrite(os.path.join(image_save_path, 'pred_' + str(i) + '.png'), np.expand_dims(pred, -1)*255.)
                cv2.imwrite(os.path.join(image_save_path, 'gt_' + str(i) + '.png'), np.expand_dims(target, -1)*255.)
        return names, dices, jacs, HDs, ASSDs    


def main():
    args = create_test_arg_parser().parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_no

    process_test_args(args)
    print(args)
    device = torch.device("cuda:{}".format(args.cuda_no[0]) if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        student_model = nn.DataParallel(Student(args)).to(device)
        expert_similar_model = nn.DataParallel(Expert(args)).to(device)
        expert_dissimilar_model = nn.DataParallel(Expert(args)).to(device)
    else:
        student_model = Student(args).to(device)
        expert_similar_model = Expert(args).to(device)
        expert_dissimilar_model = Expert(args).to(device)

    student_model.load_state_dict(torch.load(args.model_file))
        
    # target_testloader = create_test_dataloader(args, val_aug(), 'test')

    # for i, (domain, testloader) in enumerate(target_testloader.items()):

    #     image_save_path = os.path.join(args.save_path, domain, 'images')
    #     if not os.path.exists(image_save_path):
    #         os.makedirs(image_save_path)
    #     names, dices, jacs, HDs, ASSDs = eval(student_model, device, testloader, image_save_path)
    #     write_csv(args.save_path, domain, names, dices, jacs, HDs, ASSDs)

    for i in range(len(args.domains)):
        domain = args.domains[i]
        image_save_path = os.path.join(args.save_path, domain, 'images')
        if not os.path.exists(image_save_path):
            os.makedirs(image_save_path)
        if domain in FOV_VESSELS:
            tester = CustomTestWithPatch(args, os.path.join(args.root, args.domains[i], 'test'), transform=val_aug(), domain=domain)
            tester.inference(student_model, device)
            names, dices, jacs, HDs, ASSDs = tester.eval(image_save_path)
            write_csv(args.save_path, domain, names, dices, jacs, HDs, ASSDs)
        else:
            tester = CustomTestWithoutPatch(args, os.path.join(args.root, args.domains[i], 'test'), transform=val_aug(), domain=domain)
            names, dices, jacs, HDs, ASSDs = tester.eval(student_model, device, image_save_path)
            write_csv(args.save_path, domain, names, dices, jacs, HDs, ASSDs)



if __name__ == "__main__":
    main()
