import torch
import torch.nn as nn
import cv2
cv2.setNumThreads(1)
import numpy as np
import os
from tqdm import tqdm
import segmentation_models_pytorch as smp
from tensorboardX import SummaryWriter
import random
from utils import create_train_arg_parser, process_train_args, AverageMeter, tensor2im, feature_align
from datasets import create_multi_train_data_loader, create_target_dataloader, create_single_data_loader
from augmentation import train_aug, val_aug
from models.student import Student
from models.expert import Expert
from WCT2.transfer import style_transfer
from adain.test_style_transfer import adain_transfer
from freq_space_augmentation import freq_aug
from models.cyclegan_networks import define_D, backward_D
import scipy.spatial
from constants import *


def pretrain_expert(args, source_dataloader, targets_dataloader, expert, optimizer, device, writer, similar=True):
    criterion = smp.losses.DiceLoss(mode='binary', from_logits=False)
    expert.train()
    if similar == True:
        expert_name = 'expert_similar_model.pth'
    else:
        expert_name = 'expert_dissimilar_model.pth'

    iter_targets = [0] * len(targets_dataloader)
    target_domains = [0] * len(targets_dataloader)
    for i, (domain, dataloader) in enumerate(targets_dataloader.items()):
        iter_targets[i] = iter(dataloader)
        target_domains[i] = domain
    minimum_losses = 10000000.
    for epoch in range(args.expert_epochs):

        source_losses = 0.
        aug_source_losses = 0.
        transferred_losses = 0.
        g_losses = 0.
        d_losses = 0.

        iter_source = iter(source_dataloader)
        source_domain = args.domains[0]
        for i in range(1, len(source_dataloader) + 1):
            data_source, label_source = iter_source.next()
            liot_source = data_source[:,2:,:,:]
            o_source = data_source[:,0,:,:].unsqueeze(1)
            data_source = data_source[:,1,:,:].unsqueeze(1)

            data_source = torch.cat((data_source, liot_source), 1)
            data_source = data_source.to(device)
            label_source = label_source.to(device)
            optimizer.zero_grad()
            o_source_outputs, _ = expert(torch.cat((o_source, liot_source), 1).to(device))
            o_source_loss = criterion(o_source_outputs.squeeze(), label_source)
            source_losses += o_source_loss.item()
            o_source_loss.backward()

            source_outputs, _ = expert(data_source)
            
            source_loss = criterion(source_outputs.squeeze(), label_source)
            aug_source_losses += source_loss.item()
            source_loss.backward()
            optimizer.step()
            
            for ix, it in enumerate(iter_targets):
                try:
                    data_target, _ = it.next()
                except StopIteration:
                    it = iter(targets_dataloader[target_domains[ix]])
                    data_target, _ = it.next()
                liot_target = data_target[:,2:,:,:]
                o_target = data_target[:,0,:,:].unsqueeze(1)
                data_target = data_target[:,1,:,:].unsqueeze(1)
                if args.transfer_type == 'wct2':
                    data_source_transferred = style_transfer(torch.cat((o_source, o_source, o_source), 1).to(device), torch.cat((o_target, o_target, o_target), 1).to(device), device)
                elif args.transfer_type == 'freq':
                    data_source_transferred = freq_aug(o_source, o_target).to(device)
                elif args.transfer_type == 'adain':
                    data_source_transferred = adain_transfer(torch.cat((o_source, o_source, o_source), 1).to(device), torch.cat((o_target, o_target, o_target), 1).to(device), device)
                data_source_transferred = torch.cat((data_source_transferred, liot_target.to(device)), 1)

                source_transferred_outputs, _ = expert(data_source_transferred)
                source_transferred_loss = criterion(source_transferred_outputs, label_source)
                transferred_losses += source_transferred_loss.item()

                optimizer.zero_grad()
                source_transferred_loss.backward()
                optimizer.step()
                
        source_losses /= len(source_dataloader)
        aug_source_losses /= len(source_dataloader)
        transferred_losses /= (len(source_dataloader)*len(iter_targets))
        g_losses /= (len(source_dataloader)*len(iter_targets))
        d_losses /= (len(source_dataloader)*len(iter_targets))

        if similar == True:
            writer.add_scalars("Similar Expert", {'Source Loss':source_losses, 'Aug Source Loss':aug_source_losses, 'Transferred Loss':transferred_losses}, epoch)
        else:
            writer.add_scalars("Dissimilar Expert", {'Source Loss':source_losses, 'Aug Source Loss':aug_source_losses, 'Transferred Loss':transferred_losses}, epoch)

        print('Epoch {}: Source Loss {:.4f}, Aug Source Loss {:.4f}, Transferred Loss {:.4f}, G Loss {:.4f}, D Loss {:.4f}'.format(epoch, source_losses, aug_source_losses, transferred_losses, g_losses, d_losses))
        if (source_losses + aug_source_losses + transferred_losses)/3 < minimum_losses:
            if torch.cuda.device_count() > 1:
                torch.save(expert.module.state_dict(), "{}/{}".format(args.model_save_path, expert_name))
            else:
                torch.save(expert.state_dict(), "{}/{}".format(args.model_save_path, expert_name))
            minimum_losses = (source_losses + aug_source_losses + transferred_losses)/3
        if epoch % 20 == 0:
            if torch.cuda.device_count() > 1:
                torch.save(expert.module.state_dict(), "{}/{}".format(args.model_save_path, expert_name[:-4]+'_'+str(epoch)+'.pth'))
            else:
                torch.save(expert.state_dict(), "{}/{}".format(args.model_save_path, expert_name[:-4]+'_'+str(epoch)+'.pth'))



def train_one_epoch(student_model, expert_models, optimizer_student, device, targets_dataloader, args):

    bce_loss = nn.BCELoss()
    dice_loss = smp.losses.DiceLoss(mode='binary', from_logits=False)

    student_kd_losses = np.zeros(len(targets_dataloader))

    iter_targets = [0] * len(targets_dataloader)
    target_domains = [0] * len(targets_dataloader)
    for i, (domain, dataloader) in enumerate(targets_dataloader.items()):
        iter_targets[i] = iter(dataloader)
        target_domains[i] = domain
    c = list(zip(iter_targets, target_domains))
    random.shuffle(c)
    iter_targets[:], target_domains[:] = zip(*c)

    source_domain = args.domains[0]

    for i in range(30):
        for ix, it in enumerate(iter_targets):
            try:
                data_target, label = it.next()
            except StopIteration:
                it = iter(targets_dataloader[target_domains[ix]])
                data_target, label = it.next()
            liot_target = data_target[:,2:,:,:]
            o_target = data_target[:,0,:,:].unsqueeze(1)
            data_target = data_target[:,1,:,:].unsqueeze(1)
            enhanced_target = torch.cat((data_target, liot_target), 1).to(device)
            data_target = torch.cat((o_target, liot_target), 1).to(device)

            if (target_domains[ix] in BLACK_VESSELS and source_domain in BLACK_VESSELS) or (target_domains[ix] in WHITE_VESSELS and source_domain in WHITE_VESSELS):
                # print('Similar Expert, {}'.format(target_domains[ix]))
                expert_target, expert_outputs = expert_models[0](data_target)
                expert_enhance, expert_enhance_outputs = expert_models[0](enhanced_target)
            else:
                # print('Dissimilar Expert, {}'.format(target_domains[ix]))
                expert_target, expert_outputs = expert_models[1](data_target)
                expert_enhance, expert_enhance_outputs = expert_models[1](enhanced_target)
            student_target, student_outputs = student_model(data_target)
            student_enhance, student_enhance_outputs = student_model(enhanced_target)

            expert_guided_target_loss = bce_loss(student_target, torch.round(expert_target).detach())
            expert_guided_target_enhance_loss = bce_loss(student_enhance, torch.round(expert_enhance).detach())

            optimizer_student.zero_grad()
            student_losses = expert_guided_target_loss + expert_guided_target_enhance_loss
            if target_domains[ix] == source_domain:
                label = label.to(device)
                gt_guided_loss = dice_loss(student_target, label)
                student_losses += gt_guided_loss
            student_losses.backward()
            optimizer_student.step()

            student_kd_losses[ix] += student_losses.item()
        student_kd_losses[ix] /= len(it)

    return student_kd_losses / 30


def getDSC(testImage, resultImage):
    """Compute the Dice Similarity Coefficient."""
    testArray = testImage.flatten()
    resultArray = resultImage.flatten()

    return 1.0 - scipy.spatial.distance.dice(testArray, resultArray)


def eval(model, device, test_loader):
    model.eval()
    dices = AverageMeter("Dice", ".4f")
    with torch.no_grad():
        for data, target in test_loader:
            o_data = data[:,0,:,:].unsqueeze(1)
            data = o_data
            data, target = data.to(device), target.to(device)
            output,__ = model(data)
            pred = torch.round(output)
            dice = getDSC(target.squeeze().detach().cpu().numpy(), pred.squeeze().detach().cpu().numpy())
            dices.update(dice, data.size(0))
    
    model.train()

    return dices.avg


def train(device, epochs, targets_dloader, targets_testloader, optimizer_student, student_model, expert_models, args=None, writer=None):

    best_student_acc = 0.
    epochs += 1
    for epoch in range(1, epochs):

        student_losses = train_one_epoch(student_model=student_model, expert_models=expert_models, optimizer_student=optimizer_student, device=device,  targets_dataloader=targets_dloader, args=args)

        print('Train Epoch {}: Student Loss {:.4f}'.format(epoch, np.average(student_losses)))

        train_student_losses = {}
        for i in range(len(args.domains)):
            train_student_losses[args.domains[i]] = student_losses[i]
        writer.add_scalars("Train Student Losses", train_student_losses, epoch)

        students_targets_acc = np.zeros(len(args.domains))

        for i, (domain, testloader) in enumerate(targets_testloader.items()):
            students_targets_acc[i] = eval(student_model, device, testloader)

        student_dice = np.average(students_targets_acc)

        print('Test Epoch {}: Student Dice {:.4f}'.format(epoch, student_dice))

        writer.add_scalar("Test Student Dice", student_dice, epoch)

        if student_dice > best_student_acc:
            best_student_acc = student_dice
            if torch.cuda.device_count() > 1:
                torch.save(student_model.module.state_dict(), "{}/student_best_model.pth".format(args.model_save_path))
            else:
                torch.save(student_model.state_dict(), "{}/student_best_model.pth".format(args.model_save_path))
        if epoch % 20 == 0:
            if torch.cuda.device_count() > 1:
                torch.save(student_model.module.state_dict(), "{}/student_model_{}.pth".format(args.model_save_path, epoch))
            else:
                torch.save(student_model.state_dict(), "{}/student_model_{}.pth".format(args.model_save_path, epoch))

    return best_student_acc

def main():
    args = create_train_arg_parser().parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_no

    process_train_args(args)
    print(args.transfer_type)

    model_save_path = os.path.join(args.save_path, "models")
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    args.model_save_path = model_save_path

    log_path = os.path.join(args.save_path, "summary/")
    writer = SummaryWriter(log_dir=log_path)

    device = torch.device("cuda:{}".format(args.cuda_no[0]) if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        student_model = nn.DataParallel(Student(args)).to(device)
        expert_similar_model = nn.DataParallel(Expert(args)).to(device)
        expert_dissimilar_model = nn.DataParallel(Expert(args)).to(device)
    else:
        student_model = Student(args).to(device)
        expert_similar_model = Expert(args).to(device)
        expert_dissimilar_model = Expert(args).to(device)

    optimizer_student = torch.optim.Adam([dict(params=student_model.parameters(), lr=1e-4)])
    optimizer_expert_similar = torch.optim.Adam([dict(params=expert_similar_model.parameters(), lr=1e-4)])
    optimizer_expert_dissimilar = torch.optim.Adam([dict(params=expert_dissimilar_model.parameters(), lr=1e-4)])

    source_dataloader = create_single_data_loader(args, train_aug(), 'train', False)
    source_invert_dataloader = create_single_data_loader(args, train_aug(), 'train', True)
    target_similar_dataloader = create_multi_train_data_loader(args, train_aug(), 'train', True)
    target_dissimilar_dataloader = create_multi_train_data_loader(args, train_aug(), 'train', False)
    target_trainloader = create_target_dataloader(args, train_aug(), 'train')
    target_testloader = create_target_dataloader(args, val_aug(), 'test')

    pretrain_expert(args=args, source_dataloader=source_dataloader, targets_dataloader=target_similar_dataloader, expert=expert_similar_model, optimizer=optimizer_expert_similar, device=device, writer=writer, similar=True)
    pretrain_expert(args=args, source_dataloader=source_invert_dataloader, targets_dataloader=target_dissimilar_dataloader, expert=expert_dissimilar_model, optimizer=optimizer_expert_dissimilar, device=device, writer=writer, similar=False)
    if torch.cuda.device_count() > 1:
        expert_similar_model.module.load_state_dict(torch.load(os.path.join(args.model_save_path, 'expert_similar_model.pth')))
        expert_dissimilar_model.module.load_state_dict(torch.load(os.path.join(args.model_save_path, 'expert_dissimilar_model.pth')))
    else:
        expert_similar_model.load_state_dict(torch.load(os.path.join(args.model_save_path, 'expert_similar_model.pth')))
        expert_dissimilar_model.load_state_dict(torch.load(os.path.join(args.model_save_path, 'expert_dissimilar_model.pth')))
    for param in expert_similar_model.parameters():
        param.requires_grad = False
    for param in expert_dissimilar_model.parameters():
        param.requires_grad = False

    experts = [expert_similar_model, expert_dissimilar_model]
    
    train(device=device, epochs=args.student_epochs, targets_dloader=target_trainloader, targets_testloader=target_testloader, optimizer_student=optimizer_student, student_model=student_model, expert_models=experts, args=args, writer=writer)

if __name__ == "__main__":
    main()