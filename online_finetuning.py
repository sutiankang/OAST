from copy import deepcopy
import torch
import argparse
import os
import os.path as osp
import numpy as np
import random
import datetime
import time
import sys
import math

import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from utils.distributed import init_distributed_mode, get_rank, is_main_process, reduce_value, save_on_master
from utils.logger import setup_logger
from utils.utils import get_list, get_size, MetricLogger, SmoothedValue
from utils.loss import create_online_iou_bce_loss
from models.mobilevit3d import MobileViTVOS
from models.discriminator import DiscriminatorFinetuning
from datasets.dataset import UVOSDataset
from metrics.compute_iou import db_eval_iou_multi

import warnings
warnings.filterwarnings("ignore")


def get_parser():

    parser = argparse.ArgumentParser("Online Finetuning")

    # common config
    parser.add_argument("--start_epoch", type=int, default=1, help="start epcoh")
    parser.add_argument("--epochs", type=int, default=10, help="training epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=2, help="batch size for DataLoader")
    parser.add_argument("--lr_g", type=float, default=1e-5, help="learning rate for single gpu")
    parser.add_argument("--lr_d", type=float, default=1e-5, help="discriminator learning rate")
    parser.add_argument("--img_size", type=int, nargs="+", default=[384, 640], help="training image size")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="choose using device")
    parser.add_argument("--print_freq", default=50, type=int, help="print information frequency")
    parser.add_argument("--data_dir", type=str, default="your/data/path", help="dataset path")

    parser.add_argument("--experiment", type=str, default="online_finetuning", help="experiment name")
    parser.add_argument("--model_scale", type=str, default="xxs", choices=["xxs", "xs", "s"], help="model size")
    parser.add_argument("--test_datasets", type=str, nargs="+", default=['DAVIS-2016', 'FBMS'])
    parser.add_argument("--lambda_adv", type=float, default=1e-4)

    # optimizer
    parser.add_argument("--optimizer", type=str, default="adamW", choices=["sgd", "adam", "adamW"],
                        help="choose optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="optimizer weight decay")
    parser.add_argument("--momentum", type=float, default=0.9, help="sgd momentum parameters")

    # data augmentation
    parser.add_argument("--mean", type=list, default=[0.485, 0.456, 0.406], help="imagenet mean")
    parser.add_argument("--std", type=list, default=[0.229, 0.224, 0.225], help="imagenet std")

    # training strategy
    parser.add_argument("--sync-bn", action="store_true", default=False,
                        help="distributed training merge batch_norm layer mean and std")
    parser.add_argument("--pretrained", type=str, default="your/pretrained/path", help="backbone pretrained weight")
    parser.add_argument("--dropout", default=0, type=float, help="before segmentation head add dropout")
    parser.add_argument("--weight", type=str, default="your/weight/path", help="finetune weight path")
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    return parser.parse_args()


def flip(x, dim):
    if x.is_cuda:
        # dim -> w dimension
        return torch.index_select(x, dim, torch.arange(x.size(dim) - 1, -1, -1).long().cuda())
    else:
        return torch.index_select(x, dim, torch.arange(x.size(dim) - 1, -1, -1).long())


def create_ema_model(model):
    ema_model = deepcopy(model)

    for param in ema_model.parameters():
        param.detach_()
    mp = list(model.parameters())
    mcp = list(ema_model.parameters())
    n = len(mp)
    for i in range(0, n):
        mcp[i].data[:] = mp[i].data[:].clone()
    return ema_model


def update_ema_variables(ema_model, model, alpha_teacher, iteration=None):
    # Use the "true" average until the exponential average is more correct
    if iteration:
        alpha_teacher = min(1 - 1 / (iteration + 1), alpha_teacher)

    if True:
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model


def online_finetuning(args):
    cudnn.benchmark = True
    init_distributed_mode(args)

    if not args.experiment:
        args.experiment = "online_finetuning"

    now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    exp_dir = osp.join("runs", args.experiment, now)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(osp.join(exp_dir, 'checkpoint'), exist_ok=True)
    logger = setup_logger(output=exp_dir, name=args.experiment)

    # fix the seed for reproducibility
    seed = args.seed + get_rank()
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = MobileViTVOS(args)
    model_D = DiscriminatorFinetuning()
    device = torch.device(args.device)
    model.to(device)
    model_D.to(device)
    ema_model = create_ema_model(model)

    criterion_seg = create_online_iou_bce_loss
    criterion_adv = nn.BCELoss()

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model_without_ddp = model
    ema_model_without_ddp = ema_model
    model_D_without_ddp = model_D

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
        model_D = torch.nn.parallel.DistributedDataParallel(model_D, device_ids=[args.gpu])
        model_D_without_ddp = model_D.module
        ema_model = torch.nn.parallel.DistributedDataParallel(ema_model, device_ids=[args.gpu])
        ema_model_without_ddp = ema_model.module

    n_learn_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_parameters = sum(p.numel() for p in model.parameters())
    n_learn_d_parameters= sum(p.numel() for p in model_D.parameters() if p.requires_grad)
    n_d_parameters = sum(p.numel() for p in model_D.parameters())

    if args.optimizer == 'adam':
        optimizer_g = torch.optim.Adam(model.parameters(), args.lr_g, weight_decay=args.weight_decay)
        optimizer_d = torch.optim.Adam(model_D.parameters(), args.lr_d, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamW':
        optimizer_g = torch.optim.AdamW(model.parameters(), args.lr_g, weight_decay=args.weight_decay)
        optimizer_d = torch.optim.AdamW(model_D.parameters(), args.lr_d, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer_g = torch.optim.SGD(model.parameters(), args.lr_g / 10 * args.batch_size,
                                      momentum=args.momentum, weight_decay=args.weight_decay)
        optimizer_d = torch.optim.SGD(model_D.parameters(), args.lr_d / 10 * args.batch_size,
                                      momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError("have no this optimizer, please check optimizer: {},"
                                  " or you can add optimizer".format(args.optimizer))

    test_dataset = UVOSDataset(data_dir=args.data_dir, size=get_size(args.img_size), mean=args.mean, std=args.std,
                               mode='test', datasets=get_list(args.test_datasets))
    if args.distributed:
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
    else:
        test_sampler = SequentialSampler(test_dataset)

    test_data_loader = DataLoader(test_dataset, args.batch_size, sampler=test_sampler, drop_last=False,
                                  pin_memory=True, num_workers=8)

    if is_main_process():
        logger.info(model)
        logger.info(model)
        logger.info("Random Seed: {}.".format(args.seed))
        logger.info("number of trainable model params (M): %.2f M." % (n_learn_parameters/1024/1024))
        logger.info('number of total model params (M): %.2f M.' % (n_parameters/1024/1024))
        logger.info("number of trainable discriminator params (M): %.2f M." % (n_learn_d_parameters/1024/1024))
        logger.info('number of total discriminator params (M): %.2f M.' % (n_d_parameters/1024/1024))
        logger.info("init learning rate of model: {}.".format(args.lr_g))
        logger.info("init learning rate of discriminator: {}.".format(args.lr_d))
        logger.info("Optimizer choice: {}.".format(args.optimizer))
        logger.info("Batch Size: {}.".format(args.batch_size))
        logger.info("Dataset sample number: {}.".format(len(test_dataset)))

    assert args.weight != "", f"--weight must exist."

    checkpoint = torch.load(args.weight, map_location='cpu')
    model_without_ddp.load_state_dict(checkpoint['model'], strict=True)
    ema_model_without_ddp.load_state_dict(checkpoint['model'], strict=True)

    if is_main_process():
        logger.info('Start Online Finetuning.')

    tic = time.time()
    best_iou, best_epoch = 0., 0

    for epoch in range(args.start_epoch, args.epochs + 1):
        if args.distributed:
            test_sampler.set_epoch(epoch)

        model.eval()
        ema_model.eval()

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr_g', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('lr_d', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Online Finetuning: Epoch: [{}/{}]'.format(epoch, args.epochs)
        for batch_test in metric_logger.log_every(test_data_loader, args.print_freq, logger, header):
            image, flow, mask = batch_test['image'], batch_test['flow'], batch_test['mask']
            image, flow, mask = image.to(device), flow.to(device), mask.to(device)
            aug_image1, aug_image2, aug_image3 = batch_test['aug_image1'], batch_test['aug_image2'], batch_test['aug_image3']
            aug_image1, aug_image2, aug_image3 = aug_image1.to(device), aug_image2.to(device), aug_image3.to(device)

            size = get_size(args.img_size)
            rates = [0.5, 0.75, 1.25, 1.5]

            aug_image1 = flip(image, 3)
            aug_flow1 = flip(flow, 3)

            aug_image2 = aug_image2
            aug_flow2 = flow

            aug_image3 = image
            aug_flow3 = flow
            rate = random.choice(rates)
            h, w = int(round(size[0] * rate / 64) * 64), int(round(size[1] * rate / 64) * 64)
            aug_image3 = F.interpolate(aug_image3, size=(h, w), mode='bilinear', align_corners=True)
            aug_flow3 = F.interpolate(aug_flow3, size=(h, w), mode='bilinear', align_corners=True)

            with torch.no_grad():

                pseudo_mask, _ = ema_model(image, flow)
                pseudo_mask_aug1, _ = ema_model(aug_image1, aug_flow1)
                pseudo_mask_aug2, _ = ema_model(aug_image2, aug_flow2)
                pseudo_mask_aug3, _ = ema_model(aug_image3, aug_flow3)
                # align
                pseudo_mask_aug1 = flip(pseudo_mask_aug1, 3)
                pseudo_mask_aug3 = F.interpolate(pseudo_mask_aug3, size=size, mode='bilinear', align_corners=True)
                pseudo_mask = torch.stack([pseudo_mask, pseudo_mask_aug1, pseudo_mask_aug2, pseudo_mask_aug3], 0).mean(0)

            predict_mask, _ = model(image, flow)
            losses = criterion_seg(predict_mask, pseudo_mask)
            loss_value_g = reduce_value(losses, average=True).item()

            if not math.isfinite(loss_value_g) and is_main_process():
                logger.info("Loss is {}, stopping training".format(loss_value_g))
                logger.info(loss_value_g)
                sys.exit(1)

            optimizer_g.zero_grad()
            optimizer_d.zero_grad()
            for param in model_D.parameters():
                param.requires_grad = False
            adv_g = criterion_adv(model_D(predict_mask), torch.ones_like(predict_mask, device=device))
            losses += args.lambda_adv * adv_g
            losses.backward()
            optimizer_g.step()

            for param in model_D.parameters():
                param.requires_grad = True
            real_loss = criterion_adv(model_D(pseudo_mask.detach()), torch.ones_like(pseudo_mask, device=device))
            fake_loss = criterion_adv(model_D(predict_mask.detach()), torch.zeros_like(predict_mask, device=device))
            d_loss = 0.5 * args.lambda_adv * (real_loss + fake_loss)
            loss_value_d = reduce_value(d_loss, average=True).item()
            d_loss.backward()
            optimizer_d.step()

            precise_student = torch.sigmoid(predict_mask).cpu().detach().numpy()
            precise_student[precise_student >= 0.5] = 1
            precise_student[precise_student < 0.5] = 0

            precise_teacher = torch.sigmoid(pseudo_mask).cpu().detach().numpy()
            precise_teacher[precise_teacher >= 0.5] = 1
            precise_teacher[precise_teacher < 0.5] = 0

            ema_model = update_ema_variables(ema_model, model, alpha_teacher=0.9996)
            student_iou = db_eval_iou_multi(mask.cpu().detach().numpy(), precise_student)
            teacher_iou = db_eval_iou_multi(mask.cpu().detach().numpy(), precise_teacher)
            student_iou = reduce_value(torch.tensor(student_iou, device=device), average=True).item()
            teacher_iou = reduce_value(torch.tensor(teacher_iou, device=device), average=True).item()

            metric_logger.update(loss_g=loss_value_g)
            metric_logger.update(loss_d=loss_value_d)
            metric_logger.update(lr_g=optimizer_g.param_groups[0]["lr"])
            metric_logger.update(lr_d=optimizer_d.param_groups[0]["lr"])
            metric_logger.update(teacher_mIoU=teacher_iou)
            metric_logger.update(student_mIoU=student_iou)

            torch.cuda.synchronize()

        metric_logger.synchronize_between_processes()
        if is_main_process():
            logger.info("Averaged stats: {}".format(metric_logger))
        stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

        if best_iou < stats["teacher_mIoU"] or best_iou < stats["student_mIoU"] and is_main_process():
            best_iou = max(stats["teacher_mIoU"], stats["student_mIoU"])
            best_epoch = epoch
            save_student_files = {'model': model_without_ddp.state_dict(),
                                  'model_D': model_D_without_ddp.state_dict(),
                                  'optimizer': optimizer_g.state_dict(),
                                  'args': args,
                                  'epoch': epoch}

            save_teacher_files = {'model': ema_model_without_ddp.state_dict(),
                                  'model_D': model_D_without_ddp,
                                  'optimizer': optimizer_g.state_dict(),
                                  'args': args,
                                  'epoch': epoch}
            save_on_master(save_student_files, osp.join(exp_dir, "checkpoint", f'{args.experiment}_student_{epoch}.pth'))
            save_on_master(save_teacher_files, osp.join(exp_dir, "checkpoint", f'{args.experiment}_teacher_{epoch}.pth'))
            logger.info("model saved {}!".format(osp.join(exp_dir, "checkpoint", f'{args.experiment}_student_{epoch}.pth')))
            logger.info("model saved {}!".format(osp.join(exp_dir, "checkpoint", f'{args.experiment}_teacher_{epoch}.pth')))
        if is_main_process():
            logger.info("best mIoU: {}, best epoch: {}, teacher mIoU: {}, student mIoU: {} ".
                        format(best_iou, best_epoch, stats["teacher_mIoU"], stats["student_mIoU"]))
    if is_main_process():
        logger.info("finish online update, cost time in {} epochs: {:.2f}h".format(args.epochs - args.start_epoch,
                                                                                   (time.time() - tic) / 3600))


if __name__ == '__main__':

    args = get_parser()
    online_finetuning(args)
