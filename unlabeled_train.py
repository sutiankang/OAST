import argparse
import numpy as np
import os
import datetime
import os.path as osp
import torch.backends.cudnn as cudnn
import torch
import random
import time
import math
import sys
import torch.nn as nn
from tqdm import tqdm

from torch.utils.data import DistributedSampler, RandomSampler, SequentialSampler, DataLoader, BatchSampler
from torch.optim import lr_scheduler

from models.mobilevit3d import MobileViTVOS
from models.discriminator import DiscriminatorTrain
from utils.distributed import init_distributed_mode, get_rank, is_main_process, reduce_value, save_on_master
from utils.logger import setup_logger
from utils.loss import create_iou_bce_loss
from utils.utils import AverageMeter, get_size, get_list
from metrics.compute_iou import db_eval_iou_multi
from datasets.dataset import UVOSDataset


import warnings
warnings.filterwarnings("ignore")


def get_parser():
    parser = argparse.ArgumentParser("Unlabeled Training")

    # common config
    parser.add_argument("--start_epoch", type=int, default=None)
    parser.add_argument("--start_adv", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="training epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=2, help="batch size for DataLoader")
    parser.add_argument("--lr_g", type=float, default=1e-4, help="learning rate for single gpu")
    parser.add_argument("--lr_d", type=float, default=1e-4, help="discriminator learning rate")
    parser.add_argument("--img_size", type=int, nargs="+", default=[384, 640], help="training image size")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="choose using device")
    parser.add_argument("--print_freq", default=50, type=int, help="print information frequency")
    parser.add_argument("--data_dir", type=str, default=None, help="dataset path")

    parser.add_argument("--experiment", type=str, default="offline_training", help="experiment name")
    parser.add_argument("--model_scale", type=str, default=None, choices=["xxs", "xs", "s"], help="model size")
    parser.add_argument("--labeled_datasets", type=str, nargs="+", default=['YouTubeVOS-2018', 'DAVIS-2016'])
    parser.add_argument("--unlabeled_datasets", type=str, nargs="+", default=['Youtube-objects'])
    parser.add_argument("--test_datasets", type=str, nargs="+", default=['DAVIS-2016', 'FBMS'])

    # optimizer
    parser.add_argument("--optimizer", type=str, default="adamW", choices=["sgd", "adam", "adamW"],
                        help="choose optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="optimizer weight decay")
    parser.add_argument("--momentum", type=float, default=0.9, help="sgd momentum parameters")
    parser.add_argument("--lambda_adv", type=float, default=1e-4)

    # data augmentation
    parser.add_argument("--mean", type=list, default=None)
    parser.add_argument("--std", type=list, default=None)
    parser.add_argument("--youtube_stride", type=int, default=10, help="youtube dataset sample")

    # training strategy
    parser.add_argument("--sync-bn", action="store_true", default=False,
                        help="distributed training merge batch_norm layer mean and std")
    parser.add_argument("--pretrained", type=str, default=None, help="backbone pretrained weight")
    parser.add_argument("--dropout", default=None, type=float, help="before segmentation head add dropout")
    parser.add_argument("--finetune", type=str, default=None, help="finetune weight path")
    parser.add_argument('--resume', default=None, help='resume from checkpoint')

    return parser.parse_args()


def main(args):
    cudnn.benchmark = True

    init_distributed_mode(args)

    if not args.experiment:
        args.experiment = "offline_training"

    now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    exp_dir = osp.join("runs", args.experiment, now)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(osp.join(exp_dir, 'checkpoint'), exist_ok=True)
    logger = setup_logger(output=exp_dir, name=args.experiment)

    for key, value in sorted(vars(args).items()):
        if is_main_process():
            logger.info(str(key) + ': ' + str(value))

    seed = args.seed + get_rank()
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = MobileViTVOS(args)
    model_D = DiscriminatorTrain()
    device = torch.device(args.device)
    model.to(device)
    model_D.to(device)

    criterion_seg = create_iou_bce_loss
    criterion_adv = nn.BCEWithLogitsLoss()

    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=True)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0 and is_main_process():
            logger.info('Missing Generator Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0 and is_main_process():
            logger.info('Unexpected Generator Keys: {}'.format(unexpected_keys))
        if is_main_process():
            logger.info("loading checkpoint from {} to finetune model.".format(args.finetune))

    img_size = get_size(args.img_size)
    labeled_dataset = UVOSDataset(data_dir=args.data_dir, size=img_size, mean=args.mean, std=args.std,
                                  mode='labeled', datasets=get_list(args.labeled_datasets), stride=args.youtube_stride)
    unlabeled_dataset = UVOSDataset(data_dir=args.data_dir, size=img_size, mean=args.mean, std=args.std,
                                    mode='unlabeled', datasets=get_list(args.unlabeled_datasets))
    test_dataset = UVOSDataset(data_dir=args.data_dir, size=img_size, mean=args.mean, std=args.std,
                               mode='test', datasets=get_list(args.test_datasets))

    if args.distributed:
        labeled_sampler = DistributedSampler(labeled_dataset)
        unlabeled_sampler = DistributedSampler(unlabeled_dataset)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
    else:
        labeled_sampler = RandomSampler(labeled_dataset)
        unlabeled_sampler = RandomSampler(unlabeled_dataset)
        test_sampler = SequentialSampler(test_dataset)

    labeled_batch_sampler = BatchSampler(labeled_sampler, args.batch_size, drop_last=True)
    labeled_data_loader = DataLoader(labeled_dataset, batch_sampler=labeled_batch_sampler, num_workers=8, pin_memory=True)
    unlabeled_batch_sampler = BatchSampler(unlabeled_sampler, args.batch_size, drop_last=True)
    unlabeled_data_loader = DataLoader(unlabeled_dataset, batch_sampler=unlabeled_batch_sampler, num_workers=8, pin_memory=True)

    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler, drop_last=False,
                                   pin_memory=True, num_workers=8)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model_D = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_D)
    model_without_ddp = model
    model_D_without_ddp = model_D
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], broadcast_buffers=False)
        model_D = torch.nn.parallel.DistributedDataParallel(model_D, device_ids=[args.gpu], broadcast_buffers=False)
        model_without_ddp = model.module
        model_D_without_ddp = model_D.module

    n_learn_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_learn_parameters_D = sum(p.numel() for p in model_D.parameters() if p.requires_grad)
    n_parameters = sum(p.numel() for p in model.parameters())
    n_parameters_D = sum(p.numel() for p in model_D.parameters())

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

    schedule_g = lr_scheduler.StepLR(optimizer_g, step_size=10, gamma=0.1)
    schedule_d = lr_scheduler.StepLR(optimizer_d, step_size=10, gamma=0.1)
    scaler = torch.cuda.amp.GradScaler()

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint["model"], strict=True)
        missing_keys_D, unexpected_keys_D = model_D_without_ddp.load_state_dict(checkpoint["model_D"], strict=True)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        unexpected_keys_D = [k for k in unexpected_keys_D if not (k.endswith('total_params') or k.endswith('total_ops'))]
        optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        optimizer_d.load_state_dict(checkpoint["optimizer_d"])

        args.start_epoch = checkpoint["epoch"] + 1
        if len(missing_keys) > 0 and is_main_process():
            logger.info('Missing Generator Keys: {}'.format(missing_keys))
        if len(missing_keys_D) > 0 and is_main_process():
            logger.info('Missing Discriminator Keys: {}'.format(missing_keys_D))
        if len(unexpected_keys) > 0 and is_main_process():
            logger.info('Unexpected Generator Keys: {}'.format(unexpected_keys))
        if len(unexpected_keys_D) > 0 and is_main_process():
            logger.info('Unexpected Discriminator Keys: {}'.format(unexpected_keys_D))
        if is_main_process():
            logger.info("loading checkpoint from {} to continue last training.".format(args.resume))

    if is_main_process():
        logger.info(model)
        logger.info("Random Seed: {}.".format(args.seed))
        logger.info("number of Generator trainable params (M): %.2f M." % (n_learn_parameters/1024/1024))
        logger.info("number of Discriminator trainable params (M): %.2f M." % (n_learn_parameters_D/1024/1024))
        logger.info('number of Generator total params (M): %.2f M.' % (n_parameters/1024/1024))
        logger.info("number of Discriminator total params (M): %.2f M." % (n_parameters_D/1024/1024))
        logger.info("init Generator learning rate: {}.".format(args.lr_g))
        logger.info("init Discriminator learning rate: {}.".format(args.lr_d))
        logger.info("Optimizer choice: {}.".format(args.optimizer))
        logger.info("Batch Size: {}.".format(args.batch_size))
        logger.info("Labeled dataset sample number: {}.".format(len(labeled_dataset)))
        logger.info("Unlabeled dataset sample number: {}.".format(len(unlabeled_dataset)))
        logger.info("Test dataset sample number: {}.".format(len(test_dataset)))

    unlabeled_loader_iter = enumerate(unlabeled_data_loader)

    if is_main_process():
        logger.info('Start training.')
    tic = time.time()
    best_iou, best_epoch = 0., 0

    start = time.time()

    for epoch in range(args.start_epoch, args.epochs + 1):
        if args.distributed:
            labeled_sampler.set_epoch(epoch)

        loss_labeled_record = AverageMeter()
        loss_d_record = AverageMeter()
        train_iou_record = AverageMeter()
        test_iou_record = AverageMeter()
        test_loss_record = AverageMeter()

        model.train()
        for idx, labeled_batch in enumerate(labeled_data_loader):
            labeled_image, labeled_flow, labeled_mask = labeled_batch['image'], labeled_batch['flow'], labeled_batch['mask']
            labeled_image, labeled_flow, labeled_mask = labeled_image.to(device), labeled_flow.to(device), labeled_mask.to(device)

            optimizer_g.zero_grad()
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                labeled_predict, labeled_feature_map = model(labeled_image, labeled_flow)
                loss_labeled = criterion_seg(labeled_predict, labeled_mask)
            precise_output = torch.sigmoid(labeled_predict).cpu().detach().numpy()
            precise_output[precise_output >= 0.5] = 1
            precise_output[precise_output < 0.5] = 0

            iou = db_eval_iou_multi(labeled_mask.cpu().detach().numpy(), precise_output)
            iou = reduce_value(torch.tensor(iou, device=device), average=True).item()
            train_iou_record.update(iou)

            if epoch < args.start_adv:
                scaler.scale(loss_labeled).backward()
                scaler.step(optimizer_g)
                scaler.update()

                loss_value = reduce_value(loss_labeled, average=True).item()
                loss_labeled_record.update(loss_value)

                if not math.isfinite(loss_value) and is_main_process():
                    logger.info("Loss is {}, stopping training".format(loss_value))
                    logger.info(loss_value)
                    sys.exit(1)

            if epoch >= args.start_adv:

                try:
                    _, unlabeled_batch = unlabeled_loader_iter.__next__()
                except StopIteration:
                    unlabeled_loader_iter = enumerate(unlabeled_data_loader)
                    _, unlabeled_batch = unlabeled_loader_iter.__next__()

                unlabeled_image, unlabeled_flow = unlabeled_batch['image'], unlabeled_batch['flow']
                unlabeled_image, unlabeled_flow = unlabeled_image.to(device), unlabeled_flow.to(device)

                optimizer_d.zero_grad()
                for param in model_D.parameters():
                    param.requires_grad = False

                with torch.cuda.amp.autocast(enabled=scaler is not None):
                    _, unlabeled_feature_map = model(unlabeled_image, unlabeled_flow)
                    adv_g = criterion_adv(model_D(unlabeled_feature_map), torch.ones(args.batch_size, device=device))
                    loss_labeled += args.lambda_adv * adv_g
                loss_value = reduce_value(loss_labeled, average=True).item()
                loss_labeled_record.update(loss_value)

                scaler.scale(loss_labeled).backward()
                scaler.step(optimizer_g)
                scaler.update()

                for param in model_D.parameters():
                    param.requires_grad = True
                with torch.cuda.amp.autocast(enabled=scaler is not None):
                    real_loss = criterion_adv(model_D(labeled_feature_map.detach()), torch.ones(args.batch_size, device=device))
                    fake_loss = criterion_adv(model_D(unlabeled_feature_map.detach()), torch.zeros(args.batch_size, device=device))
                    d_loss = 0.5 * args.lambda_adv * (real_loss + fake_loss)
                loss_value = reduce_value(d_loss, average=True).item()
                loss_d_record.update(loss_value)

                scaler.scale(d_loss).backward()
                scaler.step(optimizer_d)
                scaler.update()

            if (idx % args.print_freq == 0 or epoch == args.epochs) and is_main_process():
                if epoch >= args.start_adv:
                    logger.info("Time: {:.2f}s, Epoch: [{}/{}] [{}/{}], Generator Learning Rate: {:.6f}, Discriminator Learning"
                                " Rate: {:.6f} Generator Loss: {}, Discriminator Loss: {}, Train mIoU: {}."
                                .format(time.time() - start, epoch, args.epochs, idx, len(labeled_data_loader),
                                        optimizer_g.param_groups[0]["lr"], optimizer_d.param_groups[0]["lr"],
                                        loss_labeled_record.average, loss_d_record.average, train_iou_record.average))
                else:
                    logger.info("Time: {:.2f}s, Epoch: [{}/{}] [{}/{}], Generator Learning Rate: {:.6f}"
                                " Generator Loss: {}, Train mIoU: {}."
                                .format(time.time() - start, epoch, args.epochs, idx, len(labeled_data_loader),
                                        optimizer_g.param_groups[0]["lr"], loss_labeled_record.average, train_iou_record.average))
                start = time.time()

        schedule_g.step()
        schedule_d.step()

        # evaluation
        print(f"Start evaluating...")
        with torch.no_grad():
            model.eval()
            for idx, batch_test in enumerate(tqdm(test_data_loader)):
                test_image, test_flow, test_mask = batch_test['image'], batch_test['flow'], batch_test['mask']
                test_image, test_flow, test_mask = test_image.to(device), test_flow.to(device), test_mask.to(device)

                test_predict, _ = model(test_image, test_flow)
                precise_output = torch.sigmoid(test_predict).cpu().detach().numpy()
                precise_output[precise_output >= 0.5] = 1
                precise_output[precise_output < 0.5] = 0
                loss_eval = criterion_seg(test_predict, test_mask)
                loss_value = reduce_value(loss_eval, average=True).item()
                iou = db_eval_iou_multi(test_mask.cpu().detach().numpy(), precise_output)
                iou = reduce_value(torch.tensor(iou, device=device), average=True).item()

                test_iou_record.update(iou)
                test_loss_record.update(loss_value)

            if best_iou < test_iou_record.average:
                best_iou = np.round(test_iou_record.average, 4)
                best_epoch = epoch
                save_files = {'model': model_without_ddp.state_dict(),
                              'model_D': model_D_without_ddp.state_dict(),
                              'optimizer_g': optimizer_g.state_dict(),
                              'optimizer_d': optimizer_d.state_dict(),
                              'args': args,
                              'epoch': epoch}
                save_on_master(save_files, osp.join(exp_dir, "checkpoint", f'{args.experiment}_{epoch}.pth'))
                save_on_master(save_files, osp.join(exp_dir, "checkpoint", f'{args.experiment}_best.pth'))
                if is_main_process():
                    logger.info("model saved {}!".format(osp.join(exp_dir, "checkpoint", f'{args.experiment}_{epoch}.pth')))

            if is_main_process():
                logger.info("current mIoU: {}, current epoch: {}, best mIoU: {}, best epoch: {}".
                            format(np.round(test_iou_record.average, 4), epoch, best_iou, best_epoch))

    if is_main_process():
        logger.info("finish training, cost training time in {} iterations: {:.2f}h".
                    format(args.epochs - args.start_epoch + 1, (time.time() - tic) / 3600))


if __name__ == '__main__':
    args = get_parser()
    main(args)
