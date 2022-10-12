import argparse
import os
import random
import shutil
import time
import warnings
import numpy as np
import pprint
import math

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F

from datasets.cifar10 import CIFAR10_LT
from datasets.cifar100 import CIFAR100_LT
from datasets.stl10 import STL10_LT
from datasets.svhn import SVHN_LT
from datasets.imagenet import ImageNet_LT

from models import resnet
from models import resnet_cifar
from models import densenet

from utils import config, update_config, create_logger
from utils import AverageMeter, ProgressMeter
from utils import accuracy, calibration

from methods import mixup_data, mixup_criterion, dot_loss, produce_Ew, produce_global_Ew


def parse_args():
    parser = argparse.ArgumentParser(description='training')
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    update_config(config, args)

    return args


best_acc1 = 0
its_ece = 100


##   for stat_mode ##
cos_avg_HH_train = []
cos_avg_WW_train = []
cos_avg_HW_train = []
cos_std_HH_train = []
cos_std_WW_train = []
cos_std_HW_train = []
HM_F2_train = []
diag_avg_HW_train = []
diag_std_HW_train = []

cos_avg_HH_val = []
cos_avg_WW_val = []
cos_avg_HW_val = []
cos_std_HH_val = []
cos_std_WW_val = []
cos_std_HW_val = []
HM_F2_val = []
diag_avg_HW_val = []
diag_std_HW_val = []
#######################


def main():
    args = parse_args()
    logger, model_dir = create_logger(config, args.cfg)
    logger.info('\n' + pprint.pformat(args))
    logger.info('\n' + str(config))

    if config.deterministic:
        seed = 0
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if config.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if config.dist_url == "env://" and config.world_size == -1:
        config.world_size = int(os.environ["WORLD_SIZE"])

#    os.environ["RANK"] = str(0)
#    os.environ["MASTER_ADDR"]='localhost'
#    os.environ["MASTER_PORT"]='5678'

    config.distributed = config.world_size > 1 or config.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if config.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        config.world_size = ngpus_per_node * config.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config, logger))
    else:
        # Simply call main_worker function
        main_worker(config.gpu, ngpus_per_node, config, logger, model_dir)


def main_worker(gpu, ngpus_per_node, config, logger, model_dir):
    global best_acc1, its_ece
    config.gpu = gpu
#     start_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    if config.gpu is not None:
        logger.info("Use GPU: {} for training".format(config.gpu))

    if config.distributed:
        if config.dist_url == "env://" and config.rank == -1:
            config.rank = int(os.environ["RANK"])
        if config.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            config.rank = config.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=config.dist_backend, init_method=config.dist_url,
                                world_size=config.world_size, rank=config.rank)

    if config.dataset == 'cifar10' or config.dataset == 'cifar100' or config.dataset == 'stl10' or config.dataset == 'svhn':
        if 'dense' in config.backbone:
            #model = densenet.DenseNet3(config.num_classes, 40, 12)# 20, 10 feat len 170
            #feat_len = 456
            #model = densenet.DenseNet3(config.num_classes, 100, 12, reduction=0.5, bottleneck=True)
            #feat_len = 342
            model = densenet.DenseNet3(config.num_classes, 150, 12, reduction=0.5, bottleneck=True)
            feat_len = 510
            if config.ETF_classifier:
                print('########   Using a ETF as the last linear classifier    ##########')
                classifier = getattr(densenet, 'ETF_Classifier')(feat_in=feat_len, num_classes=config.num_classes)
            else:
                classifier = getattr(densenet, 'Classifier')(feat_in=feat_len, num_classes=config.num_classes)
        elif 'res' in config.backbone:
            model = getattr(resnet_cifar, config.backbone)()
            feat_len = 64 if config.dataset == 'cifar10' or config.dataset == 'stl10' or config.dataset=='svhn' else 128
            if config.ETF_classifier:
                print('########   Using a ETF as the last linear classifier    ##########')
                classifier = getattr(resnet_cifar, 'ETF_Classifier')(feat_in=feat_len, num_classes=config.num_classes)
            else:
                classifier = getattr(resnet_cifar, 'Classifier')(feat_in=feat_len, num_classes=config.num_classes)

    elif config.dataset == 'imagenet':
        model = getattr(resnet, config.backbone)()
        if config.ETF_classifier:
            print('########   Using a ETF as the last linear classifier    ##########')
            classifier = getattr(resnet, 'ETF_Classifier')(feat_in=4096, num_classes=config.num_classes)
        else:
            classifier = getattr(resnet, 'Classifier')(feat_in=4096, num_classes=config.num_classes)



    if not torch.cuda.is_available():
        logger.info('using CPU, this will be slow')
    elif config.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if config.gpu is not None:
            torch.cuda.set_device(config.gpu)
            model.cuda(config.gpu)
            classifier.cuda(config.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            config.batch_size = int(config.batch_size / ngpus_per_node)
            config.workers = int((config.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.gpu])
            classifier = torch.nn.parallel.DistributedDataParallel(classifier, device_ids=[config.gpu])
            if config.dataset == 'places':
                block.cuda(config.gpu)
                block = torch.nn.parallel.DistributedDataParallel(block, device_ids=[config.gpu])
        else:
            model.cuda()
            classifier.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
            classifier = torch.nn.parallel.DistributedDataParallel(classifier)
            if config.dataset == 'places':
                block.cuda()
                block = torch.nn.parallel.DistributedDataParallel(block)
    elif config.gpu is not None:
        torch.cuda.set_device(config.gpu)
        model = model.cuda(config.gpu)
        classifier = classifier.cuda(config.gpu)
        if config.dataset == 'places':
            block.cuda(config.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()
        classifier = torch.nn.DataParallel(classifier).cuda()
        if config.dataset == 'places':
            block = torch.nn.DataParallel(block).cuda()

    # optionally resume from a checkpoint
    if config.resume:
        if os.path.isfile(config.resume):
            logger.info("=> loading checkpoint '{}'".format(config.resume))
            if config.gpu is None:
                checkpoint = torch.load(config.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(config.gpu)
                checkpoint = torch.load(config.resume, map_location=loc)
            # config.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if config.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(config.gpu)
            model.load_state_dict(checkpoint['state_dict_model'])
            classifier.load_state_dict(checkpoint['state_dict_classifier'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                        .format(config.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(config.resume))

    # Data loading code
    if config.dataset == 'cifar10':
        dataset = CIFAR10_LT(config.distributed, root=config.data_path, imb_factor=config.imb_factor,
                             batch_size=config.batch_size, num_works=config.workers)

    elif config.dataset == 'cifar100':
        dataset = CIFAR100_LT(config.distributed, root=config.data_path, imb_factor=config.imb_factor,
                              batch_size=config.batch_size, num_works=config.workers)
    elif config.dataset == 'stl10':
        dataset = STL10_LT(config.distributed, root=config.data_path, imb_factor=config.imb_factor,
                              batch_size=config.batch_size, num_works=config.workers)
    elif config.dataset == 'svhn':
        dataset = SVHN_LT(config.distributed, root=config.data_path, imb_factor=config.imb_factor,
                              batch_size=config.batch_size, num_works=config.workers)

    elif config.dataset == 'imagenet':
        dataset = ImageNet_LT(config.distributed, root=config.data_path,
                              batch_size=config.batch_size, num_works=config.workers)


    if config.CLS_BALAN:
        train_loader = dataset.train_balance
        print('%%%%%%%%%%%%%   classes balanced training   %%%%%%%%%%%%%%%')
    else:
        train_loader = dataset.train_instance
        print('%%%%%%%%%%%%%   instance balanced training   %%%%%%%%%%%%%%%')
    val_loader = dataset.eval
    if config.distributed:
        train_sampler = dataset.dist_sampler

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(config.gpu)

    if config.reg_dot_loss:
        criterion = config.criterion
        print('----  Dot-Regression Loss is adopted ----')


    optimizer = torch.optim.SGD([{"params": model.parameters()},
                                {"params": classifier.parameters()}], config.lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)
#        optimizer = torch.optim.AdamW([{"params": model.parameters()},
#                                        {"params": classifier.parameters()}], eps=1e-8, betas=(0.9, 0.999), lr=5e-4,
#                                        weight_decay=config.weight_decay)

    for epoch in range(config.num_epochs):
        if config.distributed:
            train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, config)

        if config.dataset != 'places':
            block = None

        # train for one epoch
        train(train_loader, model, classifier, criterion, optimizer, epoch, config, logger, block)
        if config.stat_mode:
            print('\n%%%%%%%%%% train stat')
            acc_tr, ece_tr = validate(train_loader, model, classifier, criterion, config, logger, block, 'train')
            print('%%%%%%%%%% val stat')

        # evaluate on validation set
        acc1, ece = validate(val_loader, model, classifier, criterion, config, logger, block, 'test')

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        if is_best:
            its_ece = ece
        logger.info('Best Prec@1: %.3f%% \n' % (best_acc1))

        if not config.multiprocessing_distributed or (config.multiprocessing_distributed
                                                      and config.rank % ngpus_per_node == 0):
            if config.dataset != 'imagenet':
                if config.ETF_classifier:
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict_model': model.state_dict(),
                        'state_dict_classifier': classifier.state_dict(),
                        'cur_M': classifier.ori_M,
                        'best_acc1': best_acc1,
                        'its_ece': its_ece,
                    }, is_best, model_dir)
                else:
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict_model': model.state_dict(),
                        'state_dict_classifier': classifier.state_dict(),
                        'best_acc1': best_acc1,
                        'its_ece': its_ece,
                    }, is_best, model_dir)
            else:
                if config.ETF_classifier:
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict_model': model.state_dict(),
                        'state_dict_classifier': classifier.state_dict(),
                        'cur_M': classifier.module.ori_M,
                        'best_acc1': best_acc1,
                        'its_ece': its_ece,
                    }, is_best, model_dir)
                else:
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict_model': model.state_dict(),
                        'state_dict_classifier': classifier.state_dict(),
                        'best_acc1': best_acc1,
                        'its_ece': its_ece,
                    }, is_best, model_dir)

    if config.stat_mode:
        np.save(model_dir+'/cos_avg_HH_train.npy', np.array(cos_avg_HH_train))
        np.save(model_dir+'/cos_avg_WW_train.npy', np.array(cos_avg_WW_train))
        np.save(model_dir+'/cos_avg_HW_train.npy', np.array(cos_avg_HW_train))
        np.save(model_dir+'/cos_std_HH_train.npy', np.array(cos_std_HH_train))
        np.save(model_dir+'/cos_std_WW_train.npy', np.array(cos_std_WW_train))
        np.save(model_dir+'/cos_std_HW_train.npy', np.array(cos_std_HW_train))
        np.save(model_dir+'/diag_avg_train.npy', np.array(diag_avg_HW_train))
        np.save(model_dir+'/diag_std_train.npy', np.array(diag_std_HW_train))
        np.save(model_dir+'/HM_F2_train.npy', np.array(HM_F2_train))
        ##
        np.save(model_dir+'/cos_avg_HH_val.npy', np.array(cos_avg_HH_val))
        np.save(model_dir+'/cos_avg_WW_val.npy', np.array(cos_avg_WW_val))
        np.save(model_dir+'/cos_avg_HW_val.npy', np.array(cos_avg_HW_val))
        np.save(model_dir+'/cos_std_HH_val.npy', np.array(cos_std_HH_val))
        np.save(model_dir+'/cos_std_WW_val.npy', np.array(cos_std_WW_val))
        np.save(model_dir+'/cos_std_HW_val.npy', np.array(cos_std_HW_val))
        np.save(model_dir+'/diag_avg_val.npy', np.array(diag_avg_HW_val))
        np.save(model_dir+'/diag_std_val.npy', np.array(diag_std_HW_val))
        np.save(model_dir+'/HM_F2_val.npy', np.array(HM_F2_val))


def train(train_loader, model, classifier, criterion, optimizer, epoch, config, logger, block):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.3f')
    top1 = AverageMeter('Acc@1', ':6.3f')
    top5 = AverageMeter('Acc@5', ':6.3f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    if config.dataset == 'places':
        model.eval()
        block.train()
    else:
        model.train()
    classifier.train()

    training_data_num = len(train_loader.dataset)
    end_steps = int(training_data_num / train_loader.batch_size)

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        if i > end_steps:
            break
        # measure data loading time
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            images = images.cuda(config.gpu, non_blocking=True)
            target = target.cuda(config.gpu, non_blocking=True)

#        if config.ETF_classifier and config.LWS:
#            assert hasattr(classifier, 'ori_M')
#            classifier.learned_norm = F.softmax(classifier.alpha, dim=-1) * config.num_classes
#            classifier.module.cur_M = classifier.learned_norm * classifier.module.ori_M

        if config.ETF_classifier:
            if config.reg_dot_loss and config.GivenEw:
                learned_norm = produce_Ew(target, config.num_classes)
                if config.dataset == 'imagenet':
                    cur_M = learned_norm * classifier.module.ori_M
                else:
                    cur_M = learned_norm * classifier.ori_M
            else:
                if config.dataset == 'imagenet':
                    cur_M = classifier.module.ori_M
                else:
                    cur_M = classifier.ori_M

        if config.mixup is True:
            images, targets_a, targets_b, lam = mixup_data(images, target, alpha=config.alpha)
            ###
            #if config.ETF_mix is True:
            #    classifier, ETF_target = mix_ETF(classifier, targets_a, targets_b, lam, images.size(0))
            ###
            feat = model(images)
            if config.ETF_classifier:
                feat = classifier(feat)
                output = torch.matmul(feat, cur_M) #+ classifier.module.bias
                if config.reg_dot_loss:  ## ETF classifier + DR loss
                    with torch.no_grad():
                        feat_nograd = feat.detach()
                        H_length = torch.clamp(torch.sqrt(torch.sum(feat_nograd ** 2, dim=1, keepdims=False)), 1e-8)
                    loss_a = dot_loss(feat, targets_a, cur_M, classifier, criterion, H_length, reg_lam=config.reg_lam)
                    loss_b = dot_loss(feat, targets_b, cur_M, classifier, criterion, H_length, reg_lam=config.reg_lam)
                    loss = lam * loss_a + (1-lam) * loss_b
                else:   ## ETF classifier + CE loss
                    loss = mixup_criterion(criterion, output, targets_a, targets_b, lam)

            else:       ## learnable classifier + CE loss
                output = classifier(feat)
                loss = mixup_criterion(criterion, output, targets_a, targets_b, lam)

        else:
            feat = model(images)
            if config.ETF_classifier:
                feat = classifier(feat)
                output = torch.matmul(feat, cur_M)
                if config.reg_dot_loss:   ## ETF classifier + DR loss
                    with torch.no_grad():
                        feat_nograd = feat.detach()
                        H_length = torch.clamp(torch.sqrt(torch.sum(feat_nograd ** 2, dim=1, keepdims=False)), 1e-8)
                    loss = dot_loss(feat, target, cur_M, classifier, criterion, H_length, reg_lam=config.reg_lam)
                else:    ## ETF classifier + CE loss
                    loss = criterion(output, target)
            else:        ## learnable classifier + CE loss
                output = classifier(feat)
                loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.print_freq == 0:
            progress.display(i, logger)


def validate(val_loader, model, classifier, criterion, config, logger, block=None, dset='train'):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.3f')
    top1 = AverageMeter('Acc@1', ':6.3f')
    top5 = AverageMeter('Acc@5', ':6.3f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Eval: ')

    # switch to evaluate mode
    model.eval()
    if config.dataset == 'places':
        block.eval()
    classifier.eval()
    class_num = torch.zeros(config.num_classes).cuda()
    correct = torch.zeros(config.num_classes).cuda()

    confidence = np.array([])
    pred_class = np.array([])
    true_class = np.array([])

    feat_dict = {}
    cnt_dict = {}

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if config.gpu is not None:
                images = images.cuda(config.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(config.gpu, non_blocking=True)

            #if config.ETF_classifier and config.LWS:
            #    assert hasattr(classifier, 'ori_M')
            #    classifier.learned_norm = F.softmax(classifier.alpha, dim=-1) * config.num_classes
            #    classifier.cur_M = classifier.learned_norm * classifier.ori_M

            if config.ETF_classifier: # and config.reg_dot_loss and config.GivenEw:
                if config.dataset == 'imagenet':
                    cur_M = classifier.module.ori_M
                else:
                    cur_M = classifier.ori_M

            # compute output

            feat = model(images)
            if config.ETF_classifier:
                feat = classifier(feat)
                output = torch.matmul(feat, cur_M)
                if config.reg_dot_loss:
                    with torch.no_grad():
                        feat_nograd = feat.detach()
                        H_length = torch.clamp(torch.sqrt(torch.sum(feat_nograd ** 2, dim=1, keepdims=False)), 1e-8)
                    loss = dot_loss(feat, target, cur_M, classifier, criterion, H_length)
                else:
                    loss = criterion(output, target)
            else:
                output = classifier(feat)
                loss = criterion(output, target)

            if config.stat_mode:
                uni_lbl, count = torch.unique(target, return_counts=True)
                lbl_num = uni_lbl.size(0)
                for kk in range(lbl_num):
                    sum_feat = torch.sum(feat[torch.where(target==uni_lbl[kk])[0], :], dim=0)
                    key = uni_lbl[kk].item()
                    if uni_lbl[kk] in feat_dict.keys():
                        feat_dict[key] = feat_dict[key]+sum_feat
                        cnt_dict[key] =  cnt_dict[key]+count[kk]
                    else:
                        feat_dict[key] = sum_feat
                        cnt_dict[key] =  count[kk]



            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            _, predicted = output.max(1)
            target_one_hot = F.one_hot(target, config.num_classes)
            predict_one_hot = F.one_hot(predicted, config.num_classes)
            class_num = class_num + target_one_hot.sum(dim=0).to(torch.float)
            correct = correct + (target_one_hot + predict_one_hot == 2).sum(dim=0).to(torch.float)

            prob = torch.softmax(output, dim=1)
            confidence_part, pred_class_part = torch.max(prob, dim=1)
            confidence = np.append(confidence, confidence_part.cpu().numpy())
            pred_class = np.append(pred_class, pred_class_part.cpu().numpy())
            true_class = np.append(true_class, target.cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.print_freq == 0:
                progress.display(i, logger)
        acc_classes = correct / class_num
        head_acc = acc_classes[config.head_class_idx[0]:config.head_class_idx[1]].mean() * 100

        med_acc = acc_classes[config.med_class_idx[0]:config.med_class_idx[1]].mean() * 100
        tail_acc = acc_classes[config.tail_class_idx[0]:config.tail_class_idx[1]].mean() * 100
        logger.info('* Acc@1 {top1.avg:.3f}% Acc@5 {top5.avg:.3f}% HAcc {head_acc:.3f}% MAcc {med_acc:.3f}% TAcc {tail_acc:.3f}%.'.format(top1=top1, top5=top5, head_acc=head_acc, med_acc=med_acc, tail_acc=tail_acc))

        cal = calibration(true_class, pred_class, confidence, num_bins=15)
        #logger.info('* ECE   {ece:.3f}%.'.format(ece=cal['expected_calibration_error'] * 100))

        if config.stat_mode:
            ### calculate statistics
            #print(feat_dict)
            total_sum_feat = sum(feat_dict.values())
            total_counts = sum(cnt_dict.values())
            global_mean = total_sum_feat / total_counts
            class_mean = torch.zeros(config.num_classes, feat.size(1)).cuda()
            for i in range(config.num_classes):
                if i in feat_dict.keys():
                    class_mean[i,:] = feat_dict[i] / cnt_dict[i]
            class_mean = class_mean - global_mean  ## K, dim
            W_mean = classifier.ori_M.T if config.ETF_classifier else classifier.fc.weight ## K,dim
            ## F dist
            class_mean_F = class_mean / torch.sqrt(torch.sum(class_mean ** 2))
            W_mean_F = W_mean / torch.sqrt(torch.sum(W_mean ** 2))
            F_dist =torch.sum((class_mean_F - W_mean_F)**2)
            ##
            class_mean = class_mean / torch.sqrt(torch.sum(class_mean**2, dim=1, keepdims=True))
            W_mean = W_mean / torch.sqrt(torch.sum(W_mean **2, dim=1,keepdims=True))
            cos_HH = torch.matmul(class_mean, class_mean.T)
            cos_WW = torch.matmul(W_mean, W_mean.T)
            cos_HW = torch.matmul(class_mean, W_mean.T)
            #print(cos_HH)
            #print(cos_WW)
            #print(cos_HW)
            ##
            diag_HW = torch.diag(cos_HW, 0)
            diag_avg = torch.mean(diag_HW)
            diag_std = torch.std(diag_HW)
            ##
            up_HH = torch.cat([torch.diag(cos_HH, i) for i in range(1, config.num_classes)])
            up_WW = torch.cat([torch.diag(cos_WW, i) for i in range(1, config.num_classes)])
            up_HW = torch.cat([torch.diag(cos_HW, i) for i in range(1, config.num_classes)])
            up_HH_avg = torch.mean(up_HH)
            up_HH_std = torch.std(up_HH)
            up_WW_avg = torch.mean(up_WW)
            up_WW_std = torch.std(up_WW)
            up_HW_avg = torch.mean(up_HW)
            up_HW_std = torch.std(up_HW)
            ##
            print('cos-avg-HH', up_HH_avg)
            print('cos-avg-WW', up_WW_avg)
            print('cos-avg-HW', up_HW_avg)
            print('cos-std-HH', up_HH_std)
            print('cos-std-WW', up_WW_std)
            print('cos-std-HW', up_HW_std)
            ##
            print('diag-avg-HW', diag_avg)
            print('diag-std-HW', diag_std)
            ##
            print('||H-M||_F^2', F_dist)
            if dset=='train':
                cos_avg_HH_train.append(up_HH_avg.item())
                cos_avg_WW_train.append(up_WW_avg.item())
                cos_avg_HW_train.append(up_HW_avg.item())
                cos_std_HH_train.append(up_HH_std.item())
                cos_std_WW_train.append(up_WW_std.item())
                cos_std_HW_train.append(up_HW_std.item())
                HM_F2_train.append(F_dist.item())
                diag_avg_HW_train.append(diag_avg.item())
                diag_std_HW_train.append(diag_std.item())
            else:
                cos_avg_HH_val.append(up_HH_avg.item())
                cos_avg_WW_val.append(up_WW_avg.item())
                cos_avg_HW_val.append(up_HW_avg.item())
                cos_std_HH_val.append(up_HH_std.item())
                cos_std_WW_val.append(up_WW_std.item())
                cos_std_HW_val.append(up_HW_std.item())
                HM_F2_val.append(F_dist.item())
                diag_avg_HW_val.append(diag_avg.item())
                diag_std_HW_val.append(diag_std.item())

    return top1.avg, cal['expected_calibration_error'] * 100


def save_checkpoint(state, is_best, model_dir):
    filename = model_dir + '/current.pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, model_dir + '/model_best.pth.tar')


def adjust_learning_rate(optimizer, epoch, config):
    """Sets the learning rate"""
    if config.cos:
        lr_min = 0
        lr_max = config.lr
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(epoch / config.num_epochs * 3.1415926535))

    else:
        epoch = epoch + 1
        if epoch <= 5:
            lr = config.lr * epoch / 5
        elif epoch > 180:
            lr = config.lr * 0.01
        elif epoch > 160:
            lr = config.lr * 0.1
        else:
            lr = config.lr

    for param_group in optimizer.param_groups:
        print('lr now:', lr)
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
