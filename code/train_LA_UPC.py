import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from networks.vnet import VNet
from dataloaders import utils
from utils import ramps, losses
from dataloaders.la_heart import LAHeart, RandomCrop, CenterCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler


######## lable num #########
num_labeled = 16
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/Training_Set/', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='UAMT-ada4-ema/decay-0.9999', help='model_name')
parser.add_argument('--max_iterations', type=int,  default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')

parser.add_argument('--pseudo', action='store_true', default=True, help='generate the pseudo label')
parser.add_argument('--pseudo_rect', action='store_true', default=False, help='Rectify the pseudo label')
parser.add_argument('--threshold', type=float, default=0.90, help='pseudo label threshold')
parser.add_argument('--T', type=float, default=1)
parser.add_argument('--ratio', type=float,  default=0.20, help='model noise ratio')
parser.add_argument('--dropout_rate', type=float,  default=0.9)
### costs
parser.add_argument('--ema_decay', type=float,  default=0.9999, help='ema_decay')
parser.add_argument('--consistency', type=float,  default=1.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float,  default=40.0, help='consistency_rampup')
args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "../model/" + args.exp + "/"


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

num_classes = 2
patch_size = (112, 112, 80)

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def update_variance(pred1, pred2, loss_origin):
    sm = nn.Softmax(dim=1)
    log_sm = nn.LogSoftmax(dim=1)
    kl_distance = nn.KLDivLoss(reduction='none')

    # 用loss_kl 近似等于 variance
    loss_kl = torch.sum(kl_distance(log_sm(pred1), sm(pred2)), dim=1)  # pred1 是student model, 被指导
    exp_loss_kl = torch.exp(-loss_kl)
    # print(variance.shape)
    # print('variance mean: %.4f' % torch.mean(exp_variance[:]))
    # print('variance min: %.4f' % torch.min(exp_variance[:]))
    # print('variance max: %.4f' % torch.max(exp_variance[:]))
    loss_rect = torch.mean(loss_origin * exp_loss_kl) + torch.mean(loss_kl)
    return loss_rect

def update_consistency_loss(pred1, pred2):
    if args.pseudo:
        criterion = nn.CrossEntropyLoss(reduction='none')
        # 用pred2生成伪标签
        pseudo_label = torch.softmax(pred2.detach() / args.T, dim=1)  # T：向前传播次数
        max_probs, targets = torch.max(pseudo_label, dim=1)    # 概率和标签下标
        # print(targets.shape)
        if args.pseudo_rect:    # 利用两个预测值的方差，对伪标签进行修正
            # Crossentropyloss作为损失函数时，iutput应该是[batchsize, n_class, h, w, d]，target是[batchsize, h, w, d]
            loss_ce = criterion(pred1, targets)  # 输出shape [batch, h, w, d]
            # print(pred1.shape, targets.shape)
            loss = update_variance(pred1, pred2, loss_ce)
        else:
            mask = max_probs.ge(args.threshold).float()  # 大于等于阈值
            loss_ce = criterion(pred1, targets)
            loss = torch.mean(loss_ce * mask)
            # print(loss)
    else:
        criterion = nn.MSELoss(reduction='none')
        loss_mse = criterion(pred1, pred2)
        loss = torch.mean(loss_mse)

    return loss


if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    def create_model(ema=False, has_dropout=False):
        # Network definition
        if has_dropout:
            net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True,
                       dropout_rate=args.dropout_rate)
        else:
            net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=False)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model(has_dropout=True)  # student model
    ema_model = create_model(ema=True, has_dropout=True)  # teacher model

    db_train = LAHeart(base_dir=train_data_path,
                       split='train',
                       transform = transforms.Compose([
                          RandomRotFlip(),
                          RandomCrop(patch_size),
                          ToTensor(),
                          ]))
    db_test = LAHeart(base_dir=train_data_path,
                       split='test',
                       transform = transforms.Compose([
                           CenterCrop(patch_size),
                           ToTensor()
                       ]))
    labeled_idxs = list(range(num_labeled))
    unlabeled_idxs = list(range(num_labeled, 80))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)


    model.train()
    ema_model.train()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations//len(trainloader)+1
    lr_ = base_lr
    # 只有student model在向前传播训练
    model.train()
    time1 = time.time()
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            ratio = args.ratio
            noise = torch.clamp(torch.randn_like(volume_batch) * ratio, -(2 * ratio), (2 * ratio))
            # student model + noise, teacher不加noise
            student_inputs = volume_batch + noise
            ema_inputs = volume_batch
            outputs = model(student_inputs)  # student model 在训练
            with torch.no_grad():
                ema_output = ema_model(ema_inputs)

            ## calculate the loss
            # the labled data loss
            loss_seg = F.cross_entropy(outputs[:labeled_bs], label_batch[:labeled_bs])   # 只取labeld的output
            outputs_soft = F.softmax(outputs, dim=1)
            loss_seg_dice = losses.dice_loss(outputs_soft[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)
            supervised_loss = 0.5 * (loss_seg + loss_seg_dice)  # only on labeled data
            # print('************ supervised loss:{}'.format(supervised_loss))

            # 计算consisitency loss
            consistency_loss = update_consistency_loss(outputs, ema_output)
            # print('************ consisitncy loss:{}'.format(consistency_loss))

            consistency_weight = get_current_consistency_weight(iter_num // 150)
            loss = supervised_loss + consistency_weight * consistency_loss
            # print('************ Total loss:{}'.format(loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 将studnet model 的参数更新到 teacher model
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            iter_num = iter_num + 1
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            writer.add_scalar('loss/loss_seg', loss_seg, iter_num)
            writer.add_scalar('loss/loss_seg_dice', loss_seg_dice, iter_num)
            writer.add_scalar('loss/supervised_loss', supervised_loss, iter_num)
            writer.add_scalar('train/consistency_loss', consistency_loss, iter_num)
            writer.add_scalar('train/consistency_weight', consistency_weight, iter_num)


            logging.info('iteration %d : loss : %f supervised_loss: %f consistency_loss: %f consistency_weight: %f' %
                         (iter_num, loss.item(), supervised_loss.item(), consistency_loss.item(), consistency_weight))

            if iter_num % 50 == 0:
                image = volume_batch[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                # image = outputs_soft[0, 3:4, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                image = torch.max(outputs_soft[0, :, :, :, 20:61:10], 0)[1].permute(2, 0, 1).data.cpu().numpy()
                image = utils.decode_seg_map_sequence(image)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label', grid_image, iter_num)

                image = label_batch[0, :, :, 20:61:10].permute(2, 0, 1)
                grid_image = make_grid(utils.decode_seg_map_sequence(image.data.cpu().numpy()), 5, normalize=False)
                writer.add_image('train/Groundtruth_label', grid_image, iter_num)

                image = volume_batch[-1, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('unlabel/Image', grid_image, iter_num)

                # image = outputs_soft[-1, 3:4, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                image = torch.max(outputs_soft[-1, :, :, :, 20:61:10], 0)[1].permute(2, 0, 1).data.cpu().numpy()
                image = utils.decode_seg_map_sequence(image)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('unlabel/Predicted_label', grid_image, iter_num)

                image = label_batch[-1, :, :, 20:61:10].permute(2, 0, 1)
                grid_image = make_grid(utils.decode_seg_map_sequence(image.data.cpu().numpy()), 5, normalize=False)
                writer.add_image('unlabel/Groundtruth_label', grid_image, iter_num)

            ## change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            if (iter_num % 1000 == 0) & (iter_num >= 6000):
                save_mode_path = os.path.join(snapshot_path, 'ada_iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            break

    time2 = time.time()
    total_time = (time2 - time1) / 3600
    print('total train time:', total_time)

    save_mode_path = os.path.join(snapshot_path, 'iter_' + str(max_iterations) + '.pth')
    torch.save(model.state_dict(), save_mode_path)

    logging.info("save model to {}".format(save_mode_path))
    writer.close()
