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
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from networks.vnet import VNet
from utils.losses import dice_loss
from dataloaders.la_heart import LAHeart, RandomCrop, CenterCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler
from test_LA import val_calculate_metric

######## lable num #########
num_labeled = 80
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/Training_Set/', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='vnet_supervisedonly_dp/labeled_80', help='model_name')
parser.add_argument('--max_iterations', type=int,  default=6000, help='maximum epoch number to train')    # max_epoch = max_iterations//len(trainloader)+1
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='7', help='GPU to use')
args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "../model/" + args.exp + "/"

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

patch_size = (112, 112, 80)  # 裁剪的大小
num_classes = 2

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

    ################ 定义神经网络 ##############
    net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
    net = net.cuda()

    ############## 定义数据，并且进行数据增强：翻转、裁剪，使用图像增广训练模型 ############
    db_train = LAHeart(base_dir=train_data_path,
                       split='train',
                       num=num_labeled,
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
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    ############### 加载数据 #################
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    print(len(trainloader))


    net.train()
    # 定义优化器
    optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))

    ################# 开始训练 ##################
    iter_num = 0
    max_epoch = max_iterations//len(trainloader)+1
    lr_ = base_lr
    net.train()
    for epoch_num in tqdm(range(max_epoch), ncols=70):  # tqdm用来显示进度条
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()
            # print('fetch data cost {}'.format(time2-time1))
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            outputs = net(volume_batch)

            # 定义损失函数
            loss_seg = F.cross_entropy(outputs, label_batch)
            outputs_soft = F.softmax(outputs, dim=1)
            loss_seg_dice = dice_loss(outputs_soft[:, 1, :, :, :], label_batch == 1)
            loss = 0.5*(loss_seg+loss_seg_dice)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num = iter_num + 1
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss_seg', loss_seg, iter_num)
            writer.add_scalar('loss/loss_seg_dice', loss_seg_dice, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))
            if iter_num % 50 == 0:
                image = volume_batch[0, 0:1, :, :, 20:61:10].permute(3,0,1,2).repeat(1,3,1,1)
                #  make_grid的作用是将若干幅图像拼成一幅图像，padding=5
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                outputs_soft = F.softmax(outputs, 1)
                image = outputs_soft[0, 1:2, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label', grid_image, iter_num)

                image = label_batch[0, :, :, 20:61:10].unsqueeze(0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_label', grid_image, iter_num)

            ## change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            # if iter_num % 1000 == 0:
            #     save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
            #     torch.save(net.state_dict(), save_mode_path)
            #     logging.info("save model to {}".format(save_mode_path))

            if iter_num > max_iterations:
                break
            time1 = time.time()
        if iter_num > max_iterations:
            break

    metric = val_calculate_metric(net.state_dict())
    ff_path = os.path.join(snapshot_path, 'test_metric.txt')
    ff = open(ff_path, 'w')
    ff.write(str(metric))
    ff.close()
    print(metric)

    save_mode_path = os.path.join(snapshot_path, 'iter_'+str(max_iterations)+'.pth')
    torch.save(net.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()

