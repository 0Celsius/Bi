#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2020/03/09
@Author  :   jhhuang96
@Mail    :   hjh096@126.com
@Version :   1.0
@Description:   
'''

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from encoder import Encoder
from decoder import Decoder
from encoder import BiEncoder
from decoder import BiDecoder
from model import ED,ED_conv
from net_params import convlstm_encoder_params, convlstm_decoder_params, convgru_encoder_params, convgru_decoder_params,convgru_encoder_params_B,convgru_decoder_params_B
from net_params import bi_convgru_decoder_params,bi_convgru_encoder_params,bi_convgru_encoder_params_B,bi_convgru_decoder_params_B
from data.mm import MovingMNIST
import torch
from torch import nn
from torch.optim import lr_scheduler
import torch.optim as optim
import sys
from earlystopping import EarlyStopping
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter
import argparse
import torch
import datetime
import matplotlib
import glob
import matplotlib.pyplot as plt
# from data.datafile import txtdata
# from data.datafile_predlen12 import txtdata
from data.datafile_predlen12_indices import txtdata
# print(torch.cuda.is_available())
TIMESTAMP = datetime.datetime.now().strftime('%Y-%m-%d')
parser = argparse.ArgumentParser()
parser.add_argument('-clstm',
                    '--convlstm',default=False,type=bool,
                    help='use convlstm as base cell',
                    )
parser.add_argument('-cgru',
                    '--convgru',default=False,type=bool,
                    help='use convgru as base cell',
                    )
parser.add_argument('--convgru_B',default=False,type=bool)
parser.add_argument('--bi_convgru',default=False,type=bool)
parser.add_argument('--bi_convgru_B',default=True,type=bool)
parser.add_argument('--batch_size',
                    default=8,
                    type=int,
                    help='mini-batch size')
parser.add_argument('-lr', default=1e-4, type=float, help='G learning rate')
parser.add_argument('-predlen', default=24, type=int, help='predict length')
parser.add_argument('-frames_input',
                    default=10,
                    type=int,
                    help='sum of input frames')
parser.add_argument('-frames_output',
                    default=10,
                    type=int,
                    help='sum of predict frames')
parser.add_argument('-epochs', default=400, type=int, help='sum of epochs')
args = parser.parse_args()

random_seed = 1996
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.device_count() > 1:
    torch.cuda.manual_seed_all(random_seed)
else:
    torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# TIMESTAMP=datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
TIMESTAMP1='2024-03-04'
save_dir = './save_model/' + TIMESTAMP+f'batchsize={args.batch_size}-lr={args.lr}-predlen={args.predlen}-batchsize={args.batch_size}-biconvgru-B'
trainFolder = txtdata(train=True,
                          root='txtDataset',
                          # n_frames_input=args.frames_input,
                          # n_frames_output=args.frames_output,
                          # num_workers=4
                      )
validFolder = txtdata(train=False,
                      test=False,
                          root='txtDataset',
                          # n_frames_input=args.frames_input,
                          # n_frames_output=args.frames_output,
                          # num_objects=[3]
                      )
print('length of trainfolder:',len(trainFolder))
print('length of validfolder:',len(validFolder))
trainLoader = torch.utils.data.DataLoader(trainFolder,
                                          batch_size=args.batch_size,
                                          shuffle=False,
                                          num_workers=0
                                          )
validLoader = torch.utils.data.DataLoader(validFolder,
                                          batch_size=args.batch_size,
                                          shuffle=False,
                                          num_workers=0
                                          )

if args.convlstm:
    encoder_params = convlstm_encoder_params
    decoder_params = convlstm_decoder_params
if args.convgru:
    encoder_params = convgru_encoder_params
    decoder_params = convgru_decoder_params
    save_dir = './save_model/' + TIMESTAMP + f'batchsize={args.batch_size}-lr={args.lr}-predlen={args.predlen}-convgru'
if args.convgru_B:
    encoder_params = convgru_encoder_params_B
    decoder_params = convgru_decoder_params_B
    save_dir = './save_model/' + TIMESTAMP + f'batchsize={args.batch_size}-lr={args.lr}-predlen={args.predlen}-convgru_B'
if args.bi_convgru:
    encoder_params = bi_convgru_encoder_params
    decoder_params = bi_convgru_decoder_params
    save_dir = './save_model/' + TIMESTAMP + f'batchsize={args.batch_size}-lr={args.lr}-predlen={args.predlen}-biconvgru'
if args.bi_convgru_B:
    encoder_params = bi_convgru_encoder_params_B
    decoder_params = bi_convgru_decoder_params_B
    save_dir = './save_model/predlen24-1517biconvgru-B_new'



def loss_plot(losses):
    iters = range(len(losses))
    matplotlib.use('Agg')
    plt.figure()
    plt.plot(iters, losses, 'red', linewidth=2, label='train loss')
    # plt.plot(iters, self.val_loss, 'coral', linewidth=2, label='val loss')
    # try:
    #     if len(self.losses) < 25:
    #         num = 5
    #     else:
    #         num = 15

    #     plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
    #     plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
    # except:
    #     pass

    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc="upper right")
    log_dir=save_dir
    # if not os.path.exists(log_dir):
    #     os.makedirs(log_dir+save_dir)
    plt.savefig(os.path.join(log_dir, "epoch_loss.png"))

    plt.cla()
    plt.close("all")
def validloss_plot(losses):
    iters = range(len(losses))
    matplotlib.use('Agg')
    plt.figure()
    plt.plot(iters, losses, 'red', linewidth=2, label='valid loss')
    # plt.plot(iters, self.val_loss, 'coral', linewidth=2, label='val loss')
    # try:
    #     if len(self.losses) < 25:
    #         num = 5
    #     else:
    #         num = 15

    #     plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
    #     plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
    # except:
    #     pass

    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc="upper right")
    log_dir=save_dir
    # if not os.path.exists(log_dir):
    #     os.makedirs(log_dir+save_dir)
    plt.savefig(os.path.join(log_dir, "valid_loss.png"))

    plt.cla()
    plt.close("all")
def loss_plot_both(lossA,lossB):
    iters = range(len(lossA))
    matplotlib.use('Agg')
    plt.figure()
    plt.plot(iters, lossA, 'red', linewidth=2, label='train loss')
    plt.plot(iters,lossB, 'coral', linewidth=2, label='val loss')
    # try:
    #     if len(self.losses) < 25:
    #         num = 5
    #     else:
    #         num = 15

    #     plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
    #     plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
    # except:
    #     pass

    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc="upper right")

    plt.savefig(os.path.join(save_dir, "train_valid_loss.png"))

    plt.cla()
    plt.close("all")
def train():
    '''
    main function to run the training
    '''
    '''1--------切换模型时修改'''
    encoder = BiEncoder(encoder_params[0], encoder_params[1]).cuda()
    decoder = BiDecoder(decoder_params[0], decoder_params[1]).cuda()

    # encoder = Encoder(encoder_params[0], encoder_params[1]).cuda()
    # decoder = Decoder(decoder_params[0], decoder_params[1]).cuda()
    '''1--------切换模型时修改'''


    if args.convgru_B or args.convgru:
        net = ED_conv(encoder, decoder)
    elif args.bi_convgru or args.bi_convgru_B:
        net = ED(encoder, decoder)
    # print(net)
    run_dir = './runs/' + TIMESTAMP
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)
    tb = SummaryWriter(run_dir)
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=20, verbose=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net.to(device)
    cur_lr = args.lr
    filelist=glob.glob(os.path.join(save_dir, '*.tar'))
    if filelist:
        # minvloss = float('inf')
        # for filename in filelist:
        #     # load existing model
        #     vloss=filename.split('\\')[-1].split('_')[-1]
        #     vloss=vloss[:-8]
        #     minvloss = float(vloss) if float(vloss) < minvloss else minvloss
        print('==> loading existing model')
        # targetfile = os.path.join(save_dir, )
        # model_info = torch.load(save_dir+f'/checkpoint_{minvloss:.06f}.pth.tar')
        model_info = torch.load(save_dir + f'/checkpoint.pth.tar')
        print(save_dir + f'/checkpoint.pth.tar')
        net.load_state_dict(model_info['state_dict'])
        optimizer = torch.optim.Adam(net.parameters())
        optimizer.load_state_dict(model_info['optimizer'])
        cur_epoch = model_info['epoch'] + 1
        # cur_lr=model_info['save_lr']
    else:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        cur_epoch = 0
    lossfunction = nn.MSELoss().cuda()
    optimizer = optim.Adam(net.parameters(), lr=cur_lr)
    pla_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                      factor=0.5,
                                                      patience=4,
                                                      verbose=True)

    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    # mini_val_loss = np.inf
    # print('loading data ......')

    for epoch in range(cur_epoch, args.epochs + 1):
        ###################
        # train the model #
        ###################
        t = tqdm(trainLoader, leave=False, total=len(trainLoader))
        # qwertyyu=len(trainLoader)
        cur_lr = optimizer.state_dict()['param_groups'][0]['lr']
        for i, (data,labels) in enumerate(t):
            inputs = data.to(device)  # B,S,C,H,W
            label = labels.to(device)  # B,S,C,H,W
            inputs=inputs.float()
            label=label.float()
            optimizer.zero_grad()
            net.train()
            pred = net(inputs)  # B,S,C,H,W
            pred=pred[:,-args.predlen:,...]
            loss = lossfunction(pred, label)
            loss_aver = loss.item() / args.batch_size
            train_losses.append(loss_aver)
            loss.backward()
            torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=10.0)
            optimizer.step()

            t.set_postfix({
                'trainloss': '{:.6f}'.format(loss_aver),
                'epoch': '{:02d}'.format(epoch),
                'lr': '{:.8f}'.format(cur_lr)
            })
        tb.add_scalar('TrainLoss', loss_aver, epoch)
        ######################
        # validate the model #
        ######################
        with torch.no_grad():
            net.eval()
            t = tqdm(validLoader, leave=False, total=len(validLoader))
            for i, (data,labels) in enumerate(t):
                if i == 3000:
                    break
                inputs = data.to(device)
                label = labels.to(device)
                inputs=inputs.float()
                label=label.float()
                pred = net(inputs)
                pred = pred[:, -args.predlen:, ...]
                loss = lossfunction(pred, label)
                loss_aver = loss.item() / args.batch_size
                # record validation loss
                valid_losses.append(loss_aver)
                #print ("validloss: {:.6f},  epoch : {:02d}".format(loss_aver,epoch),end = '\r', flush=True)
                t.set_postfix({
                    'validloss': '{:.6f}'.format(loss_aver),
                    'epoch': '{:02d}'.format(epoch),
                    'lr':'{:.8f}'.format(cur_lr)
                })
        tb.add_scalar('ValidLoss', loss_aver, epoch)
        torch.cuda.empty_cache()
        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        with open(os.path.join(save_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(train_loss))
            f.write("\n")
        with open(os.path.join(save_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(valid_loss))
            f.write("\n")
        loss_plot(avg_train_losses)
        validloss_plot(avg_valid_losses)
        loss_plot_both(avg_train_losses, avg_valid_losses)
        epoch_len = len(str(args.epochs))
        print_msg = (f'[{epoch:>{epoch_len}}/{args.epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.6f} ' +
                     f'valid_loss: {valid_loss:.6f} '+
                     f'lr: {cur_lr:.8f}' )

        print(print_msg)
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        pla_lr_scheduler.step(valid_loss)  # lr_scheduler
        model_dict = {
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'save_lr':optimizer.state_dict()['param_groups'][0]['lr']
        }
        early_stopping(valid_loss.item(), model_dict, epoch, save_dir)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    with open(save_dir+"/avg_train_losses.txt", 'wt') as f:
        for i in avg_train_losses:
            print(i, file=f)

    with open(save_dir+"/avg_valid_losses.txt", 'wt') as f:
        for i in avg_valid_losses:
            print(i, file=f)


if __name__ == "__main__":
    train()
