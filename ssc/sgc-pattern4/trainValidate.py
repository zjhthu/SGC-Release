# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import sparseconvnet as s
import time
import os
import math
import numpy as np
from PIL import Image
import sparseconvnet.legacy as scn
import sys
sys.path.append("../eval/")
sys.path.append("../util/")
from logger import Logger
from weighted_cross_entropy import weighted_cross_entropy
from data import SUNCG_DATA, SUNCGTestDataset
from metric import eval_suncg


def TrainValidate(model, train_dataset, test_dataset, config):

    optimizer = optim.SGD(model.parameters(),
        lr=config.initial_lr,
        momentum = config.momentum,
        weight_decay = config.weight_decay,
        nesterov=True)

    check_point_path = 'log/'+config.prefix+'-model.pth'

    if os.path.isfile(check_point_path):
        check_point = torch.load(check_point_path)
        train_iter = check_point['train_iter']
        start_epoch = int(train_iter / config.train_num) + 1
        best_acc = check_point['best_acc']
        print('Restarting at iter ' + str(train_iter) + ' from '+ check_point_path + '..')
        model.load_state_dict(check_point['state_dict'])
        optimizer.load_state_dict(check_point['optimizer'])
        logger_train = Logger('log/log_train.txt', title='ssc', resume=True)
        logger_valid = Logger('log/log_valid.txt', title='ssc', resume=True)
    else:
        train_iter = 0
        start_epoch = 1
        best_acc = -1
        if not os.path.exists('log'):
            os.mkdir('log')
        logger_train = Logger('log/log_train.txt', title='ssc')
        logger_train.set_names(['Learning Rate', 'Train Loss1', 'Train Loss2'])
        logger_valid = Logger('log/log_valid.txt', title='ssc')
        logger_valid.set_names(['SSC IOU', 'CMP IOU'])

    print('#parameters', sum([x.nelement() for x in model.parameters()]))
    
    for epoch in range(start_epoch, config.n_epochs + 1):
        model.train()
        for param_group in optimizer.param_groups:
            cur_lr = config.initial_lr * math.exp((1 - epoch) * config.lr_decay)
            param_group['lr'] = cur_lr
        print('set learning rate '+ str(cur_lr))

        for batch_idx, batch in enumerate(train_dataset['train']()):
            train_start = time.time()
            if config.use_gpu:
                batch['input'], batch['target'], batch['weight'], batch['target2'], batch['weight2'] \
                    = batch['input'].cuda(), batch['target'].cuda(), batch['weight'].cuda(), batch['target2'].cuda(), batch['weight2'].cuda()

            batch['input'].to_variable(requires_grad=True)
            batch['target'], batch['weight'], batch['target2'], batch['weight2'] \
                = Variable(batch['target']), Variable(batch['weight']), Variable(batch['target2']), Variable(batch['weight2'])
            optimizer.zero_grad()
            output = model(batch['input'], batch['input_groups'])
            output1 = output[0][:,:,config.output_offset[0]:(config.output_offset[0]+config.dataset_outputSize[0]),\
                                    config.output_offset[1]:(config.output_offset[1]+config.dataset_outputSize[1]),\
                                    config.output_offset[2]:(config.output_offset[2]+config.dataset_outputSize[2])]

            output2 = output[1][:,:,config.output_offset2[0]:(config.output_offset2[0]+config.dataset_outputSize2[0]),\
                                    config.output_offset2[1]:(config.output_offset2[1]+config.dataset_outputSize2[1]),\
                                    config.output_offset2[2]:(config.output_offset2[2]+config.dataset_outputSize2[2])]

            loss1 = weighted_cross_entropy(output1, batch['target'], batch['weight'])
            loss2 = weighted_cross_entropy(output2, batch['target2'], batch['weight2'])
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            train_time = time.time() - train_start
            print('epoch:' + str(epoch) + ', batch:' + str((train_iter) % config.train_num) + \
                ', loss1:' + str(loss1.data[0]) +', loss2:' + str(loss2.data[0]) + ', time: ' + str(train_time))
            logger_train.append([cur_lr, loss1.data[0], loss2.data[0]])
            
            train_iter = train_iter + 1

            # Check if we want to write or validation
            b_validate = (train_iter % config.train_num) == 0
            if b_validate:
                ssc_iou, cmp_iou = valid_model(model, test_dataset)
                logger_valid.append([ssc_iou, cmp_iou])
                if ssc_iou > best_acc:
                    print('Saving best model with va_res = {}'.format(ssc_iou))
                    best_acc = ssc_iou
                    torch.save({
                        'train_iter': train_iter,
                        'best_acc': best_acc,
                        'optimizer': optimizer.state_dict(),
                        'state_dict': model.state_dict()
                        }, 'log/'+config.prefix+'-best_model.pth')

            b_save = (train_iter % 200) == 0
            if b_save:
                torch.save({
                    'train_iter': train_iter,
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                    'state_dict': model.state_dict()
                    }, check_point_path)
            if train_iter % config.train_num == 0:
                break

def valid_model(model, test_dataset):
    model.eval()
    predictions = []
    output_offset = test_dataset.output_offset
    dataset_outputSize = test_dataset.dataset_outputSize

    for idx in range(len(test_dataset)):
        test_input = test_dataset[idx]
        input = test_input['input']
        input_groups = test_input['input_groups']
        input = input.cuda()
        output = model(input, input_groups)
        predictions.append(output[0].cpu().data.numpy()[:,:,output_offset[0]:(output_offset[0]+dataset_outputSize[0]),\
                                                           output_offset[1]:(output_offset[1]+dataset_outputSize[1]),\
                                                           output_offset[2]:(output_offset[2]+dataset_outputSize[2])])
    predictions = np.vstack(predictions)
    # import h5py
    # fp = h5py.File('predictions.msg', "w")
    # result = fp.create_dataset("result", predictions.shape, dtype='f')
    # result[...] = predictions
    # fp.close()
    return eval_suncg(predictions)


