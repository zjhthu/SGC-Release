# by zjh, ILC, 2017.11.11
import torch
import torchnet
from torch.utils.data import Dataset, DataLoader
import sparseconvnet.legacy as scn
import msgpack
import msgpack_numpy as m
import math
import random
import numpy as np
import os
import pdb
import time
import sys
sys.path.append('../suncg_data_tools/lib')
import SUNCGData
m.patch()
from unet_add import partition

# 2*2 downsample
def precomputeMetadata(input, scale, conv, batch_size, abc, group_num):
    input_size = input.getSpatialSize()
    input_groups = []
    for scale_idx in range(scale):
        spatialSize = torch.LongTensor([int(size/(2**scale_idx)) for size in input_size])
        if conv[scale_idx] == 'g':#group
            locations = input.getSpatialLocations(spatialSize)

            locations = partition(locations, abc[0], abc[1], abc[2], group_num, batch_size)

            #precompute rules for spatial group convolution to save time
            group_input = scn.InputBatch(3, spatialSize)
            group_input.setLocations(locations, torch.Tensor(len(locations)).view(-1,1))
            group_input.getValidRules(spatialSize,3)
            input_groups.append(group_input)
        else:
            input.getValidRules(spatialSize,3)
        # compute rules for downsample convolution
        input.getConvRulesAndOutput(spatialSize,2,2)
    return input_groups


def SUNCG_DATA():
    SUNCG = {}
    SUNCG['spatialSize'] = torch.LongTensor([256, 256, 256])
    SUNCG['dataset_spatialSize'] = torch.LongTensor([240,144,240])

    SUNCG['train_data_path'] = '../data/msg/train/'
    SUNCG['train_data_num'] = 139368
    SUNCG['train_batch_size'] = 4
    SUNCG['train_weight_path'] = '../data/weight/train/'
    
    
    SUNCG['output_spatialSize'] = SUNCG['spatialSize']/4
    SUNCG['input_offset'] = SUNCG['spatialSize']/2 - SUNCG['dataset_spatialSize']/2
    SUNCG['dataset_outputSize'] = SUNCG['dataset_spatialSize']/4
    SUNCG['output_offset'] = (SUNCG['output_spatialSize'] - SUNCG['dataset_spatialSize']/4)/2

    #output size of low resolution
    SUNCG['dataset_outputSize2'] = SUNCG['dataset_outputSize']/2
    SUNCG['output_offset2'] = SUNCG['output_offset']/2

    SUNCG['easy_ratio'] = 0.1
    SUNCG['neg_ratio'] = 2
    return SUNCG


def train(DATASET, config):
    spatialSize = DATASET['spatialSize']
    output_spatialSize = DATASET['output_spatialSize']
    dataset_outputSize = DATASET['dataset_outputSize']
    dataset_outputSize2 = DATASET['dataset_outputSize2']

    train_data_path = DATASET['train_data_path']
    train_data_num = DATASET['train_data_num']
    train_weight_path = DATASET['train_weight_path']
    input_offset = DATASET['input_offset']
    output_offset = DATASET['output_offset']
    train_batch_size = DATASET['train_batch_size']

    easy_ratio = DATASET['easy_ratio']
    neg_ratio = DATASET['neg_ratio']
    d = range(train_data_num)
    
    def loadData(idx):
        data = msgpack.load(open(train_data_path+str(idx)+'.msg', 'rb'))
        while np.asarray(data[b'input_nz']).size == 0: # check
            print('discard data:' + train_data_path+str(idx)+'.msg')
            idx = np.random.randint(train_data_num)
            data = msgpack.load(open(train_data_path+str(idx)+'.msg', 'rb'))
        data[b'weight'] = msgpack.load(open(train_weight_path+str(idx)+'_weight.msg', 'rb'))
        return data

    d = torchnet.dataset.ListDataset(d, load=loadData)
    randperm = torch.randperm(len(d))
    def sampler(dataset, idx):
        return randperm[idx]
    d = torchnet.dataset.ResampleDataset(d, sampler=sampler, size=config.train_num*train_batch_size)

    def perm(idx, size):
        return idx

    def merge(tbl):
        merge_time=time.time()
        input = scn.InputBatch(3, spatialSize)
        target = []
        target2 = []
        batch_weight = []
        batch_weight2 = []
        count = 0
        locations = np.empty((0,3)).astype(int)
        vals = []
        nz_nums = torch.LongTensor(train_batch_size)


        for input_nz, input_val, target_nz, target_val, weight_info in zip(tbl[b'input_nz'], tbl[b'input_val'], tbl[b'target_nz'], tbl[b'target_val'], tbl[b'weight']):
            locations = np.vstack((locations, input_nz))
            vals = np.concatenate([vals,input_val])
            nz_nums[count] = len(input_nz)
            label = np.zeros(dataset_outputSize.tolist()).astype(np.float32)
            label[target_nz.astype(int).tolist()] = target_val
            target.append(label)
            # compute downscale label
            label2 = SUNCGData.downsampleLabel2(label, 2).reshape(dataset_outputSize2.tolist())
            target2.append(label2)
            # compute weight
            weight = np.zeros(dataset_outputSize.tolist())
            weight[weight_info[b'hard_pos'].astype(int).tolist()] = 1
            hard_pos_num = weight_info[b'hard_pos'].shape[1]
            easy_pos_num = min(int(hard_pos_num*easy_ratio), weight_info[b'easy_pos'].shape[1])
            if easy_pos_num > 0:
                easy_pos_idx = np.random.permutation(weight_info[b'easy_pos'].shape[1])[:easy_pos_num]
                weight[weight_info[b'easy_pos'][:,easy_pos_idx].astype(int).tolist()] = 1
            neg_num = int(min((hard_pos_num+easy_pos_num)*neg_ratio, weight_info[b'hard_neg'].shape[1] / (1-easy_ratio)))
            hard_neg_num = int(min(neg_num * (1-easy_ratio), weight_info[b'hard_neg'].shape[1]))
            if hard_neg_num > 0:
                hard_neg_idx = np.random.permutation(weight_info[b'hard_neg'].shape[1])[:hard_neg_num]
                weight[weight_info[b'hard_neg'][:,hard_neg_idx].astype(int).tolist()] = 1
            easy_neg_num = neg_num - hard_neg_num
            if easy_neg_num > 0:
                easy_neg_idx = np.random.permutation(weight_info[b'easy_neg'].shape[1])[:easy_neg_num]
                weight[weight_info[b'easy_neg'][:,easy_neg_idx].astype(int).tolist()] = 1
            batch_weight.append(weight)

            weight2 = weight.reshape([dataset_outputSize2[0], 2, dataset_outputSize2[1], 2, dataset_outputSize2[2], 2]).mean(5).mean(3).mean(1)
            weight2[weight2 > 0] = 1
            batch_weight2.append(weight2)
            count = count + 1
        

        input.setInputBatchLocations(torch.from_numpy(locations).long()+input_offset.view(1,3), \
            torch.FloatTensor(vals).view(nz_nums.sum(),1), nz_nums)
        
        input_groups = precomputeMetadata(input, 6, ['g','g','g','g','n','n'], train_batch_size, config.abc, config.group_num)

        return {'input': input, 'target': torch.LongTensor(np.array(target).astype(int)), 'weight':torch.FloatTensor(np.array(batch_weight)), \
                                'target2': torch.LongTensor(np.array(target2).astype(int)), 'weight2':torch.FloatTensor(np.array(batch_weight2)),\
                                'input_groups':input_groups }
    bd = torchnet.dataset.BatchDataset(d, train_batch_size, perm=perm, merge=merge)
    tdi = scn.threadDatasetIterator(bd)

    def iter():
        return tdi()
    return iter

def getIterators(*args):

    return {'train': train(*args)}

class SUNCGTestDataset(Dataset):
    def __init__(self, SUNCG, config):
        self.test_data_path = '../data/msg/test/'
        self.spatialSize = SUNCG['spatialSize']
        self.input_offset = SUNCG['input_offset']
        self.precomputeStride = 2
        self.test_data_num = 470 
        self.output_offset = SUNCG['output_offset']
        self.dataset_outputSize = SUNCG['dataset_outputSize']
        self.abc = config.abc
        self.group_num = config.group_num
    def __len__(self):
        return self.test_data_num

    def __getitem__(self, idx):
        print('load ' + self.test_data_path +str(idx)+'.msg')
        data = msgpack.load(open(self.test_data_path+str(idx)+'.msg', 'rb'))
        input_nz, input_val = data[b'input_nz'], data[b'input_val']
        input = scn.InputBatch(3, self.spatialSize)
        input.addSample()
        input.setLocations(torch.from_numpy(input_nz).long()+self.input_offset.view(1,3), torch.from_numpy(input_val).view(len(input_val), 1), 0)
        input_groups = precomputeMetadata(input, 6, ['g','g','g','g','n','n'], 1, self.abc, self.group_num)
        return {'input':input, 'input_groups':input_groups}
    
    
