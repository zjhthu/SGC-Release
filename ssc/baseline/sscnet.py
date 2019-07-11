# by zjh, ILC, 2017.12.09

import pdb
import os
import argparse

parser = argparse.ArgumentParser(description='SSCNET Training')
parser.add_argument('--gpu_id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--n_epochs', default=10, type=int, help='training epoch nums')
parser.add_argument('--initial_lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--lr_decay', default=0.5, type=float, help='learning rate decay')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='learning rate decay')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--train_num', default=10000, type=int, help='training batch number in each epoch')
parser.add_argument('--prefix', default='sscnet', type=str, help='predix for saving')
parser.add_argument('--output_offset', default=[], type=int, help='offset of the model prediction')
parser.add_argument('--dataset_outputSize', default=[], type=int, help='output size of suncg dataset')
parser.add_argument('--output_offset2', default=[], type=int, help='offset for low resolution prediction')
parser.add_argument('--dataset_outputSize2', default=[], type=int, help='output size of suncg dataset at low resolution')

config = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id

import torch
import torch.nn as nn
import sparseconvnet as scn
import torch.nn.functional as F
from trainValidate import TrainValidate
from data import getIterators, SUNCG_DATA, SUNCGTestDataset
from unet_add import UNet6, res

def main():
    # Use the GPU if there is one, otherwise CPU
    dtype = 'torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor'
    nClasses = 12
    # three-dimensional SparseConvNet

    class Model(nn.Module):
        def __init__(self):
            nn.Module.__init__(self)
            self.stage1 = scn.Sequential().add(
               scn.ValidConvolution(3, 1, 16, 3, False))
            self.stage1.add(scn.MaxPooling(3, 2, 2))
            res(self.stage1, 3, 16, 64)
            self.stage1.add(scn.MaxPooling(3, 2, 2))
            self.stage2 = UNet6(3, nClasses)
            self.densePred = scn.SparseToDense(3, nClasses)
        def forward(self, x):
            x1 = self.stage1(x)
            x2 = self.stage2(x1)
            o1 = self.densePred(x2[0])
            o2 = self.densePred(x2[1])
            return [o1, o2]

    model=Model()
    model.type(dtype)
    print(model)


    SUNCG = SUNCG_DATA()
    config.output_offset = SUNCG['output_offset'].tolist()
    config.dataset_outputSize = SUNCG['dataset_outputSize'].tolist()
    config.output_offset2 = SUNCG['output_offset2'].tolist()
    config.dataset_outputSize2 = SUNCG['dataset_outputSize2'].tolist()

    train_dataset = getIterators(SUNCG, config.train_num)
    test_dataset = SUNCGTestDataset(SUNCG)

    TrainValidate(model, train_dataset, test_dataset, config)

if __name__ == '__main__':
    main()
