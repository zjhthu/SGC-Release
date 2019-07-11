# by zjh, ILC, 2017.12.09

import os
import argparse

parser = argparse.ArgumentParser(description='SSCNET Training')
parser.add_argument('--gpu_id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--n_epochs', default=10, type=int, help='training epoch nums')
parser.add_argument('--initial_lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--lr_decay', default=0.5, type=float, help='learning rate decay')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='learning rate decay')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--use_gpu', default=True, type=bool, help='whether use gpu')
parser.add_argument('--train_num', default=10000, type=int, help='training batch number in each epoch')
parser.add_argument('--prefix', default='sscnet', type=str, help='predix for saving')
parser.add_argument('--output_offset', default=[], type=int, help='offset of the model prediction')
parser.add_argument('--dataset_outputSize', default=[], type=int, help='output size of suncg dataset')
parser.add_argument('--output_offset2', default=[], type=int, help='offset for low resolution prediction')
parser.add_argument('--dataset_outputSize2', default=[], type=int, help='output size of suncg dataset at low resolution')
parser.add_argument('--abc', default=[1,2,3], type=int, help='param of sgc')
parser.add_argument('--group_num', default=4, type=int, help='group num of sgc')

config = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id
import torch
import torch.nn as nn
import sparseconvnet as scn
import pdb
from trainValidate import TrainValidate
import torch.nn.functional as F
from data import getIterators, SUNCG_DATA, SUNCGTestDataset
from unet_add import UNet6, res, spatialGroupConv

def main():
    # Use the GPU if there is one, otherwise CPU
    dtype = 'torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor'
    nClasses = 12
    # three-dimensional SparseConvNet

    class Model(nn.Module):
        def __init__(self, sgc_config):
            nn.Module.__init__(self)
            self.stage1 = scn.Sequential().add(
               scn.ValidConvolution(3, 1, 16, 3, False))
            self.stage1_2 = scn.MaxPooling(3, 2, 2)
            self.stage2 = scn.Sequential()
            res(self.stage2, 3, 16, 64)
            self.stage2_2 = scn.MaxPooling(3, 2, 2)
            self.stage3 = UNet6(3, nClasses, sgc_config=sgc_config)
            self.densePred = scn.SparseToDense(3, nClasses)
            self.sgc_config = sgc_config

        def forward(self, x, group_x):
            x1 = spatialGroupConv(x, self.stage1, self.sgc_config[0], self.sgc_config[1], group_x[0])
            x1 = self.stage1_2(x1)

            x2 = spatialGroupConv(x1, self.stage2, self.sgc_config[0], self.sgc_config[1], group_x[1])
            x2 = self.stage2_2(x2)

            x3 = self.stage3(x2, group_x[2:])
            o1 = self.densePred(x3[0])
            o2 = self.densePred(x3[1])
            
            return [o1, o2]

    model=Model([config.abc, config.group_num])
    model.type(dtype)
    print(model)


    SUNCG = SUNCG_DATA()
    config.output_offset = SUNCG['output_offset'].tolist()
    config.dataset_outputSize = SUNCG['dataset_outputSize'].tolist()
    config.output_offset2 = SUNCG['output_offset2'].tolist()
    config.dataset_outputSize2 = SUNCG['dataset_outputSize2'].tolist()

    train_dataset = getIterators(SUNCG, config)
    test_dataset = SUNCGTestDataset(SUNCG, config)

    TrainValidate(model, train_dataset, test_dataset, config)

if __name__ == '__main__':
    main()
