# by zjh, ILC, 2017.12.09

import torch
import torch.nn as nn
import sparseconvnet as scn
import time
import pdb

def res(m, dimension, a, b):
    m.add(scn.ConcatTable()
          .add(scn.Identity() if a == b else scn.NetworkInNetwork(a, b, False))
          .add(scn.Sequential()
               .add(scn.BatchNormReLU(a))
               .add(scn.SubmanifoldConvolution(dimension, a, b, 3, False))
               .add(scn.BatchNormReLU(b))
               .add(scn.SubmanifoldConvolution(dimension, b, b, 3, False))))\
     .add(scn.AddTable())

class down(nn.Module):
    def __init__(self, dimension, nPlanes_up, nPlanes_down, reps):
        super(down, self).__init__()
        self.net = scn.Sequential()
        self.net.add(scn.BatchNormReLU(nPlanes_up)).add(scn.Convolution(dimension, nPlanes_up, nPlanes_down, 2, 2,False))
        for _ in range(reps):
            res(self.net, dimension, nPlanes_down, nPlanes_down)       
    def forward(self, x):
        return self.net(x)


class flat(nn.Module):
    def __init__(self, dimension, nPlanes_up, nPlanes_down, reps):
        super(flat, self).__init__()
        self.net = scn.Sequential()
        for i in range(reps):
            res(self.net, dimension, nPlanes_up if i ==0 else nPlanes_down, nPlanes_down)
    def forward(self, x):
        return self.net(x)

class up(nn.Module):
    def __init__(self, dimension, nPlanes_down, nPlanes_up, reps, type, predict=False, nClasses=None, extract=False, kernel_size=None):
        super(up, self).__init__()
        self.dimension = dimension
        self.type = type
        self.predict = predict
        self.extract = extract
        self.kernel_size = kernel_size

        if type == 'c':
            self.upsample = scn.Sequential().add(scn.BatchNormReLU(nPlanes_down)).add(scn.DenseDeconvolution(dimension, nPlanes_down, nPlanes_up, 2, 2, False))
        elif type == 'v':
            self.upsample = scn.Sequential().add(scn.BatchNormReLU(nPlanes_down)).add(scn.Deconvolution(dimension, nPlanes_down, nPlanes_up, 2, 2, False))
        self.conv = scn.Sequential()
        for i in range(reps):
                res(self.conv, dimension, nPlanes_up, nPlanes_up)
        if predict:
            self.Linear = scn.Sequential().add(scn.BatchNormReLU(nPlanes_up)).add(scn.Linear(dimension, nPlanes_up, nClasses))
    def forward(self, down_x, x):
        down_x = self.upsample(down_x)
        if self.type == 'c':
            # add according to down stream
            x = scn.Add2_fun(x, down_x)
        elif self.type == 'v':
            x = scn.Add_fun(x, down_x)
        x = self.conv(x)
        
        prediction = []
        if self.predict:
            prediction = self.Linear(x)

        if self.extract:
            x = scn.abstract(self.dimension, x, prediction, self.kernel_size)
        return [x, prediction]

class UNet6(nn.Module):
    def __init__(self, dimension, nClasses, nPlanes = [64,64,64,64,64,64,64], reps = [2,2,2,2,2,2,2]):
        nn.Module.__init__(self)
        self.flat = flat(dimension, nPlanes[0], nPlanes[1], reps[1])
        self.down1 = down(dimension, nPlanes[1], nPlanes[2], reps[2])
        self.down2 = down(dimension, nPlanes[2], nPlanes[3], reps[3])
        self.down3 = down(dimension, nPlanes[3], nPlanes[4], reps[4])
        self.down4 = down(dimension, nPlanes[4], nPlanes[5], reps[5])
        self.down5 = down(dimension, nPlanes[5], nPlanes[6], reps[6])
        self.up5 = up(dimension, nPlanes[6], nPlanes[5], reps[5], 'v')
        self.up4 = up(dimension, nPlanes[5], nPlanes[4], reps[4], 'c')
        self.up3 = up(dimension, nPlanes[4], nPlanes[3], reps[3], 'c')
        self.up2 = up(dimension, nPlanes[3], nPlanes[2], reps[2], 'c', True, nClasses, True, 3)
        self.up1 = up(dimension, nPlanes[2], nPlanes[1], reps[1], 'c', True, nClasses, False) 

    def forward(self, x):
        x1 = self.flat(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        y5,_ = self.up5(x6, x5)
        y4,_ = self.up4(y5, x4)
        y3,_ = self.up3(y4, x3)
        y2,p2 = self.up2(y3, x2)
        y1,p1 = self.up1(y2, x1)


        return [p1,p2]
