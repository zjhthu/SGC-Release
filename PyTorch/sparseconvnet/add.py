# by zjh, ILC, 2018.01.03

import torch
from torch.autograd import Variable
from torch.nn import Module
import sparseconvnet as scn
from .sparseConvNetTensor import SparseConvNetTensor

def Add_fun(input1, input2):
    #input1 and input2 has the same positions
    output = SparseConvNetTensor()
    output.metadata =  input1.metadata
    output.spatial_size = input1.spatial_size
    output.features = input1.features + input2.features
    return output

def Add2_fun(input1, input2):
    # output position is the same as input2
    output = SparseConvNetTensor()
    output.metadata = input2.metadata
    output.spatial_size = input2.spatial_size
    input1_features = Variable(torch.zeros(input2.features.size()).cuda())
    idxs = input2.getLocationsIndexInRef(input1).cuda()
    hit = (idxs != -1).nonzero().view(-1)
    input1_features[hit] = input1.features[idxs[hit]]
    output.features = input1_features + input2.features
    return output
