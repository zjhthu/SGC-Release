# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from .utils import dim_fn, toLongTensor
from torch.autograd import Variable
import numpy as np
import pdb

class SparseConvNetTensor(object):
    def __init__(self, features=None, metadata=None, spatial_size=None):
        self.features = features
        self.metadata = metadata
        self.spatial_size = spatial_size
        self.locations = None

    def getSpatialLocations(self, spatial_size=None):
        "Coordinates and batch index for the active spatial locations"
        if spatial_size is None:
            spatial_size = self.spatial_size

        t = torch.LongTensor()
        dim_fn(self.metadata.dimension, 'getSpatialLocations')(self.metadata.ffi, spatial_size, t)
        return t

    def getLocationsIndexInRef(self, ref):
        "Get spatial locations index in reference metadata"
        idxs = torch.LongTensor(self.features.size(0)).fill_(-1)
        dim_fn(self.metadata.dimension, 'getLocationsIndexInRef')(self.metadata.ffi, ref.metadata.ffi, self.spatial_size, idxs)
        return idxs

    # def getRefIdx(self, ref, idx):
    #     "Get ref spatial locations index in self metadata"
    #     dim_fn(self.metadata.dimension, 'getLocationsIndexInRef')(self.metadata.ffi, ref.metadata.ffi, self.spatial_size, idxs)
    #     return idxs

    def concatTensor(self, in1, in2):
        "concat two sparse tensor"
        idx1 = torch.LongTensor(in1.features.size(0)).zero_()
        idx2 = torch.LongTensor(in2.features.size(0)).zero_()
        nActive = dim_fn(self.metadata.dimension, 'concatTensor')(self.metadata.ffi, in1.metadata.ffi, in2.metadata.ffi, self.spatial_size, idx1, idx2)
        return [idx1, idx2, nActive]

    def extractStructure(self, label, kernel_size):
        "extract locations according to label and kernel size"
        subset = torch.LongTensor(label.size(0)).zero_()
        kernel_size = toLongTensor(self.metadata.dimension, kernel_size)
        dim_fn(self.metadata.dimension, 'extractStructure')(self.metadata.ffi, self.spatial_size, label, subset, kernel_size)
        return subset

    def extractStructure2(self, p, label, kernel_size, p_threshold):
        "extract locations according to possibility and kernel_size"
        subset = torch.LongTensor(label.size(0)).zero_()
        kernel_size = toLongTensor(self.metadata.dimension, kernel_size)
        dim_fn(self.metadata.dimension, 'extractStructure2')(self.metadata.ffi, self.spatial_size, p, label, subset, kernel_size, float(p_threshold))
        return subset

    def getSampleWeight(self, sample, spatialSize=None):
        if spatialSize is None:
            spatialSize = self.spatial_size
        "compute sample weight\n sample 1:sample, 0:reserved"
        position = torch.LongTensor()
        M = (sample == 0).sum()
        N = sample.size(0)
        val = torch.FloatTensor()
        dim_fn(self.metadata.dimension, 'getSampleWeight')(self.metadata.ffi, spatialSize, sample, position, val)
        M_new = (sample == 0).sum()
        #if (M_new-M) != 0:
        #    print('add '+str(M-M_new)+'undetermined points')
        reserved_idx_map = torch.LongTensor(N).fill_(-1)
        reserved_idx_map[(sample==0).nonzero().view(-1)] = torch.LongTensor(range(M_new))
        sample_idx_map = torch.LongTensor(N).fill_(-1)
        sample_idx_map[sample.nonzero().view(-1)] = torch.LongTensor(range(N-M_new))
        # map position index
        #def idx_map (x):
        #    pdb.set_trace()
        #    return [reserved_idx_map[x[0]], sample_idx_map[x[1]]]
        #vf = np.vectorize(idx_map)
        #position_mapped = torch.Tensor(vf(position))
        position_mapped = torch.LongTensor([ [reserved_idx_map[x[0]], sample_idx_map[x[1]]] for x in position])
        weight = torch.sparse.FloatTensor(position_mapped.t(), val, torch.Size([M_new,N-M_new]))
        return [weight, sample]

    def getValidRules(self, spatialSize, filterSize):
        filterSize = toLongTensor(self.metadata.dimension, filterSize)
        dim_fn(self.metadata.dimension, 'getValidRules')(self.metadata.ffi, spatialSize, filterSize)
    
    def getConvRulesAndOutput(self, inputSize, filterSize, stride):
        assert filterSize == 2 and stride == 2
        outputSize = inputSize/2
        filterSize = toLongTensor(self.metadata.dimension, filterSize)
        stride = toLongTensor(self.metadata.dimension, stride)
        dim_fn(self.metadata.dimension, 'getConvRulesAndOutput')(self.metadata.ffi, inputSize, outputSize, filterSize, stride)

    def getConvRules2AndOutput(self, inputSize, filterSize, stride):
        assert filterSize == 2 and stride == 2
        outputSize = inputSize*2
        filterSize = toLongTensor(self.metadata.dimension, filterSize)
        stride = toLongTensor(self.metadata.dimension, stride)
        dim_fn(self.metadata.dimension, 'getConvRules2AndOutput')(self.metadata.ffi, inputSize, outputSize, filterSize, stride)


    def getSpatialSize(self):
        return self.spatial_size

    def getConvMask(self, filter_size):
       filter_size = toLongTensor(self.metadata.dimension, filter_size)
       mask = torch.FloatTensor(self.features.size(0)).fill_(0)
       dim_fn(self.metadata.dimension, 'getConvMask')(self.metadata.ffi, self.spatial_size, filter_size, mask)
       return mask

    def type(self, t=None):
        if t:
            self.features = self.features.type(t)
            return self
        return self.features.type()

    def cuda(self):
        self.features = self.features.cuda()
        return self

    def cpu(self):
        self.features = self.features.cpu()
        return self

    def set_(self):
        self.features.set_(self.features.storage_type()())
        self.metadata.set_()
        self.spatialSize = None

    def __repr__(self):
        return 'SparseConvNetTensor<<' + \
            repr(self.features) + repr(self.metadata) + repr(self.spatial_size) + '>>'

    def to_variable(self, requires_grad = False, volatile=False):
        "Convert self.features to a variable for use with modern PyTorch interface."
        self.features=Variable(self.features, requires_grad=requires_grad, volatile=volatile)
        return self
