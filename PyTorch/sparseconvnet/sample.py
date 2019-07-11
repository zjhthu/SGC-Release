import torch
from torch.autograd import Variable
from torch.nn import Module
import sparseconvnet as scn
import time

# def Subset(dimension, input, subset):



def Sample(dimension, x, sample):
	locations = x.getSpatialLocations()
	# sample = (torch.Tensor(locations.size(0)).uniform_(0,1) < sample_p).float()
	sample_idx = sample.nonzero().view(-1)
	sample_locations = torch.index_select(locations, 0, sample_idx)
	sample_features = torch.index_select(x.features, 0, Variable(sample_idx.cuda(), requires_grad=False))
	x_sample = scn.InputBatch(dimension, x.getSpatialSize())
	x_sample.setLocations(sample_locations, torch.Tensor(sample_features.data.shape))
	x_sample.features = sample_features
	# x.locations = locations
	return x_sample

def SampleLocation(dimension, x, sample, sptialSize):
	locations = x.getSpatialLocations(sptialSize)
	sample_idx = sample.nonzero().view(-1)
	sample_locations = torch.index_select(locations, 0, sample_idx)
	x_sample = scn.InputBatch(dimension, x.getSpatialSize())
	x_sample.setLocations(sample_locations, torch.Tensor(sample_locations.size(0),1))
	return x_sample

def SampleFeature(x, x_sample, sample_flag):
	sample_idx = sample_flag.nonzero().view(-1)
	sample_features = torch.index_select(x.features, 0, Variable(sample_idx.cuda(), requires_grad=False))
	x_sample.features = sample_features
	return x_sample

