import torch
from torch.autograd import Variable
from torch.nn import Module
import sparseconvnet
from torch.autograd import Function, Variable
from torch.nn import Module, Parameter
from .utils import *
from .sparseConvNetTensor import SparseConvNetTensor


def upSampleLocations(input, filter_size=2, filter_stride=2):
	#locations1 = input.getSpatialLocations()
	#print(locations1)
	input.getConvRules2AndOutput(input.spatial_size, 2, 2)
	input.spatial_size = input.spatial_size*2
	input.features = torch.Tensor(input.features.shape[0]*8)
	#locations2 = input.getSpatialLocations()
	#print(locations2)
	return input
