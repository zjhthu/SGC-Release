import torch
from torch.autograd import Variable
from torch.nn import Module
import sparseconvnet as scn
from .sparseConvNetTensor import SparseConvNetTensor
import time


def Merge(x, sample_flag, prediction1, prediction2):
	output = SparseConvNetTensor()
	output.metadata = x.metadata
	output.spatial_size = x.spatial_size
	output.features = Variable(torch.FloatTensor(x.features.size(0), prediction1.features.size(1)).cuda())
	#output.features = torch.cat((prediction1.features,prediction2.features), 1)
	output.features[(sample_flag == 1).nonzero().view(-1).tolist(),:] = prediction1.features
	output.features[(sample_flag == 0).nonzero().view(-1).tolist(),:] = prediction2.features
	return output

