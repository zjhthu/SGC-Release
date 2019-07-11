import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.autograd import Variable

# reference https://github.com/pytorch/pytorch/issues/563

def weighted_cross_entropy(output, target, weight=None):
	# ouput n c * 
	# target n *
	# weight n *

	output_size = output.size()
	n, c = output_size[0], output_size[1]
	if weight is None:
		weight = Variable((output.data.abs().sum(dim=1) > 0).float())
	output = output.contiguous().view(n,c,-1).transpose(1,2).contiguous().view(-1,c)
	logp = F.log_softmax(output)
	try:
	    logpy = torch.gather(logp, 1, target.view(-1,1))
	except Exception as e:
            print(e)
            print(logp)
            print(target.view(-1,1))
            print(target.sum())
            exit()
	losses = -logpy*(weight.view(-1,1))
	return losses.sum() / weight.sum()

