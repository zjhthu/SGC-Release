import torch
from torch.autograd import Variable, Function

# reference http://pytorch.org/docs/master/notes/extending.html

class Fusion(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight):

        ctx.save_for_backward(weight)
        output = torch.mm(weight, input)     
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        weight, = ctx.saved_variables
        grad_input = grad_weight = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        grad_input = torch.mm(weight.data.t(), grad_output.data)
        grad_input = Variable(grad_input)
        return grad_input, grad_weight
