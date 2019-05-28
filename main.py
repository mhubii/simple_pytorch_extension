import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Function

# sources to read
# pytorch implementation of linear layer
# https://pytorch.org/docs/stable/notes/extending.html
#
# computation of gradient wrt to parameters
# http://cs231n.stanford.edu/handouts/linear-backprop.pdf
#
# also a nice tutorial (didnt read though)
# https://pytorch.org/tutorials/beginner/pytorch_with_examples.html

# as defined in setup.py 
import simple_extension_cpp

class SimpleExtensionFunction(Function):
	@staticmethod
	def forward(ctx, input):
		output = simple_extension_cpp.forward(input)
		ctx.save_for_backward(input)
		return output

	@staticmethod
	def backward(ctx, grad_output):
		input, = ctx.saved_tensors
		grad_input = simple_extension_cpp.backward(input)*grad_output	
		return grad_input

class SimpleExtension(nn.Module):
	def __init__(self):
		super(SimpleExtension, self).__init__()

	def forward(self, input):
		return SimpleExtensionFunction.apply(input)

if __name__ == '__main__':
	se = SimpleExtension()

	input = torch.ones(2, requires_grad=True)
	goal = torch.full([1], 3)

	opt = optim.Adam([input])
	criterion = nn.MSELoss()

	for i in range(1000):

		output = se.forward(input)

		opt.zero_grad()
		loss = criterion(output, goal)	
		loss.backward()
		opt.step()
		print("Loss: ", loss.item())

	print("Learning input:\n", input)

