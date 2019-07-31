import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Function
import numpy as np
import matplotlib.pyplot as plt

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
import gaussian_extension_cpp

class gaussianExtensionFunction(Function):
	@staticmethod
	def forward(ctx, input):
		output = gaussian_extension_cpp.forward(input)
		ctx.save_for_backward(input)
		return output

	@staticmethod
	def backward(ctx, grad_output):
		input, = ctx.saved_tensors
		grad_input = gaussian_extension_cpp.backward(input, grad_output)
		return grad_input

class gaussianExtension(nn.Module):
	def __init__(self):
		super(gaussianExtension, self).__init__()

	def forward(self, input):
		return gaussianExtensionFunction.apply(input)

if __name__ == '__main__':
	ge = gaussianExtension()

	input = torch.full([2], 5, requires_grad=True)
	print(input)
	mu = np.array([10, 5])
	
	gaussian = torch.zeros([32, 32])

	for x in range(32):
		for y in range(32):
	 			a = torch.full([1], -0.05*((x - mu[0])**2+(y - mu[1])**2))
	 			gaussian[x, y] = 255*a.exp()

	opt = optim.Adam([input], lr=0.1)
	criterion = nn.MSELoss()

	path = []

	for i in range(100):

		print(i)
		output = ge.forward(input)

		path.append([input.detach().numpy()[0], input.detach().numpy()[1]])

		opt.zero_grad()
		loss = criterion(output, gaussian)	
		loss.backward()
		opt.step()
		print("Loss: ", loss.item())

	np.savetxt('path.csv', np.array(path))

	output = ge.forward(input).detach().numpy()
	gaussian = gaussian.numpy()

	plt.subplot(121)
	plt.title('Predicted')
	plt.imshow(output)
	plt.subplot(122)
	plt.title('Ground Truth')
	plt.imshow(gaussian)

	plt.savefig('output.png')


	

