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

	# generate a gaussian distribution
	mu_opt = torch.full([2], 8, requires_grad=True)   # optimized mean
	mu_gt = np.array([16, 16])                        # ground truth mean
	
	size = 32
	gaussian = torch.zeros([size, size])
	sigma = 5 # in units of pixel

	for x in range(size):
		for y in range(size):
	 			a = torch.full([1], -0.5/sigma**2*((x - mu_gt[0])**2+(y - mu_gt[1])**2))
	 			gaussian[x, y] = 1/np.sqrt(2*np.pi*sigma**2)*a.exp()

	opt = optim.Adam([mu_opt], lr=0.1)
	criterion = nn.MSELoss()

	# run for some iterations
	for i in range(80):

		output = ge.forward(mu_opt)

		opt.zero_grad()
		loss = criterion(output, gaussian)	
		loss.backward()
		opt.step()
		
		# store progress
		if (i % 10 == 0):
			print(i)
			print("Loss: ", loss.item())
			gaussian_opt = ge.forward(mu_opt).detach().numpy()
			gaussian_gt = gaussian.numpy()

			plt.suptitle('Gaussian Models at Iteration {}'.format(i))
			plt.subplot(121)
			plt.title('Optimized')
			plt.imshow(gaussian_opt)
			plt.subplot(122)
			plt.title('Ground Truth')
			plt.imshow(gaussian_gt)

			plt.savefig('img/output_iteration_{}.jpg'.format(i))	
			plt.clf()		




	

