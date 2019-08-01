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
	mu_pred = torch.full([2], 8, requires_grad=True) # predicted mean
	co_pred = torch.eye(2, requires_grad=True)        # predicted covariance
	mu_gt = np.array([16, 16])                        # ground truth mean
	
	gaussian = torch.zeros([32, 32])

	for x in range(32):
		for y in range(32):
	 			a = torch.full([1], -0.05*((x - mu_gt[0])**2+(y - mu_gt[1])**2))
	 			gaussian[x, y] = a.exp()

	opt = optim.Adam([mu_pred], lr=0.1)
	criterion = nn.MSELoss()

	# run for some iterations
	for i in range(200):

		output = ge.forward(mu_pred)

		opt.zero_grad()
		loss = criterion(output, gaussian)	
		loss.backward()
		opt.step()
		
		# store progress
		if (i % 10 == 0):
			print(i)
			print("Loss: ", loss.item())
			pred = ge.forward(mu_pred).detach().numpy()
			gt = gaussian.numpy()

			plt.suptitle('Gaussian Models at Iteration {}'.format(i))
			plt.subplot(121)
			plt.title('Predicted')
			plt.imshow(pred)
			plt.subplot(122)
			plt.title('Ground Truth')
			plt.imshow(gt)

			plt.savefig('img/output_iteration_{}.png'.format(i))	
			plt.clf()		




	

