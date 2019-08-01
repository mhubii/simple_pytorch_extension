#include <torch/extension.h>
#include <math.h>

torch::Tensor forward(torch::Tensor input) {

	// Gaussian.
	torch::Tensor gaussian = torch::zeros({32, 32});

	for (int32_t x = 0; x < gaussian.size(0); ++x) {
		for (int32_t y = 0; y < gaussian.size(1); ++y) {

			gaussian[x][y] = torch::exp(-0.05*((x - input[0]).pow(2)+(y - input[1]).pow(2)));
		}
	}

	return gaussian;
}

torch::Tensor backward(torch::Tensor input, torch::Tensor grad_output) {

	// Derivative of gaussian.
	torch::Tensor output = torch::zeros({2});

	for (int32_t x = 0; x < 32; ++x) {
		for (int32_t y = 0; y < 32; ++y) {

			// d/dx
			output[0] += grad_output[x][y]*(x - input[0])*torch::exp(-0.05*((x - input[0]).pow(2)+(y - input[1]).pow(2)));

			// d/dy
			output[1] += grad_output[x][y]*(y - input[1])*torch::exp(-0.05*((x - input[0]).pow(2)+(y - input[1]).pow(2)));
		}
	}

	return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "gaussian extension forward");
  m.def("backward", &backward, "gaussian extension backward");
}
