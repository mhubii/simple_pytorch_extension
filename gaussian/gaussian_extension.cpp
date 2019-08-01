#include <torch/extension.h>
#include <math.h>

float sigma = 5; // in units of pixel
int64_t size = 32;

torch::Tensor forward(torch::Tensor input) {

	// Gaussian.
	torch::Tensor gaussian = torch::zeros({size, size});

	for (int64_t x = 0; x < size; ++x) {
		for (int64_t y = 0; y < size; ++y) {

			gaussian[x][y] = 1/sqrt(2*M_PI*sigma)*torch::exp(-0.5/std::pow(sigma, 2)*((x - input[0]).pow(2)+(y - input[1]).pow(2)));
		}
	}

	return gaussian;
}

torch::Tensor backward(torch::Tensor input, torch::Tensor grad_output) {

	// Derivative of gaussian.
	torch::Tensor output = torch::zeros({2});

	for (int64_t x = 0; x < grad_output.size(0); ++x) {
		for (int64_t y = 0; y < grad_output.size(1); ++y) {

			// d/dx
			output[0] += grad_output[x][y]/(sqrt(2*M_PI)*std::pow(sigma, 3./2.))*(x - input[0])*torch::exp(-0.5/std::pow(sigma, 2)*((x - input[0]).pow(2)+(y - input[1]).pow(2)));

			// d/dy
			output[1] += grad_output[x][y]/(sqrt(2*M_PI)*std::pow(sigma, 3./2.))*(y - input[1])*torch::exp(-0.5/std::pow(sigma, 2)*((x - input[0]).pow(2)+(y - input[1]).pow(2)));
		}
	}

	return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "gaussian extension forward");
  m.def("backward", &backward, "gaussian extension backward");
}
