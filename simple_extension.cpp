#include <torch/extension.h>

torch::Tensor forward(torch::Tensor input) {

	// Compute x^2 + y^2 + ...
	return input.dot(input);
}

torch::Tensor backward(torch::Tensor input) {

	// Return the gradient
	return 2*input;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Simple extension forward");
  m.def("backward", &backward, "Simple extension backward");
}

