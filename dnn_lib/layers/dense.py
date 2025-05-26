# Placeholder for the Dense Layer (Fully Connected Layer)
# You will define its forward pass, backward pass, and weight initialization here. 

import numpy as np

class Dense:
    def __init__(self, input_size, output_size):
        # Initialize weights. Multiplying by a small factor like 0.01 helps 
        # prevent activation outputs from becoming too large too quickly.
        self.weights = np.random.randn(output_size, input_size) * 0.01 
        self.biases = np.zeros(output_size)  # Shape (output_size,)

        self.input = None  # To store input from forward pass for use in backward pass
        self.grad_weights = None  # To store gradients of weights after backward pass
        self.grad_biases = None   # To store gradients of biases after backward pass

    def forward(self, x):
        """
        Computes the forward pass for the dense layer.
        x shape: (batch_size, input_size)
        Returns output of shape: (batch_size, output_size)
        """
        self.input = x  # Store the input for use in the backward pass
        
        # Output = X @ W.T + B
        # X: (batch_size, input_size)
        # self.weights.T: (input_size, output_size)
        # self.biases: (output_size,) - numpy broadcasts this correctly
        output = np.dot(x, self.weights.T) + self.biases
        return output

    def backward(self, upstream_gradient):
        """
        Computes the backward pass for the dense layer.
        upstream_gradient (dL/dY) shape: (batch_size, output_size)
        
        Sets self.grad_weights, self.grad_biases.
        Returns grad_input (dL/dX) of shape: (batch_size, input_size)
        """
        # upstream_gradient is dL/dY (gradient of loss w.r.t. the output of this layer)

        # 1. Calculate gradient of loss w.r.t. weights (dL/dW)
        # dL/dW = dL/dY * dY/dW 
        # dY/dW = X.T (if Y = WX, here Y = XW.T, so dL/dW = X.T @ dL/dY effectively after transpositions)
        # More directly: dL/dW_ij = sum_k (dL/dY_ki * X_kj) if W has shape (output, input)
        # upstream_gradient.T shape: (output_size, batch_size)
        # self.input (X) shape: (batch_size, input_size)
        # self.grad_weights (dL/dW) shape: (output_size, input_size)
        self.grad_weights = np.dot(upstream_gradient.T, self.input)

        # 2. Calculate gradient of loss w.r.t. biases (dL/dB)
        # dL/dB = dL/dY * dY/dB. Since dY/dB = 1 for each element.
        # We sum the upstream_gradient over the batch dimension.
        # self.grad_biases (dL/dB) shape: (output_size,)
        self.grad_biases = np.sum(upstream_gradient, axis=0)

        # 3. Calculate gradient of loss w.r.t. input (dL/dX) to pass to the previous layer
        # dL/dX = dL/dY * dY/dX
        # dY/dX = W
        # self.weights shape: (output_size, input_size)
        # grad_input (dL/dX) shape: (batch_size, input_size)
        grad_input = np.dot(upstream_gradient, self.weights)
        
        return grad_input


